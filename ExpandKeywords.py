import os
import json
import time
import pathlib
import tempfile
import pandas as pd
from litellm import completion
from litellm import exceptions as litellm_exception
from fuzzywuzzy import process
from typing import Literal, List, Dict, Union
from agency_swarm.tools import BaseTool

from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from pydantic import Field, model_validator, BaseModel, PrivateAttr

from tools.ConductorAgent.util.prompts import EXPAND_KEYWORDS_PROMPT
from tools.ConductorAgent.util.constants import keyword_response_schema

from dotenv import load_dotenv

load_dotenv()


# Get the current directory path
CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"
service_account_info = json.loads(os.getenv("SERVICE_ACCOUNT_KEY_ADS"))

# Create a temporary file to store the JSON key
with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
    temp_file.write(json.dumps(service_account_info).encode())
    service_account_key_path = temp_file.name

# Initialize Google Ads client
client = GoogleAdsClient.load_from_dict(
    {
        "developer_token": os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN"),
        "login_customer_id": os.getenv("GOOGLE_ADS_MANAGER_ID"),
        "use_proto_plus": True,
        "json_key_file_path": service_account_key_path,
    }
)
client.login_customer_id = os.getenv("GOOGLE_ADS_MANAGER_ID")

# Remove the temporary file
os.remove(service_account_key_path)

THREAD_TIMEOUT = int(os.getenv("THREAD_TIMEOUT"))

class Keyword(BaseModel):
    keyword: str
    match_type: Literal["BROAD", "PHRASE", "EXACT"]


class ExpandKeywords(BaseTool):
    """
    Tool that generates more keywords based on the previously scraped data from KeywordSearch tool.
    """

    num_keywords: int = Field(
        ...,
        description="Number of keywords to generate.",
    )

    model: Literal["gemini"] = Field(
        "gemini",
        description=(
            "Model to use for keyword generation. Only Gemini is supported."
        ),
    )

    location: Union[str, Literal["Worldwide"]] = Field(
        ...,
        description="Location name to search keywords for. Can be a city, state, country, etc.",
    )
    country_code: str = Field(
        None,
        description=(
            "A 2 letter country code to search keywords for. "
            "Only required if location is not 'Worldwide'."
        ),
    )
    location_type: Literal[
        "City",
        "Municipality",
        "County",
        "Region",
        "Province",
        "State",
        "Country",
        "Territory",
        "Postal Code",
    ] = Field(
        None,
        description=(
            "Location type to search keywords for. Only required if location is not 'Worldwide'."
        ),
    )

    _start_time: float = PrivateAttr(default_factory=time.time)
    _timeout: int = PrivateAttr(default=THREAD_TIMEOUT)

    @model_validator(mode="after")
    def validate_location_type(self):
        if self.location != "Worldwide" and (
            self.location_type is None or self.country_code is None
        ):
            raise ValueError(
                "Location type and country code are required if location is not 'Worldwide'."
            )

    class ToolConfig:
        one_call_at_a_time = True

    def run(self) -> Dict[str, int]:
        """
        Generates more keywords based on the previously scraped data.
        """
        website_data = self._shared_state.get("SEARCH_DATA", None)
        website_url = self._shared_state.get("URL", None)
        if self.model == "gemini":
            if not os.getenv("GEMINI_API_KEY"):
                raise ValueError("GEMINI_API_KEY is not set.")
            model = f"gemini/{GEMINI_MODEL}"
        else:
            raise ValueError(f"Invalid model: {self.model}")

        if not website_data:
            raise ValueError(
                "Website data not found. Please run KeywordSearch tool first."
            )
        try:
            # First, check if provided location exists and get its ID
            if self.location != "Worldwide":
                location_id, error_msg = self.get_location_id(
                    self.location, self.location_type, self.country_code
                )
                if error_msg:
                    raise ValueError(error_msg)
            else:
                location_id = None

            # Generate additional keywords
            results = {}
            keywords_to_save = []
            initial_keywords = self._shared_state.get("KEYWORDS", [])
            used_names = [keyword["keyword"] for keyword in initial_keywords]
            retry_count = 0
            while len(keywords_to_save) < self.num_keywords and retry_count < 5:
                self.check_timeout()
                prompt = EXPAND_KEYWORDS_PROMPT.format(
                    num_keywords=self.num_keywords + 30,
                    webpage_content=website_data,
                    keywords=used_names,
                )

                try:
                    result = completion(
                        model=model,
                        response_format=keyword_response_schema,
                        messages=[
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": (
                                    f"Please generate google ads keywords for this page: {website_url}\n"
                                    "Return answer in JSON format:\n"
                                    "keyword = {'keyword': str, 'match_type': Literal['BROAD', 'PHRASE', 'EXACT']}\n"
                                    "Return: {'keywords': list[keyword]}"
                                ),
                            },
                        ],
                    )
                except litellm_exception.APIError as e:
                    print(f"Error make completion request: {e}\n\nRetrying...")
                    retry_count += 1
                    continue
                try:
                    keywords = json.loads(result.choices[0].message.content)["keywords"]
                except json.JSONDecodeError:
                    try:
                        keywords = json.loads(
                            result.choices[0]
                            .message.content.split("```")[1]
                            .strip("json")
                        )["keywords"]
                    except:
                        print("Invalid response format, retrying...")
                        retry_count += 1
                        continue

                keywords = [
                    {
                        "keyword": keyword["keyword"].lower(),
                        "match_type": keyword["match_type"],
                    }
                    for keyword in keywords
                ]
                keyword_names = [keyword["keyword"] for keyword in keywords]

                # Get the keyword plan ideas service
                keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
                # Generate keyword ideas
                request = self.get_search_request_metrics(
                    location_id=location_id, keywords=keyword_names
                )
                # Get keyword metrics
                keyword_metrics = keyword_plan_idea_service.generate_keyword_historical_metrics(
                    request=request).results

                # Process results
                for keyword_data in keyword_metrics:
                    keyword = keyword_data.text

                    if keyword in used_names or keyword not in keyword_names:
                        continue

                    avg_monthly_searches = keyword_data.keyword_metrics.avg_monthly_searches
                    if avg_monthly_searches > 15:
                        results[keyword] = {
                            "avg_monthly_searches": avg_monthly_searches,
                            "match_type": keywords[keyword_names.index(keyword)][
                                "match_type"
                            ],
                        }
                        keywords_to_save.append(
                            {
                                "keyword": keyword,
                                "match_type": keywords[
                                    keyword_names.index(keyword)
                                ]["match_type"],
                            }
                        )
                # Add all names to the list to avoid generating duplicates
                used_names.extend(keyword_names)
                retry_count += 1

            results = dict(
                sorted(
                    results.items(),
                    key=lambda x: x[1]["avg_monthly_searches"],
                    reverse=True,
                )
            )
            keywords_to_save.sort(
                key=lambda x: results[x["keyword"]]["avg_monthly_searches"],
                reverse=True,
            )

            updated_keywords = initial_keywords + keywords_to_save[: self.num_keywords]
            self._shared_state.set("KEYWORDS", updated_keywords)

            if len(keywords_to_save) < self.num_keywords:
                return (
                    "Keywords have been generated successfully, however less than requested amount "
                    "was generated due to monthly search threshold. "
                    "You do not need to save the new list, new keywords "
                    "have been added to the previous list. "
                    "Notify the user about the issue and provide it with the updated list. "
                    f"Here's the list of additional keywords: \n{dict(list(results.items())[:self.num_keywords])}"
                )

            return (
                "Keywords have been generated successfully. "
                "You do not need to save the new list, new keywords "
                "have been added to the previous list."
                "Provide user with the updated list of keywords. "
                f"Here's the list of additional keywords: \n{dict(list(results.items())[:self.num_keywords])}"
            )

        except GoogleAdsException as ex:
            error_message = []
            for error in ex.failure.errors:
                error_message.append(f"Error: {error.message}")
            raise ValueError(" | ".join(error_message))

    def map_locations_ids_to_resource_names(self, client, location_ids):
        """Converts a list of location IDs to resource names.

        Args:
            client: an initialized GoogleAdsClient instance.
            location_ids: a list of location ID strings.

        Returns:
            a list of resource name strings using the given location IDs.
        """
        build_resource_name = client.get_service(
            "GeoTargetConstantService"
        ).geo_target_constant_path
        return [build_resource_name(location_id) for location_id in location_ids]

    def get_search_request_metrics(
        self,
        location_id: str,
        keywords: List[str],
    ):
        """
        Get the keyword search request.
        """
        if location_id:
            location_rns = self.map_locations_ids_to_resource_names(
                client, [location_id]
            )

        language_rn = client.get_service("GoogleAdsService").language_constant_path(
            "1000"
        )
        request = client.get_type("GenerateKeywordHistoricalMetricsRequest")
        request.customer_id = os.getenv("GOOGLE_ADS_MANAGER_ID")
        request.keywords = keywords
        request.keyword_plan_network = (
            client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
        )
        request.language = language_rn
        if location_id:
            request.geo_target_constants = location_rns

        return request

    def get_location_id(
        self, location_name: str, location_type: str, country_code: str
    ) -> tuple[str, str]:
        """
        Get location ID from CSV file. If location not found, suggest similar locations.
        Returns tuple of (location_id, error_message). Error message is None if location found.
        """
        # Read CSV file - adjust path as needed
        df = pd.read_csv(f"{CURRENT_DIR}/util/geotargets-2024-10-10.csv")
        # Filter dataframe to only include rows matching the specified location type
        df = df[df["Target Type"] == location_type]
        df = df[df["Country Code"] == country_code]
        if df.empty:
            return None, f"No `{location_type}` locations found in {country_code}"

        # Check if city exists exactly
        city_match = df[df["Name"].str.lower() == location_name.lower()]
        if not city_match.empty:
            return city_match.iloc[0]["Criteria ID"], None

        # If no exact match, find closest matches
        closest_matches = process.extract(location_name, df["Name"], limit=15)
        suggestions = [match[0] for match in closest_matches]
        suggestions = list(dict.fromkeys(suggestions))  # Drop duplicating city names
        error_msg = f"Location '{location_name}' not found. Did you mean one of these? {', '.join(suggestions[:5])}"
        return None, error_msg
    
    def check_timeout(self):
        if time.time() >= self._start_time + self._timeout:
            raise TimeoutError("Operation timed out")


if __name__ == "__main__":
    # Test the ExpandKeywords tool
    print("Testing ExpandKeywords tool...")
    

    expand_keywords = ExpandKeywords(
        num_keywords=5,
        model="gemini",
        location="Virginia",
        location_type="State",
        country_code="US",
    )
        
    # Run ExpandKeywords
    expand_keywords._shared_state.set("SEARCH_DATA", """
undefined

[](https://www.dlapiper.com/)

[Skip to main content](#mainContainer)

United States|en-US

  * [People](/en-us/people)
  * [Capabilities](/en-us/capabilities)
  * [About us](/en-us/about-us)
  * [Insights](/en-us/insights/featured-insights)
  * [Careers](/en-us/careers)
  *     * [Locations](/en-us/locations "Locations")
    * [News](/en-us/news "News")
    * [Events](/en-us/events "Events")
    * [Blogs](/en-us/insights/blogs "Blogs")
    * [Alumni](/en-us/alumni "Alumni")
    * [Pro bono](/en-us/about-us/pro-bono "Pro bono")
  * United States|en-US




Add a bookmark to get started

[Bookmarks info](/en-us/bookmarks)

[Global Site](/en)

Africa

Morocco[English](/en-MA)

South Africa[English](/en-ZA)

Asia Pacific

Australia[English](/en-AU)

China[English](/en-CN)[简体中文](/zh-CN)

Hong Kong SAR China[English](/en-HK)[简体中文](/zh-HK)

Japan[English](/en-JP)[日本語](/ja-JP)

Korea[English](/en-KR)

New Zealand[English](/en-NZ)

Singapore[English](/en-SG)

Thailand[English](/en-TH)

Europe

Austria[English](/en-AT)[Deutsch](/de-AT)

Belgium[English](/en-BE)

Czech Republic[English](/en-CZ)

Denmark[English](https://denmark.dlapiper.com/en)[Dansk](https://denmark.dlapiper.com/da)

Finland[English](https://finland.dlapiper.com/en/landing/dla-piper-finland)[Suomi](https://finland.dlapiper.com/en/landing/dla-piper-finland)

France[English](/en-FR)[Français](/fr-FR)

Germany[English](/en-DE)[Deutsch](/de-DE)

Hungary[English](/en-HU)

Ireland[English](/en-IE)

Italy[English](/en-IT)[Italiano](/it-IT)

Luxembourg[English](/en-LU)

Netherlands[English](/en-NL)

Norway[English](https://norway.dlapiper.com/en/landing/dla-piper-norway)[Norsk](https://norway.dlapiper.com/no/landing/dla-piper-norway)

Poland[English](/en-PL)

Portugal[English](/en-PT)

Romania[English](/en-RO)

Slovak Republic[English](/en-SK)

Spain[English](/en-ES)[Español](/es-ES)

Sweden[English](https://sweden.dlapiper.com/en/landing/dla-piper-sweden)[Svenska](https://sweden.dlapiper.com/sv/landing/dla-piper-sverige)

United Kingdom[English](/en-GB)

Latin America

Argentina[English](/en-AR)[Español](/es-AR)

Brazil[English](/en-BR)

Chile[English](/en-CL)[Español](/es-CL)

Mexico[English](/en-MX)[Español](/es-MX)

Peru[English](/en-PE)[Español](/es-PE)

Puerto Rico[English](https://www.dlapiper.com/en-pr)[Español](/es-PR)

Middle East

Bahrain[English](/en-BH)

Oman[English](/en-OM)

Qatar[English](/en-QA)

Saudi Arabia[English](/en-SA)

UAE[English](/en-AE)

North America

Canada[English](/en-CA)[Français](/fr-CA)

Puerto Rico[English](/en-PR)

United States[English](/en-US)

OtherForMigration

Latest news

and insights

[NewsDLA Piper advises LLR Partners in strategic growth investment in TruTechnologies™![](/-/media/project/dlapiper-tenant/dlapiper/editorialdefaultimages/abstract-building-cropped.jpg?rev=-1?w=3840&q=75)](/en-us/news/2025/05/dla-piper-advises-llr-partners-in-strategic-growth-investment-in-trutechnologies)

[PublicationMake Our Children Healthy Again Assessment: Unpacking the report![](/-/media/project/dlapiper-tenant/dlapiper/editorialdefaultimages/colorful-drinks-2.jpg?rev=-1?w=3840&q=75)](/en-us/insights/publications/2025/05/maha-making-our-children-healthy-again-assessment)

[NewsDLA Piper advises Axsome Therapeutics on US$570 million term loan and credit facility with...![](/-/media/project/dlapiper-tenant/dlapiper/editorialdefaultimages/abstract_architectural_shapes_p_0044crop.jpg?rev=-1?w=3840&q=75)](/en-us/news/2025/05/dla-piper-advises-axsome-therapeutics-on-usd570-million-term-loan-and-credit-facility)

# Success,solved.

Tell us where you want to be tomorrow. 

Our people will get you there with cutting-edge legal and commercial insight.

## Global

## 

G

l

o

b

a

l

Wherever business takes you, our insight gives you an advantage.

![](/-/media/project/dlapiper-tenant/dlapiper/insights/publications/horizons-newsletter-mockup/horizon---chinese-national-park-at-sunset.jpg?h=975&iar=0&w=2560&rev=-1&hash=8215B6F20915725E32014D538416DBF0)

Horizon: Bite-sized coverage of legislative and policy developments in sustainability

[Read more](/en-us/insights/publications/horizon/2025/horizon-esg-regulatory-news-and-trends-may-2025)

![](/-/media/project/dlapiper-tenant/dlapiper/editorialdefaultimages/website_hero_abstact_architectural_ceiling_p_0089_mono.jpg?h=975&iar=0&w=2560&rev=-1&hash=F50EDE3C39ABF0A800499ED0CF67A638)

M&A market recovers in 2024 despite headwinds

[Read more](/en-us/news/2025/05/manda-market-recovers-in-2024-despite-headwinds)

## Visionary

## 

V

i

s

i

o

n

a

r

y

However bold your ambition, you can trust we’ll make it happen.

![](/-/media/project/dlapiper-tenant/dlapiper/editorialdefaultimages/abstract_architectural_red_wave_p_0056.jpg?h=975&iar=0&w=2560&rev=-1&hash=393621286EA416011D39879557DC8289)

SEC emphasizes focus on "AI washing" despite perceived enforcement slowdown

[Read more](/en-us/insights/publications/ai-outlook/2025/sec-emphasizes-focus-on-ai-washing)

![](/-/media/project/dlapiper-tenant/dlapiper/editorialdefaultimages/stars_s_2651c.jpg?h=975&iar=0&w=2650&rev=-1&hash=CCF0163DB3290E1F78A956052FA068C3)

The Trump Administration commits to maintaining the National Space Council

[Read more](/en-us/insights/publications/interstellar-insights/2025/interstellar-insights-may-2025)

![](/-/media/project/dlapiper-tenant/dlapiper/editorialdefaultimages/green-tunnel.jpg?h=975&iar=0&w=2560&rev=-1&hash=78CFDD1F5CC779AC95E5D8DF428575AB)

Causing collusion? Understanding antitrust considerations when using artificial intelligence

[Register for our webinar](/en-us/events/2025/06/causing-collusion-understanding-antitrust-considerations-when-using-artificial-intelligence)

![](/-/media/project/dlapiper-tenant/dlapiper/editorialdefaultimages/capitol_daytime.jpg?h=975&iar=0&w=2560&rev=-1&hash=AE0A7CF8713075D684CA061FEFF12C11)

Office Hours with Senator Richard Burr

[Register for our webinar](/en-us/events/office-hours-with-senator-richard-burr)

## Partners

## 

P

a

r

t

n

e

r

s

Whatever your destination, we’re by your side on the journey.

![](/-/media/project/dlapiper-tenant/dlapiper/editorialdefaultimages/colorful-drinks-2.jpg?h=975&iar=0&w=2560&rev=-1&hash=98C73B7B61B41D55A9F8185A722A42B8)

Make Our Children Healthy Again Assessment: Unpacking the report

[Read more](/en-us/insights/publications/2025/05/maha-making-our-children-healthy-again-assessment)

![](/-/media/project/dlapiper-tenant/dlapiper/editorialdefaultimages/abstract_modern_architecture_n_2364-x3.jpg?h=975&iar=0&w=2560&rev=-1&hash=8D2E938F369E4A650BD322D339043910)

House passes sweeping tax bill: Top points for the investment funds industry

[Read more](/en-us/insights/publications/2025/05/house-passes-sweeping-tax-bill)

![](/-/media/project/dlapiper-tenant/dlapiper/editorialdefaultimages/abstract_building_p_0053.jpg?h=975&iar=0&w=2560&rev=-1&hash=5857B1089498B7F150BE91751F07D10D)

FTC pauses "click-to-cancel" rule until July 14

[Read more](/en-us/insights/publications/2025/05/ftc-pauses-click-to-cancel-rule-until-july-14)

![](/-/media/project/dlapiper-tenant/dlapiper/editorialdefaultimages/abstract-business-building-background.jpg?h=975&iar=0&w=2560&rev=-1&hash=6A9DDB6E4B6D68C83A79CB0A904089AC)

Long-awaited DOJ guidance on white collar enforcement priorities and policies: Key takeaways

[Read more](/en-us/insights/publications/2025/05/doj-guidance-on-white-collar-enforcement-priorities-and-policies)

## Featured insights

[News![]()DLA Piper advises Nerdio in securing US$500 Million in Series C Investment from General...12 May 2025 .1 minute read](/en-us/news/2025/05/dla-piper-advises-nerdio-in-securing-usd500-million-in-series-c-investment-from-general-atlantic)

[Publication![]()Chile publishes instructions on appraisal authority and its application to corporate...11 April 2025 .4 minute read](/en-us/insights/publications/2025/04/chile-publishes-instructions-on-appraisal-authority-and-its-application-to-corporate-reorganizations)

[Publication![]()The second Trump Administration’s first 100 days6 May 2025 .29 minute read](/en-us/insights/publications/2025/05/100-days-of-trump-enforcement)

## DLA Piper in the news

[News![]()Multifamily investors: Buying time?29 May 2025 .1 minute read](/en-us/news/2025/05/multifamily-investors-buying-time)

[Publication![]()California Shows Path to States Stepping Up for FCPA Enforcement13 May 2025 .1 minute read](/en-us/insights/publications/2025/05/california-shows-path-to-states-stepping-up-for-fcpa-enforcement)

[News![]()Tony Samp named to Washingtonian magazine’s 2025 list of 500 Most Influential People in...8 May 2025 .3 minute read](/en-us/news/2025/05/tony-samp-named-to-washingtonian-magazines-2025-list-of-500-most-influential-people-in-dc)

## [Find a career](/en-us/careers)

## [Find a lawyer](/en-us/people)

  * [People](/en-us/people)
  * [Capabilities](/en-us/capabilities)
  * [About us](/en-us/about-us)
  * [Insights](/en-us/insights)
  * [Careers](/en-us/careers)


  * [Locations](/en-us/locations)
  * [News](/en-us/news)
  * [Events](/en-us/events)
  * [Blogs](/en-us/insights/blogs)
  * [Alumni](/en-us/alumni)
  * [Pro bono](/en-us/about-us/pro-bono)



[](https://www.linkedin.com/company/dla-piper/)[](https://twitter.com/DLA_Piper)[](https://instagram.com/dlapiper/)

[](https://www.facebook.com/DLAPiperGlobal/)[](https://www.youtube.com/user/DLAPipervideos)

  * [Contact us](/en-us/contact-us)
  * [Find an office](/en-us/locations)
  * [Subscribe](/en-us/subscribe)



Also of interest

  * [Franchise](https://www.dlapiper.com/en-us/capabilities/practice-area/franchise)
  * [Intellectual Property](https://www.dlapiper.com/en-us/capabilities/practice-area/intellectual-property)
  * [Global Coverage for a Global Asset Class](https://www.dlapiper.com/en-us/capabilities/practice-area/real-estate)



[Privacy policy](/en-us/legal-notices/additional/privacy-policy "internal")[Your privacy choices](/en-us/legal-notices/additional/your-privacy-choices "internal")[Legal notices](/en-us/legal-notices/global-legal-notices "internal")[Cookie policy](/en-us/legal-notices/additional/cookie-policy "internal")[Fraud Alert](/en-us/legal-notices/additional/fraud-alert "internal")[Make a payment](https://paymentportal.dlapiper.com/ "external")[Sitemap](https://www.dlapiper.com/sitemap.xml "external")

DLA Piper is a global law firm operating through various separate and distinct legal entities. For further information about these entities and DLA Piper's structure, please refer to the [Legal Notices](/en-us/legal-notices/country/united-states) page of this website. All rights reserved. Attorney advertising.

© 2025 DLA Piper US

![youtube](/-/media/project/dlapiper-tenant/dlapiper/icons/social-youtube-hover.svg)![wechat](/-/media/project/dlapiper-tenant/dlapiper/icons/social-wechat-hover.svg)![twitter](/-/media/project/dlapiper-tenant/dlapiper/icons/social-twitter-hover.svg)![linkedin](/-/media/project/dlapiper-tenant/dlapiper/icons/social-linkedin-hover.svg)![instagram](/-/media/project/dlapiper-tenant/dlapiper/icons/social-instagram-hover.svg)![facebook](/-/media/project/dlapiper-tenant/dlapiper/icons/social-facebook-hover.svg)![regular](/-/media/project/dlapiper-tenant/dlapiper/icons/regular.svg)![hover-white](/-/media/project/dlapiper-tenant/dlapiper/icons/hover-white.svg)![hover](/-/media/project/dlapiper-tenant/dlapiper/icons/hover.svg)![globe-icon-hover](/-/media/project/dlapiper-tenant/dlapiper/icons/globe-icon-hover.svg)![footer-external](/-/media/project/dlapiper-tenant/dlapiper/icons/footerexternaliconhoversmall.svg)![bookmarks](/-/media/project/dlapiper-tenant/dlapiper/icons/bookmarks-header-hover.svg)![bookmarks](/-/media/project/dlapiper-tenant/dlapiper/icons/bookmarks-header.svg)![arrow](/-/media/project/dlapiper-tenant/dlapiper/icons/arrow_forward_stretch_alert_24.svg)![arrow](/-/media/project/dlapiper-tenant/dlapiper/icons/arrow_forward_stretch_24.svg)![facebook](/-/media/project/dlapiper-tenant/dlapiper/icons/facebook-icon.svg)![twitter](/-/media/project/dlapiper-tenant/dlapiper/icons/twitter-icon.svg)![email](/-/media/project/dlapiper-tenant/dlapiper/icons/email-icon.svg)![linkedin](/-/media/project/dlapiper-tenant/dlapiper/icons/linkedin-icon.svg)![whatsapp](/-/media/project/dlapiper-tenant/dlapiper/icons/whatsapp-icon.svg)![line](/-/media/project/dlapiper-tenant/dlapiper/icons/line-icon.svg)![wechat](/-/media/project/dlapiper-tenant/dlapiper/icons/wechat-icon.svg)![xing](/-/media/project/dlapiper-tenant/dlapiper/icons/xing-icon.svg)
""")
    expand_keywords._shared_state.set("KEYWORDS", [{'keyword': 'law firm', 'match_type': 'BROAD'}, {'keyword': 'dla piper', 'match_type': 'EXACT'}, {'keyword': 'mergers acquisitions', 'match_type': 'BROAD'}, {'keyword': 'legal services', 'match_type': 'BROAD'}, {'keyword': 'find a lawyer', 'match_type': 'EXACT'}, {'keyword': 'intellectual property law', 'match_type': 'PHRASE'}, {'keyword': 'top law firms us', 'match_type': 'PHRASE'}, {'keyword': 'dla piper careers', 'match_type': 'PHRASE'}, {'keyword': 'corporate counsel', 'match_type': 'BROAD'}, {'keyword': 'international law firm', 'match_type': 'PHRASE'}, {'keyword': 'global law firm', 'match_type': 'PHRASE'}, {'keyword': 'click to cancel rule', 'match_type': 'EXACT'}, {'keyword': 'international lawyers', 'match_type': 'BROAD'}, {'keyword': 'dla piper law firm', 'match_type': 'PHRASE'}, {'keyword': 'mergers and acquisitions lawyer', 'match_type': 'PHRASE'}, {'keyword': 'franchise law attorney', 'match_type': 'PHRASE'}, {'keyword': 'real estate law firm', 'match_type': 'PHRASE'}, {'keyword': 'mergers and acquisitions', 'match_type': 'PHRASE'}, {'keyword': 'm&as', 'match_type': 'PHRASE'}, {'keyword': 'business lawyer', 'match_type': 'PHRASE'}, {'keyword': 'business attorneys', 'match_type': 'PHRASE'}, {'keyword': 'business legal services', 'match_type': 'PHRASE'}, {'keyword': 'm&a deals', 'match_type': 'PHRASE'}, {'keyword': 'merger and acquisition deals', 'match_type': 'PHRASE'}, {'keyword': 'international legal firm', 'match_type': 'PHRASE'}, {'keyword': 'merger integration', 'match_type': 'PHRASE'}, {'keyword': 'merger and acquisition integration', 'match_type': 'PHRASE'}, {'keyword': 'commercial law firm', 'match_type': 'PHRASE'}, {'keyword': 'corp counsel', 'match_type': 'PHRASE'}, {'keyword': 'international business lawyer', 'match_type': 'PHRASE'}, {'keyword': 'business acquisitions', 'match_type': 'PHRASE'}, {'keyword': 'top global law firm', 'match_type': 'PHRASE'}, {'keyword': 'business law firm', 'match_type': 'PHRASE'}, {'keyword': 'global legal firm', 'match_type': 'PHRASE'}, {'keyword': 'new mergers and acquisitions', 'match_type': 'PHRASE'}, {'keyword': 'dla piper jobs', 'match_type': 'PHRASE'}, {'keyword': 'international attorney', 'match_type': 'PHRASE'}, {'keyword': 'm&a integration', 'match_type': 'PHRASE'}, {'keyword': 'law firm for business', 'match_type': 'PHRASE'}, {'keyword': 'law firm business', 'match_type': 'PHRASE'}, {'keyword': 'company acquisitions', 'match_type': 'PHRASE'}, {'keyword': 'merger and integration', 'match_type': 'PHRASE'}]
)
    
    expand_keywords._shared_state.set("URL", "https://www.dlapiper.com/en-us")
    expand_result = expand_keywords.run()
    print(f"Result: {expand_result}")
    print(f"✅ ExpandKeywords test completed successfully!")
    