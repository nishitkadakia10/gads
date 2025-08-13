import os
import re
import json
import time
import base64
import inspect
import pathlib
import asyncio
import litellm
import requests
import tempfile
import pandas as pd
import nodriver as uc

from openai import OpenAI
from fuzzywuzzy import process
from typing import Literal, List, Dict, Union
from agency_swarm.tools import BaseTool
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import Field, HttpUrl, BaseModel, model_validator, PrivateAttr
from crawl4ai import AsyncWebCrawler
from crawl4ai.cache_context import CacheMode

from tools.ConductorAgent.util.prompts import (
    GENERATE_KEYWORDS_PROMPT,
    CHOOSE_KEYWORDS_PROMPT,
    GENERATE_KEYWORDS_PROMPT_GEMINI,
)

from dotenv import load_dotenv

load_dotenv()

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

# Checks page content and filters irrelevant keywords
FILTER_MODEL = "o3"
FILTER_API_KEY = os.getenv("OPENAI_API_KEY")
# Performs main page data extraction, including headings, chapters and keywords
SCRAPER_MODEL = "o3"
SCRAPER_API_KEY = os.getenv("O1_API_KEY")
# Performs additional keyword generation (using Gemini)
GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Used for web scraping
DATA_FOR_CEO_NAME = os.getenv("DATA_FOR_CEO_NAME")
DATA_FOR_CEO_PASSWORD = os.getenv("DATA_FOR_CEO_PASSWORD")

service_account_info = json.loads(os.getenv("SERVICE_ACCOUNT_KEY_ADS"))
openai_client = OpenAI(api_key=FILTER_API_KEY)

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

THREAD_TIMEOUT = int(os.getenv("THREAD_TIMEOUT"))

# Remove the temporary file
os.remove(service_account_key_path)


class ScrapingError(Exception):
    pass


class KeywordSearch(BaseTool):
    """
    Tool that performs web scraping and keywords generation for a given web page url.
    Returns a list of keywords with their respective search volume data for a given region.
    Saves the scraped data and keywords to the database to be used in other tools.
    """

    url: HttpUrl = Field(..., description="URL to search keywords for.")

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
    # page_content: str = Field(
    #     None,
    #     description=(
    #         "Optional parameter for user to pass page content in html format. "
    #         "Should only be used when scraper is unable to retrieve page data."
    #     ),
    # )

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
        Fetches monthly search volumes for provided keywords
        Returns a dictionary with keywords as keys and search volumes as values
        """
        self.url = str(self.url).rstrip("/")

        # Save page content to local file if provided
        data_dir = os.path.join(CURRENT_DIR, "data")
        os.makedirs(data_dir, exist_ok=True)

        url_parts = self.url.replace("https://", "").replace("http://", "").split("/")
        filename = "_".join(url_parts).replace(".", "_")
        # Limit filename length and make it safe for filesystem
        filename = (
            "".join(c if c.isalnum() or c in "_-" else "_" for c in filename) + ".html"
        )
        try:
            self.scrape_page_to_file_api(url=self.url, save_path=filename)
        except Exception as e:
            print(f"Error scraping page using api, resorting to manual scraping: {e}")
            asyncio.run(self.scrape_page_to_file_local(url=self.url, save_path=filename))
        # scrape_url = pathlib.Path(temp_path).as_uri().replace("///", "//")
        scrape_url = "file://" + filename
        self.check_timeout()

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

            # Scrape webpage for keywords
            max_retries = 3
            retry_count = 0
            while True:
                scrape_result = asyncio.run(self.extract_page_data(scrape_url))
                self.check_timeout()

                if not scrape_result.success:
                    if (
                        scrape_result.error_message
                        and scrape_result.error_message != ""
                    ):
                        raise ScrapingError(
                            "Error scraping webpage: " + scrape_result.error_message
                        )
                    else:
                        raise ScrapingError(
                            "Unknown error encountered while scraping webpage data."
                        )

                try:
                    keywords = json.loads(scrape_result.extracted_content)[0][
                        "keywords"
                    ]
                    if not keywords or keywords == []:
                        raise ScrapingError(
                            "Page was not loaded properly or blocked with bot protection. "
                            "Ask user to provide a different url.\n"
                            "Page content for reference:\n"
                            + (
                                scrape_result.markdown_v2.raw_markdown[:1000] + "..."
                                if len(scrape_result.markdown_v2.raw_markdown) > 1000
                                else scrape_result.markdown_v2.raw_markdown
                            )
                        )
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"Error: {e}")
                    extracted_content = json.loads(scrape_result.extracted_content)
                    print(
                        f"Extracted content: {extracted_content}"
                    )
                    print(f"Scrape result:\n{scrape_result.markdown_v2.raw_markdown}")
                    if "OpenAIException - Connection error" in str(extracted_content):
                        print("Error accessing o3 model, switching to gpt-o4-mini")
                        global SCRAPER_MODEL
                        SCRAPER_MODEL = "gpt-o4-mini"
                    if retry_count >= max_retries:
                        raise ScrapingError(
                            "Failed to parse page data after multiple attempts."
                        )
                    continue

            keywords = [
                {
                    "keyword": re.sub(r'[^a-zA-Z0-9\s&]', '', keyword["keyword"].lower().replace('-', ' ')),
                    "match_type": keyword["match_type"],
                }
                for keyword in keywords
            ]
            keyword_names = [keyword["keyword"] for keyword in keywords]

            if not self.check_page_data(scrape_result.markdown_v2.raw_markdown):
                raise ScrapingError(
                    "Page was not loaded properly or blocked with bot protection. "
                    "Ask user to provide a different url."
                )

            # Get the keyword plan ideas service
            keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
            # Generate keyword ideas
            results = {}
            filtered_out = {}
            keywords_to_save = []
            request = self.get_search_request_metrics(
                location_id=location_id, keywords=keyword_names
            )
            # Get keyword metrics
            keyword_metrics = (
                keyword_plan_idea_service.generate_keyword_historical_metrics(
                    request=request
                ).results
            )

            # Process results
            for keyword_data in keyword_metrics:
                self.check_timeout()
                keyword = keyword_data.text
                avg_monthly_searches = keyword_data.keyword_metrics.avg_monthly_searches
                if avg_monthly_searches > 15 and keyword in keyword_names:
                    results[keyword] = {
                        "avg_monthly_searches": avg_monthly_searches,
                        "match_type": keywords[keyword_names.index(keyword)][
                            "match_type"
                        ],
                    }
                    keywords_to_save.append(
                        {
                            "keyword": keyword,
                            "match_type": keywords[keyword_names.index(keyword)][
                                "match_type"
                            ],
                        }
                    )
                else:
                    filtered_out[keyword] = {
                        "avg_monthly_searches": avg_monthly_searches,
                        "match_type": keywords[keyword_names.index(keyword)][
                            "match_type"
                        ],
                    }

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
            filtered_out = dict(
                sorted(
                    filtered_out.items(),
                    key=lambda x: x[1]["avg_monthly_searches"],
                    reverse=True,
                )
            )

            # Expand the keywords using google's suggestions
            if len(keywords_to_save) < 50:
                print("Expanding keywords using google's suggestions")
                # if not self.page_content:
                if len(list(results.keys())[:10]) > 0:
                    search_type = "keyword_and_url"
                else:
                    search_type = "url"

                request = self.get_search_request_ideas(
                    type=search_type,
                    location_id=location_id,
                    keywords=list(results.keys())[:10],
                )
                url_ideas = keyword_plan_idea_service.generate_keyword_ideas(
                    request=request
                )
                self.check_timeout()

                generated_keywords = [
                    idea.text
                    for idea in url_ideas
                    if idea.keyword_idea_metrics.avg_monthly_searches > 15
                ]
                page_content = f"Title: {json.loads(scrape_result.extracted_content)[0]['title']}\n\n"
                page_content += f"Chapters: {json.loads(scrape_result.extracted_content)[0]['chapters']}\n\n"
                approved_keywords = self.filter_keywords(
                    page_content, generated_keywords
                )

                url_results = {}
                url_keywords = []
                for idea in url_ideas:
                    keyword = idea.text
                    avg_monthly_searches = (
                        idea.keyword_idea_metrics.avg_monthly_searches
                    )
                    if (
                        (avg_monthly_searches > 15)
                        and (keyword not in results)
                        and (keyword.lower() in approved_keywords)
                    ):
                        url_results[keyword] = {
                            "avg_monthly_searches": avg_monthly_searches,
                            "match_type": "PHRASE",
                        }
                        url_keywords.append(
                            {"keyword": keyword, "match_type": "PHRASE"}
                        )

                # Sort url_results by search volume (descending)
                url_results = dict(
                    sorted(
                        url_results.items(),
                        key=lambda x: x[1]["avg_monthly_searches"],
                        reverse=True,
                    )
                )
                url_keywords.sort(
                    key=lambda x: url_results[x["keyword"]]["avg_monthly_searches"],
                    reverse=True,
                )
                results.update(url_results)
                keywords_to_save.extend(url_keywords)

            # Take only first 50 results
            results = dict(list(results.items())[:50])

            self._shared_state.set(
                "SEARCH_DATA", scrape_result.markdown_v2.raw_markdown
            )
            self._shared_state.set("KEYWORDS", keywords_to_save[:50])
            self._shared_state.set("URL", self.url)

            os.remove(filename)
            print(f"filtered_out: {filtered_out}")

            return (
                "Keywords generated successfully, present following list "
                f"to the user in the readable format:\n{results}"
            )

        except GoogleAdsException as ex:
            error_message = []
            for error in ex.failure.errors:
                error_message.append(f"Error: {error.message}")
            raise GoogleAdsException(" | ".join(error_message))

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

    def get_search_request_ideas(
        self,
        type: Literal["keyword", "url", "keyword_and_url"],
        location_id: str,
        keywords: List[str],
    ):
        """
        Get the keyword search request.
        """
        keyword_plan_network = (
            client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS
        )
        if location_id:
            location_rns = self.map_locations_ids_to_resource_names(
                client, [location_id]
            )

        language_rn = client.get_service("GoogleAdsService").language_constant_path(
            "1000"
        )
        request = client.get_type("GenerateKeywordIdeasRequest")
        request.customer_id = os.getenv("GOOGLE_ADS_MANAGER_ID")
        request.language = language_rn
        if location_id:
            request.geo_target_constants = location_rns

        request.include_adult_keywords = False
        request.keyword_plan_network = keyword_plan_network
        if type == "url":
            request.site_seed.site = self.url
        elif type == "keyword":
            request.keyword_seed.keywords.extend(keywords)
        elif type == "keyword_and_url":
            request.keyword_and_url_seed.url = self.url
            request.keyword_and_url_seed.keywords.extend(keywords)
        else:
            raise ValueError(f"Invalid type: {type}")

        return request

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
        request.keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
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

    async def extract_page_data(self, url: str):
        litellm.drop_params = True

        class Chapter(BaseModel):
            heading: str
            content: str

        class Keyword(BaseModel):
            keyword: str
            match_type: Literal["BROAD", "PHRASE", "EXACT"]

        # Separate class for gemini
        class KeywordList(BaseModel):
            keywords: List[Keyword]

        class PageData(BaseModel):
            title: str
            keywords: List[Keyword]
            chapters: List[Chapter]

        data_scrape_strategy = LLMExtractionStrategy(
            provider=SCRAPER_MODEL,
            api_token=SCRAPER_API_KEY,
            schema=PageData.model_json_schema(),
            instruction=inspect.cleandoc(GENERATE_KEYWORDS_PROMPT),
            # **({"temperature": 0} if "o1" not in SCRAPER_MODEL else {})
        )

        gemini_strategy = LLMExtractionStrategy(
            provider=f"gemini/{GEMINI_MODEL}",
            api_token=GEMINI_API_KEY,
            schema=KeywordList.model_json_schema(),
            instruction=inspect.cleandoc(GENERATE_KEYWORDS_PROMPT_GEMINI),
        )

        # Script to wait for the page to load
        js_code = [
            # Sync delay to avoid crashing on page redirects
            """
        // Synchronous delay using a blocking loop
        const delay_timer_start = Date.now();
        while (Date.now() - delay_timer_start < 3000) {
            // Wait for 3 seconds
        }

        // Basic scroll
        window.scrollTo(0, document.body.scrollHeight);        
        """,
            # Wait for page to be fully loaded
            """
        (async () => {

            // Wait for page to be fully loaded
            if (document.readyState !== 'complete') {
                await new Promise(resolve => {
                    window.addEventListener('load', resolve);
                });
            }  

            window.scrollTo(0, document.body.scrollHeight);       
        })();
        """,
        ]
        async with AsyncWebCrawler(verbose=True) as crawler:
            # Base scraping to extract keywords and page data
            scraper_result = await crawler.arun(
                url=url,
                extraction_strategy=data_scrape_strategy,
                js_code=js_code,
                cache_mode=CacheMode.DISABLED,
            )

        async with AsyncWebCrawler(verbose=True) as crawler:
            # Generate keywords using Gemini
            gemini_result = await crawler.arun(
                url=url,
                extraction_strategy=gemini_strategy,
                js_code=js_code,
                cache_mode=CacheMode.DISABLED,
            )
        result = scraper_result.model_copy()

        # Combine keywords from both results
        try:
            gemini_data = json.loads(gemini_result.extracted_content)
            if isinstance(gemini_data, list):
                gemini_data = gemini_data[0]

            scraper_data = json.loads(scraper_result.extracted_content)
            if isinstance(scraper_data, list):
                scraper_data = scraper_data[0]

            # Create a set of existing keywords to avoid duplicates
            existing_keywords = {k["keyword"].lower() for k in scraper_data["keywords"]}

            # Add new keywords from Gemini that aren't already present
            for keyword in gemini_data.get("keywords", []):
                if keyword["keyword"].lower() not in existing_keywords:
                    scraper_data["keywords"].append(keyword)
                    existing_keywords.add(keyword["keyword"].lower())

            # Update the result with combined keywords
            result.extracted_content = json.dumps([scraper_data])
        except Exception as e:
            print(f"Error combining keywords: {e}")

        return result

    def filter_keywords(self, page_data: dict, keyword_list: List[str]):
        class KeywordList(BaseModel):
            keywords: List[str] = Field(
                description="List of chosen keywords, most relevant to the page content."
            )

        user_input = f"Page content: {page_data}\n\nGenerated keywords: {keyword_list}"
        
        try:
            response = openai_client.beta.chat.completions.parse(
                model=FILTER_MODEL,
                messages=[
                    {"role": "system", "content": CHOOSE_KEYWORDS_PROMPT},
                    {"role": "user", "content": user_input},
                ],
                # temperature=0.2,
                max_completion_tokens=4096,
                response_format=KeywordList,
            )
            
            # Parse the structured response
            result = json.loads(response.choices[0].message.content)
            chosen_keywords = result["keywords"]
            chosen_keywords = [keyword.lower() for keyword in chosen_keywords]
            return chosen_keywords
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error in filter_keywords: {e}")
            print(f"Response content: {response.choices[0].message.content}")
            # Return the original list if we can't parse the response
            return [keyword.lower() for keyword in keyword_list]
        except Exception as e:
            print(f"Error in filter_keywords: {e}")
            # Return the original list if there's any other error
            return [keyword.lower() for keyword in keyword_list]

    def check_page_data(self, page_data: dict):
        class IsPageLoaded(BaseModel):
            is_loaded: bool = Field(
                description=(
                    "True if page was loaded successfully, "
                    "False if it is partially loaded or blocked with bot protection."
                )
            )

        user_input = f"Page content:\n{page_data}"
        
        try:
            response = openai_client.beta.chat.completions.parse(
                model=FILTER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Analyze the page markdown content "
                            "and determine if it is loaded properly. "
                            "If page content indicates that page was not loaded fully or blocked with bot protection, "
                            "return False. Otherwise, return True. "
                            "Return your response as JSON with 'is_loaded' field."
                        ),
                    },
                    {"role": "user", "content": user_input},
                ],
                # temperature=0.2,
                max_completion_tokens=4096,
                response_format=IsPageLoaded,
            )
            
            # Parse the structured response
            result = json.loads(response.choices[0].message.content)
            return result["is_loaded"]
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error in check_page_data: {e}")
            print(f"Response content: {response.choices[0].message.content}")
            # Default to True if we can't parse the response
            return True
        except Exception as e:
            print(f"Error in check_page_data: {e}")
            # Default to True if there's any other error
            return True

    async def scrape_page_to_file_local(self, url, save_path):
        browser = None
        try:
            # Use Chrome browser with undetected-chromedriver's stealth mode
            browser = await uc.start(
                browser="chrome",
                browser_args=[
                    "--ignore-certificate-errors",
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--window-size=1920,1080",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-blink-features",
                    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                ],
            )

            page = await browser.get(url)
            time.sleep(2)
            js_code = [
                # Sync delay to avoid crashing on page redirects
                """
                // Synchronous delay using a blocking loop
                const delay_timer_start = Date.now();
                while (Date.now() - delay_timer_start < 3000) {
                    // Wait for 3 seconds
                }

                // Basic scroll (using scrollHeight here can lead to an error)
                window.scrollTo(0, 100000);
                """,
                # Wait for page to be fully loaded
                """
                (async () => {
                    // Wait for page to be fully loaded
                    if (document.readyState !== 'complete') {
                        await new Promise(resolve => {
                            window.addEventListener('load', resolve);
                        });
                    }  
                    window.scrollTo(0, document.body.scrollHeight);       
                })();
                """,
            ]
            await page.evaluate(js_code[0])
            await page.evaluate(js_code[1])

            await page.save_screenshot("debug.png")
            content = await page.get_content()

            # Save page content to a file
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Page content saved successfully to {save_path}")
            except Exception as e:
                print(f"Error saving page content: {e}")

        finally:
            if browser:
                # Give time for subprocess to cleanup
                await asyncio.sleep(1)
                browser.stop()
                # Wait for any remaining subprocess tasks
                await asyncio.sleep(0.5)
                # Get all tasks except current one
                pending = [
                    task
                    for task in asyncio.all_tasks()
                    if task is not asyncio.current_task()
                ]
                # Cancel any remaining tasks
                for task in pending:
                    task.cancel()
                # Wait for cancellation to complete
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)


    def scrape_page_to_file_api(self, url, save_path):
        auth_token = base64.b64encode(f"{DATA_FOR_CEO_NAME}:{DATA_FOR_CEO_PASSWORD}".encode()).decode()
        
        data = {
            "url": url,
            "enable_javascript": True,
            "browser_preset": "desktop",
            "custom_js": "meta = {}; meta.url = document.URL; meta;",
            "store_raw_html": True,
            "load_resources": True,
            "enable_browser_rendering": True,
            "disable_cookie_popup": True,
            "enable_xhr": True,
        }

        headers = {
            "Authorization": f"Basic {auth_token}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post("https://api.dataforseo.com/v3/on_page/instant_pages", headers=headers, json=[data])
            response.raise_for_status()  # Raise an exception for bad status codes
            response_data = response.json()
            if response_data["status_code"] == 20000 or response_data["status_code"] == 20100:
                task_id = response_data["tasks"][0]["id"]
            else:
                raise Exception("error. Code: %d Message: %s" % (response_data["status_code"], response_data["status_message"]))
            
            post_data = {
                "id": task_id,
            }
            
            timeout = 30
            start_time = time.time()
            while time.time() < start_time + timeout:
                response = requests.post("https://api.dataforseo.com/v3/on_page/raw_html", headers=headers, json=[post_data])
                response.raise_for_status()
                response_data = response.json()
                if response_data["status_code"] == 20000:
                    html_text = response_data["tasks"][0]["result"][0]["items"]["html"]
                    break
                time.sleep(1)
            
            # Save page content to a file
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html_text)
            print(f"Page content saved successfully to {save_path}")
            
        except requests.RequestException as e:
            raise Exception(f"Error fetching page content: {e}")
        except IOError as e:
            raise Exception(f"Error saving page content: {e}")
        

    def check_timeout(self):
        if time.time() >= self._start_time + self._timeout:
            raise TimeoutError("Operation timed out")


if __name__ == "__main__":
    # Test the KeywordSearch tool
    print("Testing KeywordSearch tool...")
    
    # Example 1: Test with a business website
    try:
        keyword_search = KeywordSearch(
            url="https://www.dlapiper.com/en-us",
            location="Virginia",
            location_type="State",
            country_code="US",
        )
        result = keyword_search.run()
        print(keyword_search._shared_state.print_data())
        print("✅ KeywordSearch test completed successfully!")
        print(f"Result: {result}")
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"❌ KeywordSearch test failed: {e}")
        print("\n" + "="*50 + "\n")
    
    # # Example 2: Test with worldwide location
    # try:
    #     keyword_search_worldwide = KeywordSearch(
    #         url="https://www.example.com",
    #         location="Worldwide",
    #     )
    #     result_worldwide = keyword_search_worldwide.run()
    #     print("✅ KeywordSearch (Worldwide) test completed successfully!")
    #     print(f"Result: {result_worldwide}")
    # except Exception as e:
    #     print(f"❌ KeywordSearch (Worldwide) test failed: {e}")
    
    # # Example 3: Test with different location type
    # try:
    #     keyword_search_city = KeywordSearch(
    #         url="https://www.nike.com",
    #         location="New York",
    #         location_type="City",
    #         country_code="US",
    #     )
    #     result_city = keyword_search_city.run()
    #     print("✅ KeywordSearch (City) test completed successfully!")
    #     print(f"Result: {result_city}")
    # except Exception as e:
    #     print(f"❌ KeywordSearch (City) test failed: {e}")
