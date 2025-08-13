import os
import re
import json
import copy
import jsonref
import threading
from openai import OpenAI
from anthropic import Anthropic
from agency_swarm.tools import BaseTool
from typing import List, Dict, Annotated
from pydantic import (
    Field,
    PrivateAttr,
    BaseModel,
    StringConstraints,
    ConfigDict,
    ValidationError,
    HttpUrl
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from tools.ConductorAgent.util.constants import (
    StructuredSnippetType,
    ConversionActionType,
    ConversionActionCategory,
)

from dotenv import load_dotenv

load_dotenv()

anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

GPT_MODEL = "o3"
CLAUDE_MODEL = "claude-sonnet-4-0"

openai_client = OpenAI()


class AdCopyResponseModel(BaseModel):
    generated_content: List[str] = Field(
        description="An array of headlines or descriptions, depending on the task."
    )


class SitelinkModel(BaseModel):
    url: HttpUrl = Field(description="URL for the sitelink. Must be a full URL path with protocol, including 'http' or 'https'.")
    link_text: str = Field(
        description="URL display text for the sitelink. The length of this string should be between 1 and 25.",
        max_length=25,
    )
    description1: str = Field(
        description="First line of the description for the sitelink. Length should be between 1 and 35. Should be a complete sentence.",
        max_length=35,
    )
    description2: str = Field(
        description="Second line of the description for the sitelink. Length should be between 1 and 35. Should be a complete sentence.",
        max_length=35,
    )


class ConversionAction(BaseModel):
    name: str = Field(description="Name of the conversion action")
    type: ConversionActionType = Field(
        description="Type of the conversion action. Choose a single value from enum list as a string.",
        examples=["SOME_ACTION_TYPE"],
    )
    category: ConversionActionCategory = Field(
        description="Category of the conversion action. Choose a single value from enum list as a string.",
        examples=["SOME_ACTION_CATEGORY"],
    )


class StructuredSnippet(BaseModel):
    header: StructuredSnippetType = Field(
        description="Header of the structured snippet. Choose a single value from enum list as a string.",
        examples=["Some header"],
    )
    values: List[Annotated[str, StringConstraints(max_length=25)]] = Field(
        description=(
            "Values of the structured snippet. Max length of a single value is 25 characters. "
            "Length of the list must be between 3 and 10."
        ),
        max_length=10,
        min_length=3,
    )


class AdExtensionsModel(BaseModel):
    structured_snippet: StructuredSnippet = Field(
        description=(
            "Dictionary, containing a structured snippet. Consists of a single header and a list of values. "
            "THIS IS NOT A LIST, but a single dictionary with a header and values."
            "Structured snippet is an ad extension that provides context on the nature of your product or service."
            "example={'header': 'Some header', 'values': ['Some value 1', 'Some value 2', 'Some value 3']}"
        ),
    )
    callouts: List[Annotated[str, StringConstraints(max_length=25)]] = Field(
        description=(
            "List of callout extensions. Each callout must be 25 chars or less."
            "Provide 4-6 items for this field."
            "example=['Some callout 1', 'Some callout 2', 'Some callout 3', 'Some callout 4']"
        ),
        max_length=6,
        min_length=4,
    )
    sitelinks: List[SitelinkModel] = Field(
        description=(
            "List of sitelinks extensions with names, descriptions, and URLs. Do not include more than 6 sitelinks. "
            "At least 4 sitelinks are required. Return less than 4 only if there are less than 4 relevant links on a page. "
            "example=[{'url': 'https://www.someurl.com','link_text': 'Some link text','description1': 'Some description 1','description2': 'Some description 2'}],"
        ),
        max_length=6,
    )
    conversion_action: ConversionAction = Field(
        description=(
            "Dictionary, containing primary conversion goal. Do not include 'enum' field in your response."
            "example={'name': 'Some action name', 'type': 'SOME_ACTION_TYPE', 'category': 'SOME_ACTION_CATEGORY'}"
        ),
    )


class GenerateAdCopy(BaseTool):
    """
    Generates multiple variations of ad headlines and descriptions using GPT-4 and Claude-3.5-Sonnet
    """

    _results: dict = PrivateAttr(default_factory=dict)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    class ToolConfig:
        one_call_at_a_time = True

    def run(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Generates ad copy variations and extensions using both GPT-4 and Claude in parallel
        """
        website_data = self._shared_state.get("SEARCH_DATA", None)
        if not website_data:
            raise ValueError(
                "Website data not found. Please run KeywordSearch tool first."
            )

        # Create tasks for parallel execution
        tasks = [
            (self.generate_text_completion, GPT_MODEL, "headline", website_data),
            (self.generate_text_completion, GPT_MODEL, "description", website_data),
            (self.generate_text_completion, CLAUDE_MODEL, "headline", website_data),
            (self.generate_text_completion, CLAUDE_MODEL, "description", website_data),
            (self.generate_extensions, GPT_MODEL, website_data, None),
            (self.generate_extensions, CLAUDE_MODEL, website_data, None),
        ]

        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(func, model, content_type, website_data)
                for func, model, content_type, website_data in tasks
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {str(e)}")

        # Extend sitelinks if there's not enough
        if len(self._results[f"{GPT_MODEL}_extensions"]["sitelinks"]) < 4:
            for sitelink in self._results[f"{CLAUDE_MODEL}_extensions"]["sitelinks"]:
                if sitelink["url"] not in [s["url"] for s in self._results[f"{GPT_MODEL}_extensions"]["sitelinks"]]:
                    self._results[f"{GPT_MODEL}_extensions"]["sitelinks"].append(sitelink)
                    if len(self._results[f"{GPT_MODEL}_extensions"]["sitelinks"])==4:
                        break

        if len(self._results[f"{CLAUDE_MODEL}_extensions"]["sitelinks"]) < 4:
            for sitelink in self._results[f"{GPT_MODEL}_extensions"]["sitelinks"]:
                if sitelink["url"] not in [s["url"] for s in self._results[f"{CLAUDE_MODEL}_extensions"]["sitelinks"]]:
                    self._results[f"{CLAUDE_MODEL}_extensions"]["sitelinks"].append(sitelink)
                    if len(self._results[f"{CLAUDE_MODEL}_extensions"]["sitelinks"])==4:
                        break
        
        if len(self._results[f"{GPT_MODEL}_extensions"]["sitelinks"]) < 4 or len(self._results[f"{CLAUDE_MODEL}_extensions"]["sitelinks"]) < 4:
            raise ValueError("Not enough sitelinks generated. Ask user to provide link to a different page.")

        result = {
            "gpt4": {
                "headlines": self._results[f"{GPT_MODEL}_headlines"],
                "descriptions": self._results[f"{GPT_MODEL}_descriptions"],
                "extensions": self._results[f"{GPT_MODEL}_extensions"],
            },
            "claude": {
                "headlines": self._results[f"{CLAUDE_MODEL}_headlines"],
                "descriptions": self._results[f"{CLAUDE_MODEL}_descriptions"],
                "extensions": self._results[f"{CLAUDE_MODEL}_extensions"],
            },
        }

        self._shared_state.set("GENERATED_AD_COPY", result)

        return result

    def generate_text_completion(
        self, model: str, content_type: str, page_data
    ) -> List[str]:
        """Generate headlines using GPT-4"""
        user_input = page_data
        if content_type == "headline":
            if model == GPT_MODEL:
                prompt = self.get_headline_prompt(30)
            elif model == CLAUDE_MODEL:
                # Reduced length for Claude as it tends to create longer content
                prompt = self.get_headline_prompt(25)
            content_length = 30
            array_length = 15
        elif content_type == "description":
            if model == GPT_MODEL:
                prompt = self.get_description_prompt(90)
            elif model == CLAUDE_MODEL:
                # Reduced length for Claude as it tends to create longer content
                prompt = self.get_description_prompt(80)
            content_length = 90
            array_length = 4

        content = []
        retries = 0
        while len(content) < array_length:
            try:
                if model == GPT_MODEL:
                    response = openai_client.beta.chat.completions.parse(
                        model=GPT_MODEL,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": user_input},
                        ],
                        # temperature=0.5,
                        max_completion_tokens=10000,
                        response_format=AdCopyResponseModel,
                    )
                    generated_content = json.loads(response.choices[0].message.content)[
                        "generated_content"
                    ]
                elif model == CLAUDE_MODEL:
                    response = anthropic.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=10000,
                        system=prompt,
                        temperature=0.5,
                        messages=[
                            {
                                "role": "user",
                                "content": f"""Please provide your response in the following JSON format. Do not include any other text than the JSON object in your response. Each sentence must be {content_length} characters or less:
                        {{
                            "content": [
                                "headline or description 1",
                                "headline or description 2",
                                ...
                            ]
                        }}
                        
                        Website content: {user_input}""",
                            }
                        ],
                    )
                    # Claude is not strictly following the JSON format and might return additional text
                    try:
                        response_json = json.loads(response.content[0].text)
                    except Exception as e:
                        json_match = re.search(r'\{[\s\S]*\}', response.content[0].text)
                        if json_match:
                            response_json = json.loads(json_match.group())
                        else:
                            raise e

                    generated_content = response_json["content"]
            except Exception as e:
                print(f"Error generating content for {model} {content_type}s: {e}")
                retries += 1
                if retries == 5:
                    # Set empty results instead of raising error to prevent KeyError later
                    print(f"Failed to generate {content_type}s for {model} after 5 retries. Setting empty results.")
                    with self._lock:
                        if content_type == "headline":
                            self._results[f"{model}_headlines"] = []
                        elif content_type == "description":
                            self._results[f"{model}_descriptions"] = []
                    return []
                continue

            # If some sentences are longer than required, ask to rewrite them
            new_prompt = f"Please rewrite following sentences to be {content_length} characters or less:\n"
            for item in generated_content:
                if len(item) > content_length:
                    new_prompt += f"{item}\n"
                else:
                    content.append(item)
            user_input = new_prompt

        with self._lock:
            if content_type == "headline":
                self._results[f"{model}_headlines"] = content[:array_length]
            elif content_type == "description":
                self._results[f"{model}_descriptions"] = content[:array_length]

    def generate_extensions(self, model: str, page_data, _) -> Dict:
        """Generate ad extensions using the specified model"""
        retries = 0
        while retries < 5:
            corrections = 0
            if model == GPT_MODEL:
                message_history = [
                    {"role": "system", "content": self.get_extensions_prompt()},
                    {"role": "user", "content": page_data},
                ]
            elif model == CLAUDE_MODEL:
                message_history = [
                    {
                        "role": "user",
                        "content": (
                            f"Generate ad extensions based on this website content. "
                            "Do not include any other text than the JSON object in your response:\n\n "
                            f"{page_data}\n\n"
                            "Please provide your response in the following JSON format, "
                            "make sure to follow rules provided in the field descriptions:\n\n"
                            f"{jsonref.replace_refs(self.get_schema_without_constraints().model_json_schema())['$defs']}"
                        ),
                    },
                ]
            while corrections < 7:
                try:
                    if model == GPT_MODEL:
                        response = openai_client.beta.chat.completions.parse(
                            model=GPT_MODEL,
                            messages=message_history,
                            # temperature=0.5,
                            max_completion_tokens=10000,
                            response_format=self.get_schema_without_constraints(),
                        )
                        extensions = json.loads(response.choices[0].message.content)
                        
                        # Post-process to fix validation issues for GPT as well
                        extensions = self._fix_validation_issues(extensions)
                    elif model == CLAUDE_MODEL:
                        response = anthropic.messages.create(
                            model=CLAUDE_MODEL,
                            max_tokens=10000,
                            system=self.get_extensions_prompt(),
                            temperature=0.5,
                            messages=message_history,
                        )
                        # Claude often wraps JSON in markdown code blocks, so we need to extract it
                        response_text = response.content[0].text
                        # Remove markdown code block markers if present
                        if response_text.strip().startswith('```'):
                            # Extract JSON from between ``` blocks
                            json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response_text, re.DOTALL)
                            if json_match:
                                response_text = json_match.group(1)
                        
                        extensions = json.loads(response_text)
                        
                        # Fix the structure to match expected model (singular keys)
                        if "structured_snippets" in extensions:
                            # Take the first structured snippet if there are multiple
                            extensions["structured_snippet"] = extensions["structured_snippets"][0]
                            del extensions["structured_snippets"]
                        
                        if "conversion_actions" in extensions:
                            # Take the first conversion action if there are multiple  
                            extensions["conversion_action"] = extensions["conversion_actions"][0]
                            del extensions["conversion_actions"]
                        
                        # Post-process to fix validation issues
                        extensions = self._fix_validation_issues(extensions)

                    message_history.append(
                        {"role": "assistant", "content": json.dumps(extensions)}
                    )
                    AdExtensionsModel.model_validate(extensions)
                    with self._lock:
                        self._results[f"{model}_extensions"] = extensions
                    return
                except ValidationError as e:
                    print(f"Validation error for {model}: {e}")
                    # Create more specific error message for the model
                    error_details = []
                    for error in e.errors():
                        field = error["loc"]
                        error_type = error["type"]
                        if error_type == "string_too_long":
                            max_length = error.get("ctx", {}).get("max_length", "unknown")
                            error_details.append(f"Field {field} is too long (max {max_length} chars)")
                        elif error_type == "too_long":
                            max_items = error.get("ctx", {}).get("max_length", "unknown")  
                            error_details.append(f"Field {field} has too many items (max {max_items})")
                        else:
                            error_details.append(f"Field {field}: {error['msg']}")
                    
                    error_message = "Please fix these specific issues:\n" + "\n".join(error_details)
                    error_message += "\nRemember: callouts max 25 chars each, sitelink descriptions max 35 chars each"
                    
                    message_history.append(
                        {
                            "role": "user",
                            "content": error_message,
                        }
                    )
                    corrections += 1
            # If model made too many corrections - it's stuck, start anew
            print("Re-generating completion")
            retries += 1

        with self._lock:
            self._results[f"{model}_extensions"] = "ERROR GENERATING EXTENSIONS"

    def get_description_prompt(self, sentence_length: int) -> str:
        description_prompt = (
            f"You are an expert ad copywriter. Create complete, impactful ad descriptions that are exactly {sentence_length} characters or less."
            "You will be given a website content by the user."
            "Generate 4 compelling ad descriptions for the following product/service.\n\n"
            "Requirements:\n"
            f"- Each description MUST be {sentence_length} characters or less\n"
            "- Each description must be a complete sentence or thought\n"
            "If you generated content that is longer than required, you will be asked to rewrite it. "
            "In which case return an array with the same number of items as in the list provided by the user, "
            "rewritten to fit the criteria."
        )
        return description_prompt

    def get_headline_prompt(self, sentence_length: int) -> str:
        headline_prompt = (
            f"You are an expert ad copywriter. Create complete, impactful headlines that are exactly {sentence_length} characters or less."
            "You will be given a website content by the user."
            "Generate 15 creative and compelling headlines variations to be used in Google Ads based "
            "on the provided landing page content."
            "Requirements:\n"
            f"- Each headline MUST be {sentence_length} characters or less\n"
            "- Be concise but impactful\n"
            "If you generated content that is longer than required, you will be asked to rewrite it. "
            "In which case return an array with the same number of items as in the list provided by the user, "
            "rewritten to fit the criteria."
        )
        return headline_prompt

    def get_extensions_prompt(self) -> str:
        extensions_prompt = (
            "You are an expert ad copywriter. Create Google Ads extensions based on the website content. "
            "Return your response as a raw JSON object only - do not wrap it in markdown code blocks or include any other text. "
            "The JSON must have exactly these fields: 'structured_snippet' (singular), 'callouts', 'sitelinks', and 'conversion_action' (singular). "
            "Follow the exact structure and field names as specified.\n\n"
            "CRITICAL LENGTH REQUIREMENTS:\n"
            "- callouts: Maximum 4-6 items, each item MUST be 25 characters or less\n"
            "- sitelinks: Maximum 6 items, each description1 and description2 MUST be 35 characters or less\n"
            "- sitelinks link_text: MUST be 25 characters or less\n"
            "- structured_snippet values: Each item MUST be 25 characters or less\n\n"
            "Example valid lengths:\n"
            "- 'Expert Legal Advice' (19 chars) ✓\n"
            "- 'Global Legal Solutions' (23 chars) ✓\n"
            "- 'Top-rated legal services' (26 chars) ✗ TOO LONG\n\n"
            "Count characters carefully and stay within limits."
        )
        return extensions_prompt

    def get_schema_without_constraints(self) -> AdExtensionsModel:
        """
        Removes min and max length constraints from schema
        as they are not supported by oai
        """
        schema_no_limits = copy.deepcopy(AdExtensionsModel.model_json_schema())

        # Remove length constraints from the schema
        def remove_length_constraints(schema_dict):
            if isinstance(schema_dict, dict):
                schema_dict.pop("minLength", None)
                schema_dict.pop("maxLength", None)
                schema_dict.pop("minItems", None)
                schema_dict.pop("maxItems", None)
                schema_dict.pop("format", None)
                # Recursively process nested objects
                for value in schema_dict.values():
                    if isinstance(value, (dict, list)):
                        remove_length_constraints(value)

        remove_length_constraints(schema_no_limits)
        schema = AdExtensionsModel
        schema.model_config = ConfigDict(
            json_schema_extra=jsonref.replace_refs(schema_no_limits)
        )
        return schema

    def _fix_validation_issues(self, extensions: Dict) -> Dict:
        """Post-process extensions to fix validation issues"""
        
        # Fix callouts - limit to 6 items and truncate if too long
        if "callouts" in extensions:
            callouts = extensions["callouts"]
            # Limit to 6 items
            callouts = callouts[:6]
            # Truncate each callout to 25 characters
            callouts = [callout[:25] if len(callout) > 25 else callout for callout in callouts]
            # Ensure we have at least 4 items
            while len(callouts) < 4 and len(callouts) > 0:
                callouts.append(callouts[0][:20] + "...")
            extensions["callouts"] = callouts
        
        # Fix sitelinks descriptions
        if "sitelinks" in extensions:
            for sitelink in extensions["sitelinks"]:
                # Truncate link_text to 25 characters
                if "link_text" in sitelink and len(sitelink["link_text"]) > 25:
                    sitelink["link_text"] = sitelink["link_text"][:22] + "..."
                
                # Truncate description1 to 35 characters
                if "description1" in sitelink and len(sitelink["description1"]) > 35:
                    sitelink["description1"] = sitelink["description1"][:32] + "..."
                
                # Truncate description2 to 35 characters  
                if "description2" in sitelink and len(sitelink["description2"]) > 35:
                    sitelink["description2"] = sitelink["description2"][:32] + "..."
        
        # Fix structured_snippet values
        if "structured_snippet" in extensions and "values" in extensions["structured_snippet"]:
            values = extensions["structured_snippet"]["values"]
            # Truncate each value to 25 characters
            values = [value[:25] if len(value) > 25 else value for value in values]
            # Limit to 10 items and ensure at least 3
            values = values[:10]
            while len(values) < 3 and len(values) > 0:
                values.append(values[0])
            extensions["structured_snippet"]["values"] = values
        
        # Fix conversion_action enum values
        if "conversion_action" in extensions:
            conversion_action = extensions["conversion_action"]
            # Map invalid enum values to valid ones
            type_mapping = {
                "SUBMIT_LEAD_FORM": "WEBPAGE",
                "PHONE_CALL_LEAD": "CLICK_TO_CALL",
                "CONTACT": "WEBPAGE",
                "SIGNUP": "WEBPAGE",
                "DOWNLOAD": "WEBPAGE",
                "PURCHASE": "WEBPAGE"
            }
            
            if "type" in conversion_action and conversion_action["type"] in type_mapping:
                conversion_action["type"] = type_mapping[conversion_action["type"]]
            elif "type" in conversion_action and conversion_action["type"] not in [
                "AD_CALL", "CLICK_TO_CALL", "GOOGLE_PLAY_DOWNLOAD", "GOOGLE_PLAY_IN_APP_PURCHASE",
                "UPLOAD_CALLS", "UPLOAD_CLICKS", "WEBPAGE", "WEBSITE_CALL"
            ]:
                # Default to WEBPAGE for any unrecognized type
                conversion_action["type"] = "WEBPAGE"
        
        return extensions


if __name__ == "__main__":
    # Test the GenerateAdCopy tool
    print("Testing GenerateAdCopy tool...")
    TEST_SEARCH_DATA = """
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

However bold your ambition, you can trust we'll make it happen.

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

Whatever your destination, we're by your side on the journey.

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

[Publication![]()The second Trump Administration's first 100 days6 May 2025 .29 minute read](/en-us/insights/publications/2025/05/100-days-of-trump-enforcement)

## DLA Piper in the news

[News![]()Multifamily investors: Buying time?29 May 2025 .1 minute read](/en-us/news/2025/05/multifamily-investors-buying-time)

[Publication![]()California Shows Path to States Stepping Up for FCPA Enforcement13 May 2025 .1 minute read](/en-us/insights/publications/2025/05/california-shows-path-to-states-stepping-up-for-fcpa-enforcement)

[News![]()Tony Samp named to Washingtonian magazine's 2025 list of 500 Most Influential People in...8 May 2025 .3 minute read](/en-us/news/2025/05/tony-samp-named-to-washingtonian-magazines-2025-list-of-500-most-influential-people-in-dc)

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
""" 
    
    ad_copy_generator = GenerateAdCopy()
    # Run GenerateAdCopy
    ad_copy_generator._shared_state.set("SEARCH_DATA", TEST_SEARCH_DATA)
    ad_copy_result = ad_copy_generator.run()
    print(f"Result: {ad_copy_result}")
    print(f"✅ GenerateAdCopy test completed successfully!")