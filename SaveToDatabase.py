import json
from typing import Union, Dict, Literal, List, Annotated
from agency_swarm.tools import BaseTool
from tools.ConductorAgent.util.constants import (
    StructuredSnippetType,
    ConversionActionType,
    ConversionActionCategory,
)
from pydantic import Field, BaseModel, StringConstraints


class AdCopyResponseModel(BaseModel):
    generated_content: List[str] = Field(
        description="An array of headlines or descriptions, depending on the task."
    )


class SitelinkModel(BaseModel):
    url: str = Field(description="URL for the sitelink")
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
            "List of sitelinks extensions with names, descriptions, and URLs. Do not include more than 6 sitelinks."
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


class AdCopy(BaseModel):
    headlines: List[Annotated[str, StringConstraints(max_length=30)]] = Field(
        description="Headline", max_length=15
    )

    descriptions: List[Annotated[str, StringConstraints(max_length=90)]] = Field(
        description="Description", max_length=4
    )
    extensions: AdExtensionsModel = Field(description="Ad extensions")


class Keyword(BaseModel):
    keyword: str
    match_type: Literal["BROAD", "PHRASE", "EXACT"]


class SaveToDatabase(BaseTool):
    """
    Tool that validates the input data and
    saves it to the database if it is valid
    """

    data_type: Literal["KEYWORDS", "GENERATED_AD_COPY"] = Field(
        description="The type of data to save"
    )
    data: Union[List[Keyword], Dict] = Field(
        description=(
            "The data to save. If data_type is KEYWORDS, data must be a list of dictionaries with the following structure: {'keyword': str, 'match_type': str}. "
            "If data_type is GENERATED_AD_COPY, data must be a dictionary with the following "
            "structure: \n{'model_name_1': {'headlines': List[str], 'descriptions': List[str], 'extensions': Dict}, 'model_name_2': {'headlines': List[str], 'descriptions': List[str], 'extensions': Dict}}"
        )
    )

    class ToolConfig:
        one_call_at_a_time = True

    def run(self):
        if self.data_type == "KEYWORDS":
            if not isinstance(self.data, list):
                raise ValueError("Keywords must be a list")
        elif self.data_type == "GENERATED_AD_COPY":
            if not isinstance(self.data, dict):
                raise ValueError("Generated ad copy must be a dictionary")
            else:
                for model in self.data:
                    AdCopy.model_validate(self.data[model])

        if self.data_type == "KEYWORDS":
            keywords = []
            for keyword in self.data:
                keywords.append(json.loads(keyword.model_dump_json()))
            self.data = keywords
        
        self._shared_state.set(self.data_type, self.data)
        return "Data saved successfully"

    def check_fields(self, data: Dict, fields: List[str], path: str = ""):
        for field in fields:
            if field not in data:
                raise ValueError(f"Empty required field '{field}' at {path or 'root'}")
            if not data[field]:
                raise ValueError(f"Empty required field '{field}' at {path or 'root'}")
            if isinstance(data[field], dict):
                self.check_fields(
                    data[field], fields, f"{path}.{field}" if path else field
                )
