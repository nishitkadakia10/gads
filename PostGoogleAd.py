import os
import json
import pathlib
import tempfile
import pandas as pd
from datetime import datetime
from fuzzywuzzy import process
from agency_swarm.tools import BaseTool
from typing import Dict, Literal, Union
from google.ads.googleads.client import GoogleAdsClient
from pydantic import Field, model_validator, field_validator

from google.ads.googleads.errors import GoogleAdsException
from google.ads.googleads.v18.enums.types.advertising_channel_type import (
    AdvertisingChannelTypeEnum,
)
from dotenv import load_dotenv

load_dotenv()

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

class PostGoogleAd(BaseTool):
    campaign_title: str = Field(..., description="Campaign title")
    total_budget: float = Field(
        ...,
        description=(
            "Lifetime budget for the campaign, in the local currency for the account."
            "The daily spent, derived from campaign length and total budget should be more"
            "than 0.1 of local currency units"
        ),
    )
    campaign_type: Literal[
        "DEMAND_GEN",
        "DISPLAY",
        "HOTEL",
        "LOCAL",
        "LOCAL_SERVICES",
        "MULTI_CHANNEL",
        "PERFORMANCE_MAX",
        "SEARCH",
        "SHOPPING",
        "SMART",
        "TRAVEL",
        "VIDEO",
    ] = Field(..., description="Campaign type")

    start_date: str = Field(
        ...,
        description="Start date in YYYYMMDD format (e.g., '20240401' for April 1, 2024)",
    )
    end_date: str = Field(
        ...,
        description="End date in YYYYMMDD format (e.g., '20241231' for December 31, 2024)",
    )
    customer_id: int = Field(
        ..., description="Google Ads customer ID. 10 digit number, without dashes."
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

    @model_validator(mode="before")
    def validate_dates(cls, values):
        start_date = values.get("start_date")
        end_date = values.get("end_date")
        total_budget = values.get("total_budget")

        # Validate date formats
        for date_str in (start_date, end_date):
            if date_str is not None:
                try:
                    datetime.strptime(date_str, "%Y%m%d")
                except ValueError:
                    raise ValueError("Dates must be in YYYYMMDD format")

        # Validate end date is after start date if both are provided
        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y%m%d")
            if start < datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ):
                raise ValueError("Start date must not be in the past")
            end = datetime.strptime(end_date, "%Y%m%d")
            if end <= start:
                raise ValueError("End date must be after start date")

        start_datetime = datetime.strptime(start_date, "%Y%m%d")
        end_datetime = datetime.strptime(end_date, "%Y%m%d")
        campaign_days = (end_datetime - start_datetime).days + 1
        daily_budget = round(total_budget / campaign_days, 1)
        if daily_budget < 0.1:
            raise ValueError(
                "Daily budget must be greater than 0.1. Adjust total budget or start and end dates."
            )

        return values

    @field_validator("customer_id")
    def validate_customer_id(cls, v):
        if not v:
            raise ValueError("Customer ID is required")
        if len(str(v)) != 10:
            raise ValueError("Customer ID must be 10 digits")
        return v

    class ToolConfig:
        one_call_at_a_time = True

    def run(self):
        # First, check if provided location exists and get its ID
        if self.location != "Worldwide":
            location_id, error_msg = self.get_location_id(
                self.location, self.location_type, self.country_code
            )
            if error_msg:
                raise ValueError(error_msg)
        else:
            location_id = None
            
        self.customer_id = str(self.customer_id)

        # Add data to Google Sheet
        json_data = self._shared_state.get("GENERATED_AD_COPY")
        if not json_data:
            raise Exception("No generated ad copy found")
        
        # Post ad to Google Ads
        try:
            self.post_to_google_ads(json_data, location_id)
            return "Ad posted successfully on Google Ads.\n"
        except Exception as e:
            print(e)
            return f"Error posting ad to Google Ads: {e}\n"

    def post_to_google_ads(self, data: Dict, location_id: str):
        # Load the JSON key from the environment variable
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

        campaign_budget_service = client.get_service("CampaignBudgetService")

        campaign_budget_operation = client.get_type("CampaignBudgetOperation")
        campaign_budget = campaign_budget_operation.create
        campaign_budget.name = self.campaign_title
        campaign_budget.delivery_method = client.enums.BudgetDeliveryMethodEnum.STANDARD
        # Calculate number of days between start and end date
        start_datetime = datetime.strptime(self.start_date, "%Y%m%d")
        end_datetime = datetime.strptime(self.end_date, "%Y%m%d")
        campaign_days = (end_datetime - start_datetime).days + 1

        # Set daily budget based on campaign duration
        campaign_budget.amount_micros = round(
            round(self.total_budget / campaign_days, 1) * 1000000, 0
        )
        campaign_budget.explicitly_shared = False

        try:
            campaign_budget_response = campaign_budget_service.mutate_campaign_budgets(
                customer_id=self.customer_id, operations=[campaign_budget_operation]
            )
        except GoogleAdsException as ex:
            error_message = self.handle_googleads_exception(ex)
            raise Exception(error_message)

        # Create a campaign
        campaign_service = client.get_service("CampaignService")
        campaign_operation = client.get_type("CampaignOperation")
        campaign = campaign_operation.create
        campaign.name = self.campaign_title
        campaign.advertising_channel_type = getattr(
            AdvertisingChannelTypeEnum.AdvertisingChannelType, self.campaign_type
        )

        # Set start date if provided
        if self.start_date:
            campaign.start_date = self.start_date

        # Set end date if provided
        if self.end_date:
            campaign.end_date = self.end_date

        campaign.network_settings.target_google_search = True
        campaign.network_settings.target_search_network = True
        campaign.network_settings.target_partner_search_network = False
        campaign.status = client.enums.CampaignStatusEnum.PAUSED

        # Set the budget using the created budget resource
        campaign.campaign_budget = campaign_budget_response.results[0].resource_name
        # campaign.bidding_strategy_type = client.enums.BiddingStrategyTypeEnum.MAXIMIZE_CONVERSIONS
        # Enabling eCPC sometimes throws an error on creation
        # manual_cpc.enhanced_cpc_enabled = True
        campaign.maximize_conversions = client.get_type("MaximizeConversions")
        
        # Create the campaign
        try:
            campaign_response = campaign_service.mutate_campaigns(
                customer_id=self.customer_id, operations=[campaign_operation]
            )
        except GoogleAdsException as ex:
            error_message = self.handle_googleads_exception(ex)
            raise Exception(error_message)
        campaign_id = campaign_response.results[0].resource_name
        print(campaign_id)

        if location_id:
            campaign_criterion_service = client.get_service("CampaignCriterionService")
            campaign_criterion_operation = self.create_location_op(client, self.customer_id, campaign_id.split("/")[-1], location_id)
            campaign_criterion_service.mutate_campaign_criteria(
                customer_id=self.customer_id, operations=[campaign_criterion_operation]
            )

        for model, details in data.items():
            if "conversion_action" in details["extensions"]:
                conversion_action = self._create_conversion_action(
                    client, details["extensions"]["conversion_action"]
                )
                if conversion_action:
                    print(f"Created conversion action: {conversion_action}")

        # Create an ad group
        ad_group_service = client.get_service("AdGroupService")
        ad_group_operation = client.get_type("AdGroupOperation")
        ad_group = ad_group_operation.create
        ad_group.name = self.campaign_title + " Ad Group"
        ad_group.campaign = campaign_id
        ad_group.status = client.enums.AdGroupStatusEnum.PAUSED

        # Create the ad group
        ad_group_response = ad_group_service.mutate_ad_groups(
            customer_id=self.customer_id, operations=[ad_group_operation]
        )
        ad_group_id = ad_group_response.results[0].resource_name
        print(ad_group_id)

        # Iterate over each model in the data
        for model, details in data.items():
            # Create an ad
            ad_group_ad_service = client.get_service("AdGroupAdService")
            ad_group_ad_operation = client.get_type("AdGroupAdOperation")
            ad_group_ad = ad_group_ad_operation.create
            ad_group_ad.status = client.enums.AdGroupAdStatusEnum.PAUSED
            ad_group_ad.ad.final_urls.append(self._shared_state.get("URL"))

            # Add headlines (up to 15)
            headlines = []
            for i, headline in enumerate(details["headlines"][:15]):
                headlines.append(self._create_ad_text_asset(client, headline))
            ad_group_ad.ad.responsive_search_ad.headlines.extend(headlines)

            # Add descriptions (up to 4)
            descriptions = []
            for i, description in enumerate(details["descriptions"][:4]):
                descriptions.append(self._create_ad_text_asset(client, description))

            ad_group_ad.ad.responsive_search_ad.descriptions.extend(descriptions)

            ad_group_ad.ad_group = ad_group_id

            # Create the ad
            ad_group_ad_response = ad_group_ad_service.mutate_ad_group_ads(
                customer_id=self.customer_id, operations=[ad_group_ad_operation]
            )
            print(ad_group_ad_response)

            # Add sitelinks
            if "sitelinks" in details["extensions"]:
                resource_names = self._create_sitelink_assets(
                    client, details["extensions"]["sitelinks"]
                )
                operations = []
                for resource_name in resource_names:
                    operation = client.get_type("AdGroupAssetOperation")
                    ad_group_asset = operation.create
                    ad_group_asset.asset = resource_name
                    ad_group_asset.ad_group = ad_group_id
                    ad_group_asset.field_type = client.enums.AssetFieldTypeEnum.SITELINK
                    operations.append(operation)

                ad_group_asset_service = client.get_service("AdGroupAssetService")
                response = ad_group_asset_service.mutate_ad_group_assets(
                    customer_id=self.customer_id, operations=operations
                )

                for result in response.results:
                    print(
                        f"Linked sitelink to ad group with resource name '{result.resource_name}'."
                    )

            if "structured_snippet" in details["extensions"]:
                resource_names = self._create_structured_snippet_assets(
                    client, details["extensions"]["structured_snippet"]
                )
                operations = []
                for resource_name in resource_names:
                    operation = client.get_type("AdGroupAssetOperation")
                    ad_group_asset = operation.create
                    ad_group_asset.asset = resource_name
                    ad_group_asset.ad_group = ad_group_id
                    ad_group_asset.field_type = (
                        client.enums.AssetFieldTypeEnum.STRUCTURED_SNIPPET
                    )
                    operations.append(operation)

                ad_group_asset_service = client.get_service("AdGroupAssetService")
                response = ad_group_asset_service.mutate_ad_group_assets(
                    customer_id=self.customer_id, operations=operations
                )

                for result in response.results:
                    print(
                        f"Linked structured snippet to ad group with resource name '{result.resource_name}'."
                    )

            if "callouts" in details["extensions"]:
                resource_names = self._create_callout_assets(
                    client, details["extensions"]["callouts"]
                )
                operations = []
                for resource_name in resource_names:
                    operation = client.get_type("AdGroupAssetOperation")
                    ad_group_asset = operation.create
                    ad_group_asset.asset = resource_name
                    ad_group_asset.ad_group = ad_group_id
                    ad_group_asset.field_type = client.enums.AssetFieldTypeEnum.CALLOUT
                    operations.append(operation)

                ad_group_asset_service = client.get_service("AdGroupAssetService")
                response = ad_group_asset_service.mutate_ad_group_assets(
                    customer_id=self.customer_id, operations=operations
                )

                for result in response.results:
                    print(
                        f"Linked callout to ad group with resource name '{result.resource_name}'."
                    )

            # Add keywords to the ad group
            ad_group_criterion_service = client.get_service("AdGroupCriterionService")
            ad_group_criterion_operation = client.get_type("AdGroupCriterionOperation")
            keyword_operations = []
            for keyword in self._shared_state.get("KEYWORDS"):
                ad_group_criterion = ad_group_criterion_operation.create
                ad_group_criterion.keyword.text = keyword["keyword"]
                ad_group_criterion.keyword.match_type = getattr(
                    client.enums.KeywordMatchTypeEnum, keyword["match_type"]
                )

                ad_group_criterion.ad_group = ad_group_id
                ad_group_criterion.status = (
                    client.enums.AdGroupCriterionStatusEnum.ENABLED
                )
                keyword_operations.append(ad_group_criterion_operation)
                ad_group_criterion_service.mutate_ad_group_criteria(
                    customer_id=self.customer_id,
                    operations=[ad_group_criterion_operation],
                )

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

    def create_location_op(self, client, customer_id, campaign_id, location_id):
        campaign_service = client.get_service("CampaignService")
        geo_target_constant_service = client.get_service("GeoTargetConstantService")

        # Create the campaign criterion.
        campaign_criterion_operation = client.get_type("CampaignCriterionOperation")
        campaign_criterion = campaign_criterion_operation.create
        campaign_criterion.campaign = campaign_service.campaign_path(
            customer_id, campaign_id
        )

        campaign_criterion.location.geo_target_constant = (
            geo_target_constant_service.geo_target_constant_path(location_id)
        )

        return campaign_criterion_operation

    def handle_googleads_exception(self, exception):
        error_message = f'Request with ID "{exception.request_id}" failed with status '
        error_message += (
            f'"{exception.error.code().name}" and includes the following errors:'
        )
        for error in exception.failure.errors:
            error_message += f'Error with message "{error.message}".\n'
            if error.location:
                for field_path_element in error.location.field_path_elements:
                    error_message += f"On field: {field_path_element.field_name}\n"

        return error_message

    def _create_ad_text_asset(self, client, text, pinned_field=None):
        """Create an AdTextAsset."""
        ad_text_asset = client.get_type("AdTextAsset")
        ad_text_asset.text = text
        if pinned_field:
            ad_text_asset.pinned_field = pinned_field
        return ad_text_asset

    def _create_sitelink_assets(self, client, sitelinks):
        """Creates sitelink assets, which can be added to campaigns."""
        operations = []
        for sitelink in sitelinks:
            if "http" not in sitelink["url"]:
                print(f"Missing protocol in sitelink: {sitelink['url']}")
                continue
            store_locator_operation = client.get_type("AssetOperation")
            store_locator_asset = store_locator_operation.create
            store_locator_asset.final_urls.append(sitelink["url"])
            store_locator_asset.final_mobile_urls.append(sitelink["url"])
            store_locator_asset.sitelink_asset.description1 = sitelink["description1"]
            store_locator_asset.sitelink_asset.description2 = sitelink["description2"]
            store_locator_asset.sitelink_asset.link_text = sitelink["link_text"]
            operations.append(store_locator_operation)

        asset_service = client.get_service("AssetService")
        response = asset_service.mutate_assets(
            customer_id=self.customer_id,
            operations=operations,
        )

        resource_names = [result.resource_name for result in response.results]

        for resource_name in resource_names:
            print(f"Created sitelink asset with resource name '{resource_name}'.")

        return resource_names

    def _create_structured_snippet_assets(self, client, structured_snippet):
        """Creates structured snippet assets, which can be added to campaigns."""
        operations = []

        snippet_operation = client.get_type("AssetOperation")
        snippet_asset = snippet_operation.create
        snippet_asset.structured_snippet_asset.header = structured_snippet["header"]
        snippet_asset.structured_snippet_asset.values.extend(
            structured_snippet["values"]
        )
        operations.append(snippet_operation)

        asset_service = client.get_service("AssetService")
        response = asset_service.mutate_assets(
            customer_id=self.customer_id,
            operations=operations,
        )

        resource_names = [result.resource_name for result in response.results]

        for resource_name in resource_names:
            print(
                f"Created structured snippet asset with resource name '{resource_name}'."
            )

        return resource_names

    def _create_callout_assets(self, client, callouts):
        """Creates callout assets, which can be added to campaigns."""
        operations = []
        for callout_text in callouts:
            callout_operation = client.get_type("AssetOperation")
            callout_asset = callout_operation.create
            callout_asset.callout_asset.callout_text = callout_text
            operations.append(callout_operation)

        asset_service = client.get_service("AssetService")
        response = asset_service.mutate_assets(
            customer_id=self.customer_id,
            operations=operations,
        )

        resource_names = [result.resource_name for result in response.results]

        for resource_name in resource_names:
            print(f"Created callout asset with resource name '{resource_name}'.")

        return resource_names

    def _create_conversion_action(self, client, conversion_data):
        """Creates a conversion action for tracking."""
        conversion_action_service = client.get_service("ConversionActionService")
        conversion_action_operation = client.get_type("ConversionActionOperation")
        conversion_action = conversion_action_operation.create

        # Set the conversion action attributes
        conversion_action.name = conversion_data["name"]
        conversion_action.type_ = client.enums.ConversionActionTypeEnum[
            conversion_data["type"]
        ]
        conversion_action.category = client.enums.ConversionActionCategoryEnum[
            conversion_data["category"]
        ]
        conversion_action.status = client.enums.ConversionActionStatusEnum.ENABLED
        conversion_action.counting_type = (
            client.enums.ConversionActionCountingTypeEnum.ONE_PER_CLICK
        )

        # Create the conversion action
        try:
            response = conversion_action_service.mutate_conversion_actions(
                customer_id=self.customer_id, operations=[conversion_action_operation]
            )
            return response.results[0].resource_name
        except GoogleAdsException as ex:
            error_message = self.handle_googleads_exception(ex)
            if "name already exists" in error_message:
                return None
            raise Exception(error_message)
