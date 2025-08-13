import os
import json
import gspread
from datetime import datetime
from typing import Dict, Literal
from agency_swarm.tools import BaseTool
from pydantic import Field, model_validator

from oauth2client.service_account import ServiceAccountCredentials
from tools.ConductorAgent.util.constants import GoogleSheetTemplate
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()


class PostGoogleSheet(BaseTool):
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

    class ToolConfig:
        one_call_at_a_time = True

    def run(self):
        # Add data to Google Sheet
        json_data = self._shared_state.get("GENERATED_AD_COPY")
        if not json_data:
            raise Exception("No generated ad copy found")

        # Add data to Google Sheet
        try:
            sheet_url = self.add_to_google_sheet(json_data)
            return f"Data added to Google Sheet successfully. Sheet URL: {sheet_url}\n"
        except Exception as e:
            print(e)
            return f"Error adding data to Google Sheet: {e}\n"

    def add_to_google_sheet(self, data: Dict):
        # Set up Google Sheets API client
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets",
        ]
        key = json.loads(os.getenv("SERVICE_ACCOUNT_KEY_FIREBASE"))
        creds = ServiceAccountCredentials.from_json_keyfile_dict(key, scope)
        client = gspread.authorize(creds)

        # Create Drive API service
        drive_service = build("drive", "v3", credentials=creds)

        # Open template sheet
        template = client.open_by_key(os.getenv("GOOGLE_SHEET_TEMPLATE"))

        # Create a copy using Drive API
        copy_title = f"{self.campaign_title}"
        copied_file = {
            "name": copy_title,
            "parents": [os.getenv("GOOGLE_DRIVE_FOLDER_ID")]
            if os.getenv("GOOGLE_DRIVE_FOLDER_ID")
            else None,
        }

        copied_sheet = (
            drive_service.files()
            .copy(fileId=template.id, body=copied_file, supportsAllDrives=True)
            .execute()
        )

        # Open the newly created sheet
        sheet = client.open_by_key(copied_sheet["id"]).sheet1
        self._update_title(sheet, data)
        self._update_campaign(sheet, data)
        self._update_keywords(sheet, data)
        self._update_extensions(sheet, data)

        sheet_url = f"https://docs.google.com/spreadsheets/d/{copied_sheet['id']}"
        return sheet_url

    def _update_title(self, sheet, data):
        sheet.update_acell(
            GoogleSheetTemplate.Title.CAMPAIGN_NAME.value, self.campaign_title
        )

        start_datetime = datetime.strptime(self.start_date, "%Y%m%d")
        end_datetime = datetime.strptime(self.end_date, "%Y%m%d")
        start_date = start_datetime.strftime("%d/%m/%Y")
        end_date = end_datetime.strftime("%d/%m/%Y")
        campaign_dates = f"{start_date} - {end_date}"
        sheet.update_acell(GoogleSheetTemplate.Title.FLIGHT.value, campaign_dates)
        sheet.update_acell(
            GoogleSheetTemplate.Title.TOTAL_BUDGET.value, self.total_budget
        )

        campaign_days = (end_datetime - start_datetime).days + 1
        daily_budget = round(self.total_budget / campaign_days, 1)
        sheet.update_acell(
            GoogleSheetTemplate.Title.DAILY_AVG_SPEND.value, daily_budget
        )
        sheet.update_acell(
            GoogleSheetTemplate.Title.OBJECTIVE.value, self.campaign_type
        )

        conversion_actions = []
        for model in data:
            conversion_action = data[model]["extensions"]["conversion_action"]
            if conversion_action:  # Check if conversion_action exists
                conversion_actions.append(conversion_action["name"])
        sheet.update_acell(
            GoogleSheetTemplate.Title.CONVERSTION_ACTIONS.value,
            ", ".join(conversion_actions),
        )

    def _update_campaign(self, sheet, data):
        sheet.update_acell(GoogleSheetTemplate.Campaign.NAME.value, self.campaign_title)
        sheet.update_acell(
            GoogleSheetTemplate.Campaign.AD_GROUP.value,
            self.campaign_title + " Ad Group",
        )
        sheet.update_acell(
            GoogleSheetTemplate.Campaign.FINAL_URL.value, self._shared_state.get("URL")
        )

        for i, model in enumerate(data):
            offset = i * 5
            # Combined array of headlines and descriptions

            lines = data[model]["headlines"][:15]
            if len(lines) < 15:
                lines.extend([""] * (15 - len(lines)))
            lines.extend(data[model]["descriptions"][:5])

            start_cell = GoogleSheetTemplate.Campaign.HEADLINE_START.value
            # Extract the column letter from start_cell (e.g., if start_cell is 'C3', get 'C')
            row, col = gspread.utils.a1_to_rowcol(start_cell)

            # Create a list of lists containing lines and length pairs
            lines_data = [[line, len(line)] for line in lines]
            range_notation = (
                f"R{row}C{col + offset}:R{row + len(lines) - 1}C{col + offset + 1}"
            )
            sheet.update(range_name=range_notation, values=lines_data)

    def _update_keywords(self, sheet, data):
        sheet.update_acell(
            GoogleSheetTemplate.Keywords.NAME.value, self.campaign_title + " Ad Group"
        )
        lines = self._shared_state.get("KEYWORDS")

        start_cell = GoogleSheetTemplate.Keywords.KEYWORD_START.value
        row, col = gspread.utils.a1_to_rowcol(start_cell)

        # Create a list of lists containing keyword and match type pairs
        keyword_data = [
            [keyword["keyword"], keyword["match_type"].lower()] for keyword in lines
        ]
        for i, model in enumerate(data):
            offset = i * 5
            # Convert row/col numbers to A1 notation
            start_col_letter = gspread.utils.rowcol_to_a1(1, col + offset)[
                :-1
            ]  # Remove row number
            end_col_letter = gspread.utils.rowcol_to_a1(1, col + offset + 1)[
                :-1
            ]  # Remove row number
            range_notation = (
                f"{start_col_letter}{row}:{end_col_letter}{row + len(lines) - 1}"
            )

            sheet.format(
                range_notation,
                {
                    "borders": {
                        "top": {"style": "SOLID", "width": 1},
                        "bottom": {"style": "SOLID", "width": 1},
                        "left": {"style": "SOLID", "width": 1},
                        "right": {"style": "SOLID", "width": 1},
                    }
                },
            )
            # Update all keywords and match types in one batch operation
            sheet.update(range_name=range_notation, values=keyword_data)

    def _update_extensions(self, sheet, data):
        # Update sitelinks
        for i, model in enumerate(data):
            offset = i * 5

            lines = [
                [
                    text[key]
                    for key in ["link_text", "description1", "description2", "url"]

                ]
                for text in data[model]["extensions"]["sitelinks"]
                if "http" in text["url"]
            ]
            start_cell = GoogleSheetTemplate.AdExtensions.SITELINK_START.value
            row, col = gspread.utils.a1_to_rowcol(start_cell)
            # Convert row/col numbers to A1 notation
            start_col_letter = gspread.utils.rowcol_to_a1(1, col + offset)[
                :-1
            ]  # Remove row number
            end_col_letter = gspread.utils.rowcol_to_a1(1, col + offset + 3)[
                :-1
            ]  # +3 for 4 columns
            range_notation = (
                f"{start_col_letter}{row}:{end_col_letter}{row + len(lines) - 1}"
            )

            sheet.format(
                range_notation,
                {
                    "borders": {
                        "top": {"style": "SOLID", "width": 1},
                        "bottom": {"style": "SOLID", "width": 1},
                        "left": {"style": "SOLID", "width": 1},
                        "right": {"style": "SOLID", "width": 1},
                    }
                },
            )

            sheet.update(range_name=range_notation, values=lines)

        # Update callouts
        for i, model in enumerate(data):
            offset = i * 5

            lines = [[callout] for callout in data[model]["extensions"]["callouts"]]
            start_cell = GoogleSheetTemplate.AdExtensions.CALLOUT_START.value
            row, col = gspread.utils.a1_to_rowcol(start_cell)
            # Convert row/col numbers to A1 notation
            start_col_letter = gspread.utils.rowcol_to_a1(1, col + offset)[
                :-1
            ]  # Remove row number
            range_notation = (
                f"{start_col_letter}{row}:{start_col_letter}{row + len(lines) - 1}"
            )

            sheet.format(
                range_notation,
                {
                    "borders": {
                        "top": {"style": "SOLID", "width": 1},
                        "bottom": {"style": "SOLID", "width": 1},
                        "left": {"style": "SOLID", "width": 1},
                        "right": {"style": "SOLID", "width": 1},
                    }
                },
            )
            sheet.update(range_name=range_notation, values=lines)

        # Update structured snippet
        for i, model in enumerate(data):
            lines = [
                [value]
                for value in data[model]["extensions"]["structured_snippet"]["values"]
            ]
            lines.insert(
                0,
                [
                    f"Header: {data[model]['extensions']['structured_snippet']['header']}"
                ],
            )
            start_cell = GoogleSheetTemplate.AdExtensions.STRUCTURED_SNIPPET_START.value
            row, col = gspread.utils.a1_to_rowcol(start_cell)
            offset = i * 5
            # Convert row/col numbers to A1 notation
            start_col_letter = gspread.utils.rowcol_to_a1(1, col + offset)[
                :-1
            ]  # Remove row number
            range_notation = (
                f"{start_col_letter}{row}:{start_col_letter}{row + len(lines) - 1}"
            )

            sheet.format(
                range_notation,
                {
                    "borders": {
                        "top": {"style": "SOLID", "width": 1},
                        "bottom": {"style": "SOLID", "width": 1},
                        "left": {"style": "SOLID", "width": 1},
                        "right": {"style": "SOLID", "width": 1},
                    }
                },
            )
            sheet.update(range_name=range_notation, values=lines)

    # Might be useful later when adding campaigns to existsing sheet is supported
    def _find_last_column(sheet):
        """Find the last column that contains data across all rows in the sheet."""
        last_col = 0
        # Get all values from the sheet (up to row 135)
        values = sheet.get_values("A1:ZZ135")

        # Iterate through each row
        for row in values:
            # Find the last non-empty cell in this row
            for i, cell in enumerate(row, 1):
                if cell.strip():
                    last_col = max(last_col, i)

        return last_col
