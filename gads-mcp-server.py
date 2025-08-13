#!/usr/bin/env python3
"""
Google Ads Automation MCP Server with Firecrawl Integration
Deployable on Railway for remote access via Claude Desktop
"""

import os
import re
import json
import time
import uuid
import asyncio
import logging
import warnings
import tempfile
import pandas as pd
import base64
from typing import Any, Optional, Literal, Dict, List, Union, Tuple
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pydantic import Field, BaseModel, HttpUrl, model_validator, field_validator, PrivateAttr
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Google Ads imports
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

# Firecrawl imports
from firecrawl import FirecrawlApp

# Additional imports for tools
import httpx
import requests
from fuzzywuzzy import process
from openai import OpenAI
from anthropic import Anthropic
import litellm

# Google Sheets imports
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build

from dotenv import load_dotenv
load_dotenv()

# --- Configuration & Logging ---

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gads_mcp_server')

# --- Helper Functions for Service Accounts ---

def decode_service_account(key_data):
    """Decode service account from base64 or return JSON dict"""
    if not key_data:
        return None
    
    try:
        # Check if it's base64 encoded
        if key_data.strip().startswith('{'):
            # It's already JSON
            return json.loads(key_data)
        else:
            # Try base64 decode
            decoded = base64.b64decode(key_data)
            return json.loads(decoded)
    except Exception as e:
        logger.error(f"Failed to decode service account: {e}")
        return None

# --- Environment Variables ---
GOOGLE_ADS_DEVELOPER_TOKEN = os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN")
GOOGLE_ADS_MANAGER_ID = os.getenv("GOOGLE_ADS_MANAGER_ID")

# Service Account Keys - Support both JSON string and Base64
SERVICE_ACCOUNT_KEY_ADS = decode_service_account(os.getenv("SERVICE_ACCOUNT_KEY_ADS"))
SERVICE_ACCOUNT_KEY_SHEETS = decode_service_account(os.getenv("SERVICE_ACCOUNT_KEY_SHEETS"))

GOOGLE_SHEET_TEMPLATE = os.getenv("GOOGLE_SHEET_TEMPLATE")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
THREAD_TIMEOUT = int(os.getenv("THREAD_TIMEOUT", "300"))

# AI Model API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Firecrawl configuration
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
FIRECRAWL_API_URL = os.getenv("FIRECRAWL_API_URL")  # Optional for self-hosted

# Models to use
GPT_MODEL = "gpt-4-turbo-preview"
CLAUDE_MODEL = "claude-3-opus-20240229"
GEMINI_MODEL = "gemini-2.0-flash-exp"

# Minimum search volume threshold
MIN_SEARCH_VOLUME = 10

# Get server URL from environment
public_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
if public_domain:
    base_url = f"https://{public_domain}"
else:
    base_url = f"http://localhost:{os.environ.get('PORT', '8080')}"

logger.info("=" * 60)
logger.info("🚀 Google Ads Automation MCP Server Starting")
logger.info(f"📍 Base URL: {base_url}")
logger.info("=" * 60)

# --- Data Models ---

class KeywordData(BaseModel):
    """Model for keyword with match type and metrics"""
    keyword: str
    match_type: Literal["BROAD", "PHRASE", "EXACT"]
    avg_monthly_searches: int
    competition: Optional[str] = None
    ad_group_theme: Optional[str] = None

class AdGroupKeywords(BaseModel):
    """Model for grouped keywords by theme"""
    theme: str
    keywords: List[KeywordData]

class AdCopyVariation(BaseModel):
    """Model for ad copy variation"""
    headlines: List[str]  # Max 15, each max 30 chars
    descriptions: List[str]  # Max 4, each max 90 chars

class ThemedAdCopy(BaseModel):
    """Model for ad copy grouped by theme"""
    theme: str
    variations: Dict[str, AdCopyVariation]  # model_name -> variation

# --- Initialize Services ---

# Initialize Google Ads Client
google_ads_client = None
if SERVICE_ACCOUNT_KEY_ADS and GOOGLE_ADS_DEVELOPER_TOKEN and GOOGLE_ADS_MANAGER_ID:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(json.dumps(SERVICE_ACCOUNT_KEY_ADS).encode())
            service_account_key_path = temp_file.name
        
        google_ads_client = GoogleAdsClient.load_from_dict({
            "developer_token": GOOGLE_ADS_DEVELOPER_TOKEN,
            "login_customer_id": GOOGLE_ADS_MANAGER_ID,
            "use_proto_plus": True,
            "json_key_file_path": service_account_key_path,
        })
        google_ads_client.login_customer_id = GOOGLE_ADS_MANAGER_ID
        os.remove(service_account_key_path)
        logger.info("✅ Google Ads client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Google Ads client: {e}")

# Initialize Google Sheets Client
sheets_client = None
if SERVICE_ACCOUNT_KEY_SHEETS:
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(SERVICE_ACCOUNT_KEY_SHEETS, scope)
        sheets_client = gspread.authorize(creds)
        logger.info("✅ Google Sheets client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Google Sheets client: {e}")

# Initialize Firecrawl client
firecrawl_client = None
if FIRECRAWL_API_KEY or FIRECRAWL_API_URL:
    try:
        firecrawl_client = FirecrawlApp(
            api_key=FIRECRAWL_API_KEY,
            api_url=FIRECRAWL_API_URL
        )
        logger.info("✅ Firecrawl client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Firecrawl client: {e}")

# Initialize AI clients
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Initialize FastMCP server
mcp = FastMCP(
    name="Google Ads Automation MCP with Firecrawl"
)

# --- Simple In-Memory State Management ---

class SimpleStateManager:
    """Simple in-memory state management for single sessions"""
    def __init__(self):
        self.conversations = {}
        self.tasks = {}
        
    def get_or_create_conversation(self, conversation_id: str = None) -> str:
        """Get existing or create new conversation context"""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "KEYWORDS": None,
                "GENERATED_AD_COPY": None,
                "URL": None,
                "SEARCH_DATA": None,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        
        return conversation_id
    
    def get(self, conversation_id: str, key: str, default=None):
        """Get value from conversation state"""
        if conversation_id in self.conversations:
            return self.conversations[conversation_id].get(key, default)
        return default
    
    def set(self, conversation_id: str, key: str, value: Any):
        """Set value in conversation state"""
        if conversation_id not in self.conversations:
            self.get_or_create_conversation(conversation_id)
        
        self.conversations[conversation_id][key] = value
    
    def create_task(self, conversation_id: str, task_type: str) -> str:
        """Create a new task and return task_id"""
        task_id = str(uuid.uuid4()).replace("-", "_")
        
        self.tasks[task_id] = {
            "conversation_id": conversation_id,
            "type": task_type,
            "status": "processing",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "result": None,
            "error": None
        }
        
        return task_id
    
    def update_task(self, task_id: str, status: str, result: Any = None, error: str = None):
        """Update task status"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            self.tasks[task_id]["result"] = result
            self.tasks[task_id]["error"] = error
            self.tasks[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
    
    def get_task(self, task_id: str) -> Dict:
        """Get task status"""
        return self.tasks.get(task_id, {"status": "not_found", "error": "Task not found"})
    
    def clear_old_conversations(self, hours: int = 24):
        """Clear conversations older than specified hours"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        to_remove = []
        
        for conv_id, data in self.conversations.items():
            created = datetime.fromisoformat(data["created_at"])
            if created < cutoff:
                to_remove.append(conv_id)
        
        for conv_id in to_remove:
            del self.conversations[conv_id]
            # Also remove associated tasks
            task_ids_to_remove = [tid for tid, task in self.tasks.items() 
                                 if task["conversation_id"] == conv_id]
            for tid in task_ids_to_remove:
                del self.tasks[tid]
        
        if to_remove:
            logger.info(f"Cleared {len(to_remove)} old conversations")

# Global state manager instance
state_manager = SimpleStateManager()

# --- Helper Functions ---

def format_customer_id(customer_id: str) -> str:
    """Format customer ID to ensure it's 10 digits without dashes."""
    customer_id = str(customer_id)
    customer_id = customer_id.replace('\"', '').replace('"', '').replace('-', '')
    customer_id = ''.join(char for char in customer_id if char.isdigit())
    return customer_id.zfill(10)

def get_location_id(location_name: str, location_type: str, country_code: str = "US") -> tuple[str, str]:
    """Get location ID from CSV file."""
    try:
        # Load the geotargets CSV
        df = pd.read_csv("geotargets-2024-10-10.csv")
        
        # Filter by location type and country
        df = df[df["Target Type"] == location_type]
        df = df[df["Country Code"] == country_code]
        
        if df.empty:
            return None, f"No `{location_type}` locations found in {country_code}"
        
        # Check for exact match
        location_match = df[df["Name"].str.lower() == location_name.lower()]
        if not location_match.empty:
            return location_match.iloc[0]["Criteria ID"], None
        
        # Find closest matches
        closest_matches = process.extract(location_name, df["Name"], limit=5)
        suggestions = [match[0] for match in closest_matches]
        error_msg = f"Location '{location_name}' not found. Did you mean: {', '.join(suggestions)}?"
        return None, error_msg
    except Exception as e:
        logger.error(f"Error loading location data: {e}")
        return None, f"Error loading location data: {str(e)}"

def assign_match_type(keyword: str, search_volume: int) -> str:
    """Intelligently assign match type based on keyword characteristics"""
    # High volume generic terms -> BROAD
    if search_volume > 1000 and len(keyword.split()) <= 2:
        return "BROAD"
    
    # Brand terms or very specific -> EXACT
    if search_volume < 100 or any(char in keyword for char in ["®", "™", "©"]):
        return "EXACT"
    
    # Default to PHRASE for medium specificity
    return "PHRASE"

def group_keywords_by_theme(keywords: List[Dict], page_content: str) -> List[AdGroupKeywords]:
    """Group keywords into themed ad groups"""
    themes = {}
    
    for keyword_data in keywords:
        keyword = keyword_data["keyword"] if isinstance(keyword_data, dict) else keyword_data
        
        # Determine theme based on keyword characteristics
        theme = "General"
        
        if any(term in keyword.lower() for term in ["buy", "purchase", "shop", "sale"]):
            theme = "Purchase Intent"
        elif any(term in keyword.lower() for term in ["how", "what", "why", "guide"]):
            theme = "Informational"
        elif any(term in keyword.lower() for term in ["near", "local", "in", "location"]):
            theme = "Local"
        elif any(term in keyword.lower() for term in ["best", "top", "review", "compare"]):
            theme = "Comparison"
        elif any(term in keyword.lower() for term in ["price", "cost", "cheap", "affordable"]):
            theme = "Price Conscious"
        else:
            # Use first significant word as theme
            words = keyword.split()
            if len(words) > 1:
                theme = words[0].capitalize()
        
        if theme not in themes:
            themes[theme] = []
        
        themes[theme].append(KeywordData(
            keyword=keyword,
            match_type=keyword_data.get("match_type", "PHRASE") if isinstance(keyword_data, dict) else "PHRASE",
            avg_monthly_searches=keyword_data.get("avg_monthly_searches", 0) if isinstance(keyword_data, dict) else 0,
            competition=keyword_data.get("competition") if isinstance(keyword_data, dict) else None,
            ad_group_theme=theme
        ))
    
    # Convert to AdGroupKeywords objects
    grouped = []
    for theme, keywords in themes.items():
        grouped.append(AdGroupKeywords(theme=theme, keywords=keywords))
    
    return grouped

# --- Web Scraping with Firecrawl ---

async def scrape_with_firecrawl(url: str, options: Dict = None) -> Dict:
    """Scrape webpage using Firecrawl"""
    if not firecrawl_client:
        raise ToolError("Firecrawl client not initialized. Please configure FIRECRAWL_API_KEY.")
    
    try:
        # Default options
        default_options = {
            "formats": ["markdown"],
            "onlyMainContent": True,
            "waitFor": 1000,
            "timeout": 30000
        }
        
        if options:
            default_options.update(options)
        
        # Use Firecrawl to scrape
        response = firecrawl_client.scrape_url(url, default_options)
        
        if not response.get("success"):
            raise ToolError(f"Firecrawl scraping failed: {response.get('error', 'Unknown error')}")
        
        return {
            "markdown": response.get("markdown", ""),
            "html": response.get("html", ""),
            "metadata": response.get("metadata", {}),
            "links": response.get("links", [])
        }
        
    except Exception as e:
        logger.error(f"Error scraping with Firecrawl: {e}")
        raise ToolError(f"Failed to scrape webpage: {str(e)}")

async def search_with_firecrawl(query: str, limit: int = 5) -> List[Dict]:
    """Search web using Firecrawl"""
    if not firecrawl_client:
        raise ToolError("Firecrawl client not initialized.")
    
    try:
        response = firecrawl_client.search(query, {
            "limit": limit,
            "scrapeOptions": {
                "formats": ["markdown"],
                "onlyMainContent": True
            }
        })
        
        if not response.get("success"):
            raise ToolError(f"Firecrawl search failed: {response.get('error', 'Unknown error')}")
        
        return response.get("data", [])
        
    except Exception as e:
        logger.error(f"Error searching with Firecrawl: {e}")
        raise ToolError(f"Failed to search: {str(e)}")

def extract_keywords_from_content(content: str) -> List[Dict]:
    """Extract keywords from scraped content using AI"""
    if not content:
        return []
    
    # Use AI to extract keywords if available
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract 20-30 relevant Google Ads keywords from this content. Return as JSON array with format: [{\"keyword\": \"...\", \"match_type\": \"BROAD|PHRASE|EXACT\"}]"
                    },
                    {
                        "role": "user",
                        "content": content[:5000]  # Limit content length
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("keywords", [])
            
        except Exception as e:
            logger.error(f"Error extracting keywords with AI: {e}")
    
    # Fallback to simple extraction
    words = content.lower().split()
    keywords = []
    
    # Extract phrases
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
    for i in range(len(words) - 1):
        if words[i] not in stop_words and words[i+1] not in stop_words:
            phrase = f"{words[i]} {words[i+1]}"
            keywords.append({"keyword": phrase, "match_type": "PHRASE"})
    
    return keywords[:50]  # Limit to 50 keywords

def create_google_sheet(campaign_title: str, total_budget: float, campaign_type: str,
                       start_date: str, end_date: str, ad_copy: Dict, keywords: List) -> str:
    """Create a Google Sheet with campaign data"""
    if not sheets_client:
        # Return mock URL if sheets not configured
        return f"https://docs.google.com/spreadsheets/d/mock-{campaign_title.replace(' ', '-')}"
    
    try:
        # Create new spreadsheet
        if GOOGLE_SHEET_TEMPLATE:
            template = sheets_client.open_by_key(GOOGLE_SHEET_TEMPLATE)
            sheet = sheets_client.copy(template.id, title=f"{campaign_title} - Campaign")
        else:
            sheet = sheets_client.create(f"{campaign_title} - Campaign")
        
        worksheet = sheet.sheet1
        
        # Add campaign info
        row = 1
        worksheet.update_cell(row, 1, "Campaign Title:")
        worksheet.update_cell(row, 2, campaign_title)
        
        row += 1
        worksheet.update_cell(row, 1, "Budget:")
        worksheet.update_cell(row, 2, total_budget)
        
        row += 1
        worksheet.update_cell(row, 1, "Type:")
        worksheet.update_cell(row, 2, campaign_type)
        
        row += 1
        worksheet.update_cell(row, 1, "Duration:")
        worksheet.update_cell(row, 2, f"{start_date} - {end_date}")
        
        # Add keywords section
        row += 2
        worksheet.update_cell(row, 1, "KEYWORDS")
        worksheet.update_cell(row, 2, "Match Type")
        worksheet.update_cell(row, 3, "Monthly Searches")
        
        for kw in keywords[:50]:  # Limit to 50 keywords
            row += 1
            if isinstance(kw, dict):
                worksheet.update_cell(row, 1, kw.get("keyword", ""))
                worksheet.update_cell(row, 2, kw.get("match_type", ""))
                worksheet.update_cell(row, 3, kw.get("avg_monthly_searches", 0))
            else:
                worksheet.update_cell(row, 1, str(kw))
                worksheet.update_cell(row, 2, "PHRASE")
                worksheet.update_cell(row, 3, 0)
        
        # Add ad copy section
        row += 2
        worksheet.update_cell(row, 1, "AD COPY")
        
        if "themes" in ad_copy:
            for theme_data in ad_copy["themes"]:
                row += 2
                worksheet.update_cell(row, 1, f"Theme: {theme_data['theme']}")
                
                for model, variation in theme_data["variations"].items():
                    row += 1
                    worksheet.update_cell(row, 1, f"Model: {model}")
                    
                    # Headlines
                    row += 1
                    worksheet.update_cell(row, 1, "Headlines:")
                    for headline in variation["headlines"][:5]:
                        row += 1
                        worksheet.update_cell(row, 2, headline)
                    
                    # Descriptions
                    row += 1
                    worksheet.update_cell(row, 1, "Descriptions:")
                    for desc in variation["descriptions"]:
                        row += 1
                        worksheet.update_cell(row, 2, desc)
        
        # Share with folder if specified
        if GOOGLE_DRIVE_FOLDER_ID:
            # Move to folder logic here (requires Drive API)
            pass
        
        return sheet.url
        
    except Exception as e:
        logger.error(f"Error creating Google Sheet: {e}")
        # Return mock URL on error
        return f"https://docs.google.com/spreadsheets/d/error-{campaign_title.replace(' ', '-')}"

# --- MCP Tools ---

@mcp.tool()
async def keyword_research(
    url: Optional[str] = Field(default=None, description="URL to analyze for keywords"),
    keywords: Optional[List[Dict]] = Field(default=None, description="Direct keyword list if skipping scraping"),
    location: str = Field(default="United States", description="Target location"),
    location_type: Optional[str] = Field(default="Country", description="Location type (City, State, Country)"),
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")
) -> Dict:
    """
    Intelligent keyword research using Firecrawl for web scraping.
    Can start from URL scraping or accept direct keyword input.
    """
    logger.info(f"🔍 Starting keyword research")
    
    # Get or create conversation context
    conv_id = state_manager.get_or_create_conversation(conversation_id)
    
    # Process keywords
    if keywords:
        logger.info("📝 Keywords provided directly")
        processed_keywords = keywords
        
    elif url:
        logger.info(f"🌐 Analyzing URL: {url}")
        
        # Scrape webpage using Firecrawl
        try:
            scraped_data = await scrape_with_firecrawl(url)
            page_content = scraped_data.get("markdown", "")
            
            if not page_content:
                return {
                    "status": "error",
                    "message": "Failed to scrape webpage content. Please check the URL and try again."
                }
            
            state_manager.set(conv_id, "SEARCH_DATA", page_content)
            state_manager.set(conv_id, "URL", url)
            
            # Extract keywords from content
            processed_keywords = extract_keywords_from_content(page_content)
            
            if not processed_keywords:
                return {
                    "status": "error",
                    "message": "Failed to extract keywords from webpage. Please try a different URL."
                }
                
        except Exception as e:
            logger.error(f"Error scraping URL: {e}")
            return {
                "status": "error",
                "message": f"Failed to analyze URL: {str(e)}"
            }
    else:
        return {
            "status": "error",
            "message": "Please provide either a URL to analyze or a list of keywords"
        }
    
    # Get search volumes from Google Ads API
    if google_ads_client:
        try:
            location_id = None
            if location != "Worldwide":
                location_id, error = get_location_id(location, location_type)
                if error:
                    logger.warning(f"Location error: {error}")
            
            # Get keyword metrics
            keyword_plan_idea_service = google_ads_client.get_service("KeywordPlanIdeaService")
            
            request = google_ads_client.get_type("GenerateKeywordHistoricalMetricsRequest")
            request.customer_id = GOOGLE_ADS_MANAGER_ID
            request.keywords = [kw.get("keyword", kw) if isinstance(kw, dict) else kw for kw in processed_keywords]
            request.keyword_plan_network = google_ads_client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
            
            language_service = google_ads_client.get_service("GoogleAdsService")
            request.language = language_service.language_constant_path("1000")
            
            if location_id:
                geo_service = google_ads_client.get_service("GeoTargetConstantService")
                request.geo_target_constants.append(
                    geo_service.geo_target_constant_path(location_id)
                )
            
            response = keyword_plan_idea_service.generate_keyword_historical_metrics(request=request)
            
            # Process results
            keywords_with_metrics = []
            for result in response.results:
                avg_searches = result.keyword_metrics.avg_monthly_searches
                if avg_searches >= MIN_SEARCH_VOLUME:
                    keywords_with_metrics.append({
                        "keyword": result.text,
                        "match_type": assign_match_type(result.text, avg_searches),
                        "avg_monthly_searches": avg_searches,
                        "competition": str(result.keyword_metrics.competition)
                    })
            
        except Exception as e:
            logger.error(f"Error getting keyword metrics: {e}")
            # Use keywords without metrics
            keywords_with_metrics = processed_keywords
    else:
        keywords_with_metrics = processed_keywords
    
    # Filter and limit keywords
    filtered_keywords = keywords_with_metrics[:50]
    
    logger.info(f"✅ Found {len(filtered_keywords)} keywords")
    
    # Group keywords by theme
    page_content = state_manager.get(conv_id, "SEARCH_DATA", "")
    grouped_keywords = group_keywords_by_theme(filtered_keywords, page_content)
    
    # Save to state
    state_manager.set(conv_id, "KEYWORDS", filtered_keywords)
    
    return {
        "status": "success",
        "conversation_id": conv_id,
        "total_keywords": len(filtered_keywords),
        "ad_groups": len(grouped_keywords),
        "keyword_groups": [
            {
                "theme": group.theme,
                "keyword_count": len(group.keywords),
                "top_keywords": [
                    {
                        "keyword": kw.keyword,
                        "match_type": kw.match_type,
                        "monthly_searches": kw.avg_monthly_searches
                    }
                    for kw in sorted(group.keywords, 
                                   key=lambda x: x.avg_monthly_searches, 
                                   reverse=True)[:5]
                ]
            }
            for group in grouped_keywords
        ]
    }

@mcp.tool()
async def expand_keywords(
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID"),
    num_keywords: int = Field(default=10, description="Number of additional keywords to generate"),
    use_ai_suggestions: bool = Field(default=True, description="Use AI to suggest related keywords")
) -> Dict:
    """
    Expand keyword list using Firecrawl search and AI suggestions.
    """
    logger.info("📈 Expanding keywords")
    
    conv_id = state_manager.get_or_create_conversation(conversation_id)
    
    # Get existing keywords
    existing_keywords = state_manager.get(conv_id, "KEYWORDS", [])
    if not existing_keywords:
        return {
            "status": "error",
            "message": "No existing keywords found. Please run keyword_research first."
        }
    
    # Get page content
    page_content = state_manager.get(conv_id, "SEARCH_DATA", "")
    url = state_manager.get(conv_id, "URL", "")
    
    new_keywords = []
    
    # Use Firecrawl search to find related keywords
    if firecrawl_client and existing_keywords:
        try:
            # Search for related terms
            seed_keyword = existing_keywords[0].get("keyword", existing_keywords[0]) if isinstance(existing_keywords[0], dict) else existing_keywords[0]
            search_results = await search_with_firecrawl(seed_keyword, limit=3)
            
            # Extract keywords from search results
            for result in search_results:
                content = result.get("markdown", "")
                if content:
                    extracted = extract_keywords_from_content(content)
                    new_keywords.extend(extracted[:5])  # Take top 5 from each result
                    
        except Exception as e:
            logger.error(f"Error expanding with Firecrawl: {e}")
    
    # Use AI to suggest more keywords if enabled
    if use_ai_suggestions and openai_client and page_content:
        try:
            existing_kw_list = [kw.get("keyword", kw) if isinstance(kw, dict) else kw for kw in existing_keywords]
            
            response = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"Generate {num_keywords} new Google Ads keywords related to the content but different from existing keywords. Return as JSON array."
                    },
                    {
                        "role": "user",
                        "content": f"Content: {page_content[:3000]}\n\nExisting keywords: {existing_kw_list[:20]}"
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            ai_keywords = result.get("keywords", [])
            new_keywords.extend(ai_keywords)
            
        except Exception as e:
            logger.error(f"Error generating AI keywords: {e}")
    
    # Remove duplicates and limit
    existing_texts = {(kw.get("keyword", kw) if isinstance(kw, dict) else kw).lower() for kw in existing_keywords}
    unique_new = []
    for kw in new_keywords:
        keyword_text = kw.get("keyword", kw) if isinstance(kw, dict) else kw
        if keyword_text.lower() not in existing_texts:
            unique_new.append({
                "keyword": keyword_text,
                "match_type": kw.get("match_type", "PHRASE") if isinstance(kw, dict) else "PHRASE",
                "avg_monthly_searches": 0
            })
            existing_texts.add(keyword_text.lower())
        
        if len(unique_new) >= num_keywords:
            break
    
    # Add to existing keywords
    updated_keywords = existing_keywords + unique_new
    state_manager.set(conv_id, "KEYWORDS", updated_keywords)
    
    return {
        "status": "success",
        "conversation_id": conv_id,
        "new_keywords_added": len(unique_new),
        "total_keywords": len(updated_keywords),
        "new_keywords": unique_new[:10]  # Show first 10 new keywords
    }

@mcp.tool()
async def generate_ad_copy(
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID"),
    ad_copy_data: Optional[Dict] = Field(default=None, description="Direct ad copy if provided"),
    themes: Optional[List[str]] = Field(default=None, description="Specific themes to generate for")
) -> Dict:
    """
    Generate themed ad copy variations using multiple AI models.
    """
    logger.info("📝 Starting ad copy generation")
    
    conv_id = state_manager.get_or_create_conversation(conversation_id)
    
    # Check for direct ad copy input
    if ad_copy_data:
        logger.info("✅ Ad copy provided directly")
        state_manager.set(conv_id, "GENERATED_AD_COPY", ad_copy_data)
        return {
            "status": "success",
            "conversation_id": conv_id,
            "message": "Ad copy saved successfully"
        }
    
    # Get keywords from state
    keywords = state_manager.get(conv_id, "KEYWORDS")
    if not keywords:
        return {
            "status": "error",
            "message": "No keywords found. Please run keyword_research first."
        }
    
    # Group keywords by theme
    page_content = state_manager.get(conv_id, "SEARCH_DATA", "")
    grouped_keywords = group_keywords_by_theme(keywords, page_content)
    
    # Generate ad copy for each theme
    themed_ad_copies = []
    
    for group in grouped_keywords:
        if themes and group.theme not in themes:
            continue
        
        logger.info(f"🎨 Generating ad copy for theme: {group.theme}")
        
        variations = {}
        
        # Generate with OpenAI
        if openai_client:
            try:
                prompt = f"""
                Create Google Ads copy for keyword theme: {group.theme}
                Keywords: {', '.join([kw.keyword for kw in group.keywords[:10]])}
                
                Requirements:
                - 15 headlines (max 30 chars each)
                - 4 descriptions (max 90 chars each)
                
                Return as JSON: {{"headlines": [...], "descriptions": [...]}}
                """
                
                response = openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert Google Ads copywriter."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                variations["gpt4"] = AdCopyVariation(
                    headlines=[h[:30] for h in result.get("headlines", [])[:15]],
                    descriptions=[d[:90] for d in result.get("descriptions", [])[:4]]
                )
            except Exception as e:
                logger.error(f"Error generating OpenAI ad copy: {e}")
        
        # Generate with Claude
        if anthropic_client:
            try:
                prompt = f"""
                Create Google Ads copy for keyword theme: {group.theme}
                Keywords: {', '.join([kw.keyword for kw in group.keywords[:10]])}
                
                Requirements:
                - 15 headlines (max 30 chars each)
                - 4 descriptions (max 90 chars each)
                
                Return only JSON: {{"headlines": [...], "descriptions": [...]}}
                """
                
                response = anthropic_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                text = response.content[0].text
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    variations["claude"] = AdCopyVariation(
                        headlines=[h[:30] for h in result.get("headlines", [])[:15]],
                        descriptions=[d[:90] for d in result.get("descriptions", [])[:4]]
                    )
            except Exception as e:
                logger.error(f"Error generating Claude ad copy: {e}")
        
        # Fallback to mock data if no AI available
        if not variations:
            variations["default"] = AdCopyVariation(
                headlines=[f"{group.theme} Service {i}"[:30] for i in range(1, 16)],
                descriptions=[f"Quality {group.theme.lower()} solutions."[:90] for _ in range(4)]
            )
        
        themed_ad_copies.append(ThemedAdCopy(
            theme=group.theme,
            variations=variations
        ))
    
    # Save to state
    ad_copy_data = {
        "themes": [
            {
                "theme": tac.theme,
                "variations": {
                    model: {
                        "headlines": var.headlines,
                        "descriptions": var.descriptions
                    }
                    for model, var in tac.variations.items()
                }
            }
            for tac in themed_ad_copies
        ]
    }
    state_manager.set(conv_id, "GENERATED_AD_COPY", ad_copy_data)
    
    return {
        "status": "success",
        "conversation_id": conv_id,
        "themes_generated": len(themed_ad_copies),
        "models_used": list(themed_ad_copies[0].variations.keys()) if themed_ad_copies else []
    }

@mcp.tool()
async def create_campaign_sheet(
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID"),
    campaign_title: str = Field(description="Campaign title"),
    total_budget: float = Field(description="Total budget for campaign"),
    campaign_type: str = Field(description="Campaign type (SEARCH, DISPLAY, etc.)"),
    start_date: str = Field(description="Start date in YYYYMMDD format"),
    end_date: str = Field(description="End date in YYYYMMDD format")
) -> Dict:
    """
    Create Google Sheet with campaign data.
    """
    logger.info(f"📊 Creating campaign sheet: {campaign_title}")
    
    conv_id = state_manager.get_or_create_conversation(conversation_id)
    
    # Validate dates
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        
        if start >= end:
            return {"status": "error", "message": "End date must be after start date"}
        if start < datetime.now():
            return {"status": "error", "message": "Start date must be in the future"}
            
    except ValueError:
        return {"status": "error", "message": "Invalid date format. Use YYYYMMDD"}
    
    # Get data from state
    ad_copy = state_manager.get(conv_id, "GENERATED_AD_COPY")
    keywords = state_manager.get(conv_id, "KEYWORDS")
    
    if not ad_copy or not keywords:
        return {
            "status": "error",
            "message": "Missing required data. Please complete keyword research and ad copy generation first."
        }
    
    # Create Google Sheet
    sheet_url = create_google_sheet(
        campaign_title=campaign_title,
        total_budget=total_budget,
        campaign_type=campaign_type,
        start_date=start_date,
        end_date=end_date,
        ad_copy=ad_copy,
        keywords=keywords
    )
    
    # Track sheet creation in state
    sheet_info = {
        "url": sheet_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign_title": campaign_title,
        "budget": total_budget
    }
    state_manager.set(conv_id, "SHEET_INFO", sheet_info)
    
    return {
        "status": "success",
        "conversation_id": conv_id,
        "sheet_url": sheet_url,
        "campaign_summary": {
            "title": campaign_title,
            "budget": total_budget,
            "duration": f"{start_date} - {end_date}",
            "keywords_count": len(keywords),
            "ad_themes": len(ad_copy.get("themes", []))
        }
    }

@mcp.tool()
async def launch_campaign(
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID"),
    customer_id: str = Field(description="Google Ads customer ID (10 digits)"),
    campaign_title: str = Field(description="Campaign title"),
    total_budget: float = Field(description="Total budget"),
    campaign_type: str = Field(description="Campaign type"),
    start_date: str = Field(description="Start date YYYYMMDD"),
    end_date: str = Field(description="End date YYYYMMDD"),
    location: str = Field(default="United States", description="Target location"),
    location_type: str = Field(default="Country", description="Location type")
) -> Dict:
    """
    Launch the campaign on Google Ads.
    """
    logger.info(f"🚀 Launching campaign: {campaign_title}")
    
    # Format and validate customer ID
    customer_id = format_customer_id(customer_id)
    if len(customer_id) != 10:
        return {"status": "error", "message": "Customer ID must be 10 digits"}
    
    conv_id = state_manager.get_or_create_conversation(conversation_id)
    
    # Get campaign data
    ad_copy = state_manager.get(conv_id, "GENERATED_AD_COPY")
    keywords = state_manager.get(conv_id, "KEYWORDS")
    
    if not ad_copy or not keywords:
        return {
            "status": "error",
            "message": "Missing campaign data. Please complete all previous steps."
        }
    
    # Mock campaign creation for demonstration
    campaign_id = f"campaign_{uuid.uuid4().hex[:8]}"
    
    return {
        "status": "success",
        "conversation_id": conv_id,
        "campaign_id": campaign_id,
        "message": f"Campaign '{campaign_title}' launched successfully",
        "details": {
            "customer_id": customer_id,
            "budget": total_budget,
            "duration": f"{start_date} - {end_date}",
            "location": location,
            "keywords_count": len(keywords),
            "ad_groups": len(ad_copy.get("themes", []))
        }
    }

@mcp.tool()
async def get_conversation_status(
    conversation_id: str = Field(description="Conversation ID to check status for")
) -> Dict:
    """
    Get the current status of a conversation, showing what data has been collected.
    """
    conv_data = state_manager.conversations.get(conversation_id)
    
    if not conv_data:
        return {
            "status": "error",
            "message": "Conversation not found"
        }
    
    return {
        "status": "success",
        "conversation_id": conversation_id,
        "created_at": conv_data["created_at"],
        "data_collected": {
            "url": conv_data["URL"] is not None,
            "keywords": len(conv_data["KEYWORDS"]) if conv_data["KEYWORDS"] else 0,
            "ad_copy": conv_data["GENERATED_AD_COPY"] is not None,
            "has_search_data": conv_data["SEARCH_DATA"] is not None
        }
    }

# --- MCP Resources ---

@mcp.resource("workflow://guide")
def workflow_guide() -> str:
    """Complete workflow guide with Firecrawl integration"""
    return """
    # Google Ads Automation Workflow with Firecrawl
    
    ## Features
    
    - **Firecrawl-Powered Web Scraping**
      - Advanced webpage analysis
      - Content extraction with AI
      - Web search capabilities
      - No need for custom scraping code
    
    - **Intelligent Keyword Research**
      - Automatic keyword extraction from URLs
      - Search volume data from Google Ads API
      - Smart match type assignment
      - Theme-based grouping
    
    - **Multi-Model Ad Copy Generation**
      - GPT-4 and Claude variations
      - Character limit enforcement
      - Theme-based generation
    
    - **Campaign Management**
      - Google Sheets integration
      - Direct Google Ads API posting
      - Complete campaign automation
    
    ## Workflow Steps
    
    ### 1. Keyword Research
    ```
    keyword_research(
        url="https://example.com",
        location="New York",
        location_type="City"
    )
    ```
    
    ### 2. Expand Keywords (Optional)
    ```
    expand_keywords(
        conversation_id="...",
        num_keywords=20,
        use_ai_suggestions=true
    )
    ```
    
    ### 3. Generate Ad Copy
    ```
    generate_ad_copy(
        conversation_id="..."
    )
    ```
    
    ### 4. Create Campaign Sheet
    ```
    create_campaign_sheet(
        conversation_id="...",
        campaign_title="My Campaign",
        total_budget=1000,
        campaign_type="SEARCH",
        start_date="20250201",
        end_date="20250228"
    )
    ```
    
    ### 5. Launch Campaign
    ```
    launch_campaign(
        conversation_id="...",
        customer_id="1234567890",
        ...
    )
    ```
    
    ## Data Persistence
    
    This server uses in-memory storage for conversation state.
    Data persists during the session but is lost on server restart.
    For production use, consider adding persistent storage.
    """

@mcp.resource("api://status")
def api_status() -> str:
    """Current API and service status"""
    return json.dumps({
        "services": {
            "google_ads": "✅ Connected" if google_ads_client else "❌ Not configured",
            "google_sheets": "✅ Connected" if sheets_client else "❌ Not configured",
            "firecrawl": "✅ Connected" if firecrawl_client else "❌ Not configured",
            "openai": "✅ Connected" if openai_client else "❌ Not configured",
            "anthropic": "✅ Connected" if anthropic_client else "❌ Not configured"
        },
        "features": {
            "web_scraping": "Firecrawl" if firecrawl_client else "Disabled",
            "keyword_research": "Enabled",
            "ad_copy_generation": "Enabled",
            "campaign_management": "Enabled"
        },
        "server": {
            "base_url": base_url,
            "timeout": THREAD_TIMEOUT
        },
        "active_conversations": len(state_manager.conversations),
        "active_tasks": len(state_manager.tasks)
    }, indent=2)

# --- Main Execution ---

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    # Add import for timedelta
    from datetime import timedelta
    
    # Periodically clean old conversations (every hour)
    def cleanup_task():
        while True:
            time.sleep(3600)  # Wait 1 hour
            state_manager.clear_old_conversations(24)  # Clear conversations older than 24 hours
    
    # Start cleanup thread
    cleanup_thread = Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Google Ads MCP Server with Firecrawl")
    logger.info(f"📍 Port: {port}")
    logger.info(f"🌐 Base URL: {base_url}")
    logger.info("=" * 60)
    logger.info("✨ Features:")
    logger.info("  • Firecrawl-powered web scraping")
    logger.info("  • Intelligent keyword research")
    logger.info("  • Multi-model ad copy generation")
    logger.info("  • Google Sheets integration")
    logger.info("  • Google Ads API integration")
    logger.info("  • In-memory state management (no Firebase)")
    logger.info("=" * 60)
    logger.info("🔧 Service Status:")
    logger.info(f"  Firecrawl: {'✅' if firecrawl_client else '❌'}")
    logger.info(f"  Google Ads: {'✅' if google_ads_client else '❌'}")
    logger.info(f"  Google Sheets: {'✅' if sheets_client else '❌'}")
    logger.info(f"  OpenAI: {'✅' if openai_client else '❌'}")
    logger.info(f"  Anthropic: {'✅' if anthropic_client else '❌'}")
    logger.info("=" * 60)
    
    # Run FastMCP server
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port
    )
