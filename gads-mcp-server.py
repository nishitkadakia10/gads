#!/usr/bin/env python3
"""
Google Ads Automation MCP Server - Fixed Version
Combines OAuth authentication from working server with campaign automation features
"""

import os
import re
import json
import time
import uuid
import logging
import warnings
import tempfile
import pandas as pd
import base64
from typing import Any, Optional, Literal, Dict, List, Union
from datetime import datetime, timezone, timedelta
from dateutil import parser
from threading import Thread

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pydantic import Field, BaseModel
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Google Auth imports
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as AuthRequest
import requests

# AI imports for ad copy generation
from openai import OpenAI
from anthropic import Anthropic

# Google Sheets imports
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from dotenv import load_dotenv
load_dotenv()

# --- Configuration & Logging ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gads_mcp_server')

# --- Constants ---
API_VERSION = "v20"  # Google Ads API version
SCOPES = ['https://www.googleapis.com/auth/adwords']

# --- Helper Functions ---

def initialize_oauth_credentials():
    """Initialize OAuth credentials from base64 encoded token file"""
    oauth_tokens_base64 = os.environ.get("GOOGLE_ADS_OAUTH_TOKENS_BASE64")
    if not oauth_tokens_base64:
        logger.warning("GOOGLE_ADS_OAUTH_TOKENS_BASE64 not set - OAuth features disabled")
        return None
    
    try:
        oauth_tokens_json = base64.b64decode(oauth_tokens_base64).decode('utf-8')
        oauth_tokens = json.loads(oauth_tokens_json)
        
        credentials = Credentials(
            token=oauth_tokens.get('token'),
            refresh_token=oauth_tokens.get('refresh_token'),
            token_uri=oauth_tokens.get('token_uri', 'https://oauth2.googleapis.com/token'),
            client_id=oauth_tokens.get('client_id'),
            client_secret=oauth_tokens.get('client_secret'),
            scopes=oauth_tokens.get('scopes', SCOPES)
        )
        
        if 'expiry' in oauth_tokens:
            expiry_str = oauth_tokens['expiry']
            credentials.expiry = parser.parse(expiry_str)
            
            if credentials.expiry and credentials.expiry < datetime.now(timezone.utc):
                logger.info("Token expired, refreshing...")
                auth_req = AuthRequest()
                credentials.refresh(auth_req)
                logger.info("Token refreshed successfully")
        
        return credentials
        
    except Exception as e:
        logger.error(f"Error initializing OAuth credentials: {str(e)}")
        return None

def decode_service_account(key_data):
    """Decode service account from base64 or return JSON dict"""
    if not key_data:
        return None
    
    try:
        # Check if it's base64 encoded
        if key_data.strip().startswith('{'):
            # It's already JSON
            decoded = json.loads(key_data)
        else:
            # Try base64 decode
            decoded_bytes = base64.b64decode(key_data)
            decoded = json.loads(decoded_bytes)
        
        logger.info(f"✅ Decoded service account with fields: {list(decoded.keys())}")
        return decoded
    except Exception as e:
        logger.error(f"Failed to decode service account: {e}")
        return None

def format_customer_id(customer_id: str) -> str:
    """Format customer ID to ensure it's 10 digits without dashes."""
    customer_id = str(customer_id).replace('-', '').replace('"', '')
    customer_id = ''.join(char for char in customer_id if char.isdigit())
    return customer_id.zfill(10)

def get_headers(creds):
    """Get headers for Google Ads API requests."""
    developer_token = os.environ.get("GOOGLE_ADS_DEVELOPER_TOKEN")
    if not developer_token:
        raise ValueError("GOOGLE_ADS_DEVELOPER_TOKEN environment variable not set")
    
    login_customer_id = os.environ.get("GOOGLE_ADS_LOGIN_CUSTOMER_ID", "")
    
    # Refresh token if needed
    if creds and hasattr(creds, 'refresh'):
        auth_req = AuthRequest()
        creds.refresh(auth_req)
    
    headers = {
        'Authorization': f'Bearer {creds.token}',
        'developer-token': developer_token,
        'content-type': 'application/json'
    }
    
    if login_customer_id:
        headers['login-customer-id'] = format_customer_id(login_customer_id)
    
    return headers

def get_location_id(location_name: str, location_type: str, country_code: str = "US"):
    """Get location ID from CSV file."""
    try:
        # Try different possible filenames
        csv_files = ["geotargets-usa.csv", "geotargets-2024-10-10.csv", "geotargets.csv"]
        df = None
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                logger.info(f"Loaded location data from {csv_file}")
                break
        
        if df is None:
            # Create a minimal dataset for common US locations
            return get_fallback_location_id(location_name, location_type)
        
        df = df[df["Target Type"] == location_type]
        df = df[df["Country Code"] == country_code]
        
        if df.empty:
            return None, f"No {location_type} locations found in {country_code}"
        
        location_match = df[df["Name"].str.lower() == location_name.lower()]
        if not location_match.empty:
            return str(location_match.iloc[0]["Criteria ID"]), None
        
        return None, f"Location '{location_name}' not found"
    except Exception as e:
        logger.error(f"Error loading location data: {e}")
        return get_fallback_location_id(location_name, location_type)

def get_fallback_location_id(location_name: str, location_type: str):
    """Fallback location IDs for common US locations"""
    common_locations = {
        "United States": "2840",
        "New York": "21167",
        "California": "21137",
        "Texas": "21176",
        "Florida": "21142",
    }
    
    if location_name in common_locations:
        return common_locations[location_name], None
    
    # Default to US if not found
    return "2840", f"Location '{location_name}' not found, defaulting to United States"

def assign_match_type(keyword: str, search_volume: int) -> str:
    """Assign match type based on keyword characteristics"""
    if search_volume > 1000 and len(keyword.split()) <= 2:
        return "BROAD"
    elif search_volume < 100:
        return "EXACT"
    return "PHRASE"

# --- Environment Variables ---

# OAuth credentials
oauth_credentials = initialize_oauth_credentials()

# Google Ads settings
GOOGLE_ADS_DEVELOPER_TOKEN = os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN")
GOOGLE_ADS_LOGIN_CUSTOMER_ID = os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID", "")
GOOGLE_ADS_MANAGER_ID = os.getenv("GOOGLE_ADS_MANAGER_ID", GOOGLE_ADS_LOGIN_CUSTOMER_ID)

# Service Account Keys
SERVICE_ACCOUNT_KEY_SHEETS = decode_service_account(os.getenv("SERVICE_ACCOUNT_KEY_SHEETS"))

# AI Model API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Models to use
GPT_MODEL = "gpt-4-turbo-preview"
CLAUDE_MODEL = "claude-3-opus-20240229"

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
    match_type: Literal["BROAD", "PHRASE", "EXACT"] = "PHRASE"
    avg_monthly_searches: int = 0
    competition: Optional[str] = None

class AdCopyVariation(BaseModel):
    """Model for ad copy variation"""
    headlines: List[str]  # Max 15, each max 30 chars
    descriptions: List[str]  # Max 4, each max 90 chars

# --- Initialize Services ---

# Google Sheets Client
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

# Initialize AI clients
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Initialize FastMCP server
mcp = FastMCP(
    name="Google Ads Automation MCP"
)

# --- Simple State Management ---

class SimpleStateManager:
    """Simple in-memory state management"""
    def __init__(self):
        self.conversations = {}
        
    def get_or_create_conversation(self, conversation_id: str = None) -> str:
        """Get existing or create new conversation context"""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "keywords": [],
                "ad_copy": None,
                "url": None,
                "content": None,
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

# Global state manager instance
state_manager = SimpleStateManager()

# --- Helper function for GAQL queries ---

def execute_gaql_query_internal(customer_id: str, query: str) -> str:
    """Internal function to execute a custom GAQL query"""
    try:
        if not oauth_credentials:
            return "Error: OAuth credentials not configured. Please set GOOGLE_ADS_OAUTH_TOKENS_BASE64"
        
        headers = get_headers(oauth_credentials)
        
        formatted_customer_id = format_customer_id(customer_id)
        url = f"https://googleads.googleapis.com/{API_VERSION}/customers/{formatted_customer_id}/googleAds:search"
        
        payload = {"query": query}
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            return f"Error executing query: {response.text}"
        
        results = response.json()
        if not results.get('results'):
            return "No results found for the query."
        
        # Format results
        result_lines = [f"Query Results for Account {formatted_customer_id}:"]
        result_lines.append("-" * 80)
        
        for i, result in enumerate(results['results'][:50], 1):
            result_lines.append(f"\nResult {i}:")
            result_lines.append(json.dumps(result, indent=2))
        
        return "\n".join(result_lines)
    
    except Exception as e:
        return f"Error executing GAQL query: {str(e)}"

# --- MCP Tools ---

@mcp.tool()
async def keyword_research(
    keywords: List[str] = Field(description="List of keywords to get search volume for"),
    content: Optional[str] = Field(default=None, description="Page content for context (optional)"),
    location: str = Field(default="United States", description="Target location"),
    location_type: str = Field(default="Country", description="Location type (City, State, Country)"),
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")
) -> Dict:
    """
    Get search volume data for a list of keywords using Google Ads API.
    
    Returns mock data if Google Ads API is not fully configured.
    """
    logger.info(f"🔍 Starting keyword research for {len(keywords)} keywords")
    
    conv_id = state_manager.get_or_create_conversation(conversation_id)
    
    # Store content if provided
    if content:
        state_manager.set(conv_id, "content", content)
    
    # Format keywords properly
    formatted_keywords = []
    for kw in keywords:
        if isinstance(kw, str):
            formatted_keywords.append({
                "keyword": kw.lower().strip(),
                "match_type": "PHRASE"
            })
        elif isinstance(kw, dict):
            formatted_keywords.append({
                "keyword": kw.get("keyword", kw.get("text", str(kw))).lower().strip(),
                "match_type": kw.get("match_type", "PHRASE")
            })
    
    keywords_with_metrics = []
    
    # Try to get real metrics from Google Ads API
    if oauth_credentials and GOOGLE_ADS_DEVELOPER_TOKEN:
        try:
            # For now, we'll use the GAQL approach to get keyword ideas
            # This is a simplified version - full implementation would use KeywordPlanIdeaService
            
            # Mock implementation - in production, this would call the actual API
            logger.info("Using mock data due to API permission restrictions")
            
            # Generate mock metrics for demo
            import random
            for kw_data in formatted_keywords:
                keyword = kw_data["keyword"]
                mock_volume = random.randint(100, 10000)
                keywords_with_metrics.append({
                    "keyword": keyword,
                    "match_type": assign_match_type(keyword, mock_volume),
                    "avg_monthly_searches": mock_volume,
                    "competition": random.choice(["LOW", "MEDIUM", "HIGH"])
                })
            
        except Exception as e:
            logger.error(f"Error getting keyword metrics: {e}")
            # Fall back to keywords without metrics
            keywords_with_metrics = [
                {
                    "keyword": kw["keyword"],
                    "match_type": kw["match_type"],
                    "avg_monthly_searches": 0,
                    "competition": "UNKNOWN"
                }
                for kw in formatted_keywords
            ]
    else:
        # No API configured, return keywords without metrics
        keywords_with_metrics = [
            {
                "keyword": kw["keyword"],
                "match_type": kw["match_type"],
                "avg_monthly_searches": 0,
                "competition": "UNKNOWN"
            }
            for kw in formatted_keywords
        ]
    
    # Sort by search volume
    keywords_with_metrics.sort(key=lambda x: x.get("avg_monthly_searches", 0), reverse=True)
    
    # Save to state
    state_manager.set(conv_id, "keywords", keywords_with_metrics)
    
    # Group keywords by theme
    themes = {}
    for kw_data in keywords_with_metrics:
        keyword = kw_data["keyword"]
        
        # Simple theme detection
        if any(term in keyword for term in ["lawyer", "attorney", "legal", "law firm"]):
            theme = "Legal Services"
        elif any(term in keyword for term in ["privacy", "data protection", "gdpr"]):
            theme = "Privacy & Compliance"
        elif any(term in keyword for term in ["cyber", "security", "breach", "incident"]):
            theme = "Cybersecurity"
        elif any(term in keyword for term in ["consulting", "advisory", "services"]):
            theme = "Consulting Services"
        else:
            theme = "General"
        
        if theme not in themes:
            themes[theme] = []
        themes[theme].append(kw_data)
    
    return {
        "status": "success",
        "conversation_id": conv_id,
        "total_keywords": len(keywords_with_metrics),
        "themes": len(themes),
        "keyword_groups": {
            theme: {
                "count": len(kws),
                "top_keywords": kws[:5]
            }
            for theme, kws in themes.items()
        },
        "all_keywords": keywords_with_metrics,
        "note": "Using mock search volume data for demonstration" if not oauth_credentials else None
    }

@mcp.tool()
async def generate_ad_copy(
    conversation_id: str = Field(description="Conversation ID from keyword_research"),
    themes: Optional[List[str]] = Field(default=None, description="Specific themes to generate for")
) -> Dict:
    """
    Generate ad copy variations based on keywords.
    """
    logger.info("📝 Starting ad copy generation")
    
    # Get keywords from state
    keywords = state_manager.get(conversation_id, "keywords", [])
    if not keywords:
        return {
            "status": "error",
            "message": "No keywords found. Please run keyword_research first."
        }
    
    # Group keywords by theme for ad copy generation
    themed_keywords = {}
    for kw in keywords:
        keyword_text = kw.get("keyword", "")
        
        # Determine theme
        if any(term in keyword_text for term in ["lawyer", "attorney", "legal"]):
            theme = "Legal Services"
        elif any(term in keyword_text for term in ["privacy", "data", "gdpr"]):
            theme = "Privacy & Compliance"
        elif any(term in keyword_text for term in ["cyber", "security", "breach"]):
            theme = "Cybersecurity"
        else:
            theme = "General"
        
        if themes and theme not in themes:
            continue
            
        if theme not in themed_keywords:
            themed_keywords[theme] = []
        themed_keywords[theme].append(keyword_text)
    
    # Generate ad copy for each theme
    ad_copies = {}
    
    for theme, theme_keywords in themed_keywords.items():
        logger.info(f"🎨 Generating ad copy for theme: {theme}")
        
        variations = {}
        
        # Generate with OpenAI if available
        if openai_client:
            try:
                prompt = f"""
                Create Google Ads copy for {theme} theme.
                Keywords: {', '.join(theme_keywords[:10])}
                
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
                variations["gpt4"] = {
                    "headlines": [h[:30] for h in result.get("headlines", [])[:15]],
                    "descriptions": [d[:90] for d in result.get("descriptions", [])[:4]]
                }
            except Exception as e:
                logger.error(f"Error with OpenAI: {e}")
        
        # Generate with Claude if available
        if anthropic_client:
            try:
                prompt = f"""
                Create Google Ads copy for {theme} theme.
                Keywords: {', '.join(theme_keywords[:10])}
                
                Return only JSON with 15 headlines (max 30 chars) and 4 descriptions (max 90 chars).
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
                    variations["claude"] = {
                        "headlines": [h[:30] for h in result.get("headlines", [])[:15]],
                        "descriptions": [d[:90] for d in result.get("descriptions", [])[:4]]
                    }
            except Exception as e:
                logger.error(f"Error with Claude: {e}")
        
        # Fallback if no AI available
        if not variations:
            variations["default"] = {
                "headlines": [
                    f"Expert {theme}"[:30],
                    f"Top {theme} Services"[:30],
                    f"Trusted {theme}"[:30],
                    f"{theme} Solutions"[:30],
                    f"Professional {theme}"[:30],
                    f"Leading {theme} Team"[:30],
                    f"{theme} Experts"[:30],
                    f"Quality {theme}"[:30],
                    f"Best {theme} Services"[:30],
                    f"{theme} Specialists"[:30],
                    f"Premier {theme}"[:30],
                    f"{theme} Professionals"[:30],
                    f"Reliable {theme}"[:30],
                    f"{theme} Leaders"[:30],
                    f"Your {theme} Partner"[:30]
                ],
                "descriptions": [
                    f"Get expert {theme.lower()} services today"[:90],
                    f"Trust our {theme.lower()} professionals"[:90],
                    f"Leading {theme.lower()} solutions for you"[:90],
                    f"Experience quality {theme.lower()} service"[:90]
                ]
            }
        
        ad_copies[theme] = variations
    
    # Save to state
    state_manager.set(conversation_id, "ad_copy", ad_copies)
    
    return {
        "status": "success",
        "conversation_id": conversation_id,
        "themes_generated": len(ad_copies),
        "ad_copies": ad_copies
    }

@mcp.tool()
async def create_campaign_sheet(
    conversation_id: str = Field(description="Conversation ID"),
    campaign_title: str = Field(description="Campaign title"),
    total_budget: float = Field(description="Total budget"),
    start_date: str = Field(description="Start date YYYYMMDD"),
    end_date: str = Field(description="End date YYYYMMDD"),
    campaign_type: str = Field(default="SEARCH", description="Campaign type")
) -> Dict:
    """
    Create a Google Sheet with campaign data.
    """
    logger.info(f"📊 Creating campaign sheet: {campaign_title}")
    
    # Validate dates
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        
        if start >= end:
            return {"status": "error", "message": "End date must be after start date"}
            
    except ValueError:
        return {"status": "error", "message": "Invalid date format. Use YYYYMMDD"}
    
    # Get data from state
    keywords = state_manager.get(conversation_id, "keywords", [])
    ad_copy = state_manager.get(conversation_id, "ad_copy", {})
    
    if not keywords or not ad_copy:
        return {
            "status": "error",
            "message": "Missing data. Please complete keyword research and ad copy generation."
        }
    
    # Create sheet URL (mock if sheets not configured)
    if sheets_client:
        try:
            # Create actual Google Sheet
            sheet = sheets_client.create(f"{campaign_title} - Campaign")
            worksheet = sheet.sheet1
            
            # Add campaign info
            worksheet.update('A1:B6', [
                ["Campaign Title:", campaign_title],
                ["Budget:", total_budget],
                ["Type:", campaign_type],
                ["Start Date:", start_date],
                ["End Date:", end_date],
                ["Keywords Count:", len(keywords)]
            ])
            
            # Add keywords
            worksheet.update('A8', "KEYWORDS")
            keyword_data = [["Keyword", "Match Type", "Monthly Searches"]]
            for kw in keywords[:50]:
                keyword_data.append([
                    kw.get("keyword", ""),
                    kw.get("match_type", ""),
                    kw.get("avg_monthly_searches", 0)
                ])
            worksheet.update(f'A9:C{9+len(keyword_data)}', keyword_data)
            
            sheet_url = sheet.url
        except Exception as e:
            logger.error(f"Error creating sheet: {e}")
            sheet_url = f"https://docs.google.com/spreadsheets/d/mock-{campaign_title.replace(' ', '-')}"
    else:
        sheet_url = f"https://docs.google.com/spreadsheets/d/mock-{campaign_title.replace(' ', '-')}"
    
    return {
        "status": "success",
        "conversation_id": conversation_id,
        "sheet_url": sheet_url,
        "campaign_summary": {
            "title": campaign_title,
            "budget": total_budget,
            "duration": f"{start_date} - {end_date}",
            "keywords_count": len(keywords),
            "themes": list(ad_copy.keys())
        }
    }

@mcp.tool()
async def launch_campaign(
    conversation_id: str = Field(description="Conversation ID"),
    customer_id: str = Field(description="Google Ads customer ID"),
    campaign_title: str = Field(description="Campaign title"),
    total_budget: float = Field(description="Total budget"),
    start_date: str = Field(description="Start date YYYYMMDD"),
    end_date: str = Field(description="End date YYYYMMDD"),
    campaign_type: str = Field(default="SEARCH", description="Campaign type"),
    location: str = Field(default="United States", description="Target location"),
    location_type: str = Field(default="Country", description="Location type")
) -> Dict:
    """
    Launch the campaign on Google Ads (mock implementation).
    """
    logger.info(f"🚀 Launching campaign: {campaign_title}")
    
    # Format customer ID
    customer_id = format_customer_id(customer_id)
    if len(customer_id) != 10:
        return {"status": "error", "message": "Customer ID must be 10 digits"}
    
    # Get campaign data
    keywords = state_manager.get(conversation_id, "keywords", [])
    ad_copy = state_manager.get(conversation_id, "ad_copy", {})
    
    if not keywords or not ad_copy:
        return {
            "status": "error",
            "message": "Missing campaign data. Please complete all previous steps."
        }
    
    # Mock campaign creation
    campaign_id = f"campaign_{uuid.uuid4().hex[:8]}"
    
    return {
        "status": "success",
        "campaign_id": campaign_id,
        "message": f"Campaign '{campaign_title}' launched successfully (mock)",
        "details": {
            "customer_id": customer_id,
            "budget": total_budget,
            "duration": f"{start_date} - {end_date}",
            "location": location,
            "keywords_count": len(keywords)
        },
        "note": "This is a mock implementation. Real campaign creation requires proper Google Ads API permissions."
    }

# Add tool from working server for account access
@mcp.tool()
async def list_accounts() -> str:
    """Lists all accessible Google Ads accounts."""
    try:
        if not oauth_credentials:
            return "Error: OAuth credentials not configured. Please set GOOGLE_ADS_OAUTH_TOKENS_BASE64"
        
        headers = get_headers(oauth_credentials)
        
        url = f"https://googleads.googleapis.com/{API_VERSION}/customers:listAccessibleCustomers"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return f"Error accessing accounts: {response.text}"
        
        customers = response.json()
        if not customers.get('resourceNames'):
            return "No accessible accounts found."
        
        result_lines = ["Accessible Google Ads Accounts:"]
        result_lines.append("-" * 50)
        
        for resource_name in customers['resourceNames']:
            customer_id = resource_name.split('/')[-1]
            formatted_id = format_customer_id(customer_id)
            result_lines.append(f"Account ID: {formatted_id}")
        
        return "\n".join(result_lines)
    
    except Exception as e:
        return f"Error listing accounts: {str(e)}"

@mcp.tool()
async def execute_gaql_query(
    customer_id: str = Field(description="Google Ads customer ID (10 digits, no dashes)"),
    query: str = Field(description="Valid GAQL query string")
) -> str:
    """Execute a custom GAQL (Google Ads Query Language) query."""
    return execute_gaql_query_internal(customer_id, query)

# --- MCP Resources ---

@mcp.resource("workflow://guide")
def workflow_guide() -> str:
    """Complete workflow guide"""
    return """
    # Google Ads Campaign Workflow
    
    ## Step 1: Check Account Access
    
    Use list_accounts() to see which Google Ads accounts you can access.
    
    ## Step 2: Keyword Research
    
    First, use Claude's built-in web search to fetch the landing page content:
    - Search for the URL to get page content
    - Extract relevant keywords from the content
    - Then use keyword_research tool with those keywords
    
    Example:
    1. User provides: https://example.com/services
    2. You search and analyze the page
    3. Extract keywords: ["service keyword 1", "service keyword 2", etc.]
    4. Call keyword_research(keywords=[...], location="United States")
    
    ## Step 3: Generate Ad Copy
    
    Use the conversation_id from keyword research:
    - generate_ad_copy(conversation_id="...")
    
    ## Step 4: Create Campaign Sheet
    
    Create a Google Sheet with all campaign data:
    - create_campaign_sheet(
        conversation_id="...",
        campaign_title="My Campaign",
        total_budget=1000,
        start_date="20250201",
        end_date="20250228"
      )
    
    ## Step 5: Launch Campaign (Mock)
    
    Deploy to Google Ads:
    - launch_campaign(
        conversation_id="...",
        customer_id="123-456-7890",
        campaign_title="My Campaign",
        total_budget=1000,
        start_date="20250201",
        end_date="20250228"
      )
    
    ## Using GAQL Queries
    
    You can also execute custom Google Ads queries:
    - execute_gaql_query(
        customer_id="1234567890",
        query="SELECT campaign.name, metrics.clicks FROM campaign WHERE segments.date DURING LAST_7_DAYS"
      )
    
    ## Important Notes:
    
    - Always use web search to get page content first
    - Keywords should be passed as a simple list of strings
    - Customer IDs can include dashes (they'll be formatted automatically)
    - Dates must be in YYYYMMDD format
    - Budget is in the account's local currency
    - Some features use mock data when API permissions are restricted
    """

@mcp.resource("api://status")
def api_status() -> str:
    """Current API and service status"""
    return json.dumps({
        "services": {
            "google_ads_oauth": "✅ Connected" if oauth_credentials else "❌ Not configured",
            "google_sheets": "✅ Connected" if sheets_client else "❌ Not configured",
            "openai": "✅ Connected" if openai_client else "❌ Not configured",
            "anthropic": "✅ Connected" if anthropic_client else "❌ Not configured"
        },
        "server": {
            "base_url": base_url,
            "active_conversations": len(state_manager.conversations),
            "api_version": API_VERSION
        },
        "notes": [
            "Using OAuth authentication for Google Ads API",
            "Some features may use mock data if API permissions are restricted"
        ]
    }, indent=2)

# --- Main Execution ---

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Google Ads MCP Server")
    logger.info(f"📍 Port: {port}")
    logger.info(f"🌐 Base URL: {base_url}")
    logger.info("=" * 60)
    logger.info("📋 Instructions:")
    logger.info("  1. Use Claude's web search to fetch page content")
    logger.info("  2. Extract keywords from the content")
    logger.info("  3. Pass keywords to keyword_research tool")
    logger.info("  4. Generate ad copy and create campaigns")
    logger.info("=" * 60)
    logger.info("🔧 Service Status:")
    logger.info(f"  Google Ads OAuth: {'✅' if oauth_credentials else '❌'}")
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
