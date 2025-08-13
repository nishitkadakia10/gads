#!/usr/bin/env python3
"""
Google Ads Automation MCP Server - No Fallbacks Version
Real API calls only - fails explicitly if API not available
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
    level=logging.DEBUG,  # Set to DEBUG for comprehensive logging
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
        logger.error("❌ GOOGLE_ADS_OAUTH_TOKENS_BASE64 not set - OAuth features disabled")
        return None
    
    try:
        logger.debug("Decoding OAuth tokens from base64...")
        oauth_tokens_json = base64.b64decode(oauth_tokens_base64).decode('utf-8')
        oauth_tokens = json.loads(oauth_tokens_json)
        
        logger.debug(f"OAuth token fields: {list(oauth_tokens.keys())}")
        
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
                logger.info("✅ Token refreshed successfully")
        
        logger.info("✅ OAuth credentials initialized successfully")
        return credentials
        
    except Exception as e:
        logger.error(f"❌ Error initializing OAuth credentials: {str(e)}")
        return None

def decode_service_account(key_data):
    """Decode service account from base64 or return JSON dict"""
    if not key_data:
        return None
    
    try:
        if key_data.strip().startswith('{'):
            decoded = json.loads(key_data)
        else:
            decoded_bytes = base64.b64decode(key_data)
            decoded = json.loads(decoded_bytes)
        
        logger.info(f"✅ Decoded service account with fields: {list(decoded.keys())}")
        return decoded
    except Exception as e:
        logger.error(f"❌ Failed to decode service account: {e}")
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
    
    logger.debug(f"Request headers prepared (token: {'present' if creds.token else 'missing'})")
    return headers

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

# Models to use - UPDATE TO NON-DEPRECATED MODEL
GPT_MODEL = "gpt-4-turbo-preview"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # Updated to non-deprecated model

# Get server URL from environment
public_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
if public_domain:
    base_url = f"https://{public_domain}"
else:
    base_url = f"http://localhost:{os.environ.get('PORT', '8080')}"

logger.info("=" * 60)
logger.info("🚀 Google Ads Automation MCP Server Starting (NO FALLBACKS)")
logger.info(f"📍 Base URL: {base_url}")
logger.info("=" * 60)

# --- Data Models ---

class KeywordData(BaseModel):
    """Model for keyword with match type and metrics"""
    keyword: str
    match_type: Literal["BROAD", "PHRASE", "EXACT"] = "PHRASE"
    avg_monthly_searches: int = 0
    competition: Optional[str] = None

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

if openai_client:
    logger.info("✅ OpenAI client initialized")
else:
    logger.warning("⚠️ OpenAI client not initialized - OPENAI_API_KEY missing")

if anthropic_client:
    logger.info("✅ Anthropic client initialized")
else:
    logger.warning("⚠️ Anthropic client not initialized - ANTHROPIC_API_KEY missing")

# Initialize FastMCP server
mcp = FastMCP(
    name="Google Ads Automation MCP (No Fallbacks)"
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
            return "❌ Error: OAuth credentials not configured. Please set GOOGLE_ADS_OAUTH_TOKENS_BASE64"
        
        headers = get_headers(oauth_credentials)
        
        formatted_customer_id = format_customer_id(customer_id)
        url = f"https://googleads.googleapis.com/{API_VERSION}/customers/{formatted_customer_id}/googleAds:search"
        
        logger.debug(f"Executing GAQL query for customer {formatted_customer_id}")
        logger.debug(f"Query: {query}")
        
        payload = {"query": query}
        response = requests.post(url, headers=headers, json=payload)
        
        logger.debug(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"GAQL query failed: {response.text}")
            return f"❌ Error executing query: {response.text}"
        
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
        logger.error(f"Exception in GAQL query: {str(e)}")
        return f"❌ Error executing GAQL query: {str(e)}"

def get_keyword_metrics_from_api(customer_id: str, keywords: List[str], location_id: Optional[str] = None) -> List[Dict]:
    """
    Get real keyword metrics from Google Ads API using GAQL
    NO FALLBACKS - returns error if API fails
    """
    if not oauth_credentials or not GOOGLE_ADS_DEVELOPER_TOKEN:
        raise ValueError("Google Ads API credentials not configured")
    
    logger.info(f"📊 Fetching real keyword metrics for {len(keywords)} keywords...")
    
    try:
        headers = get_headers(oauth_credentials)
        formatted_customer_id = format_customer_id(customer_id)
        
        # Use KeywordPlanIdeaService endpoint
        url = f"https://googleads.googleapis.com/{API_VERSION}/customers/{formatted_customer_id}:generateKeywordIdeas"
        
        # Build the request payload
        payload = {
            "customerId": formatted_customer_id,
            "keywordPlanNetwork": "GOOGLE_SEARCH",
            "keywordSeed": {
                "keywords": keywords[:100]  # Limit to 100 keywords
            }
        }
        
        if location_id:
            payload["geoTargetConstants"] = [f"geoTargetConstants/{location_id}"]
        
        logger.debug(f"Requesting keyword ideas from API...")
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            error_msg = f"API request failed: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        results = response.json()
        keyword_metrics = []
        
        for result in results.get('results', []):
            keyword_text = result.get('text', '')
            metrics = result.get('keywordIdeaMetrics', {})
            
            keyword_metrics.append({
                "keyword": keyword_text,
                "avg_monthly_searches": metrics.get('avgMonthlySearches', 0),
                "competition": metrics.get('competition', 'UNKNOWN'),
                "competition_index": metrics.get('competitionIndex', 0)
            })
        
        logger.info(f"✅ Retrieved metrics for {len(keyword_metrics)} keywords")
        return keyword_metrics
        
    except Exception as e:
        logger.error(f"Failed to get keyword metrics: {str(e)}")
        raise

# --- MCP Tools ---

@mcp.tool()
async def keyword_research(
    keywords: List[str] = Field(description="List of keywords to get search volume for"),
    customer_id: Optional[str] = Field(default=None, description="Google Ads customer ID for API access"),
    content: Optional[str] = Field(default=None, description="Page content for context (optional)"),
    location: str = Field(default="United States", description="Target location"),
    location_type: str = Field(default="Country", description="Location type (City, State, Country)"),
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")
) -> Dict:
    """
    Get REAL search volume data for keywords from Google Ads API.
    NO MOCK DATA - fails if API is not available.
    
    Requires:
    - customer_id: A valid Google Ads account ID
    - Proper OAuth credentials configured
    """
    logger.info(f"🔍 Starting REAL keyword research for {len(keywords)} keywords")
    
    # Validate we have everything needed for real API calls
    if not oauth_credentials:
        return {
            "status": "error",
            "message": "❌ Google Ads OAuth credentials not configured. Set GOOGLE_ADS_OAUTH_TOKENS_BASE64"
        }
    
    if not customer_id:
        return {
            "status": "error",
            "message": "❌ customer_id is required for real keyword metrics. Please provide a valid Google Ads account ID."
        }
    
    conv_id = state_manager.get_or_create_conversation(conversation_id)
    
    # Store content if provided
    if content:
        state_manager.set(conv_id, "content", content)
    
    # Format keywords properly
    formatted_keywords = []
    for kw in keywords:
        if isinstance(kw, str):
            formatted_keywords.append(kw.lower().strip())
    
    logger.debug(f"Processing keywords: {formatted_keywords[:5]}...")  # Log first 5 keywords
    
    # Get REAL metrics from Google Ads API
    try:
        # Try using the Keyword Planner approach
        keyword_metrics = get_keyword_metrics_from_api(
            customer_id=customer_id,
            keywords=formatted_keywords,
            location_id="2840"  # US location ID
        )
        
        # Process and enhance the results
        keywords_with_metrics = []
        for metric in keyword_metrics:
            avg_searches = metric.get('avg_monthly_searches', 0)
            
            # Determine match type based on search volume
            if avg_searches > 1000:
                match_type = "BROAD"
            elif avg_searches < 100:
                match_type = "EXACT"
            else:
                match_type = "PHRASE"
            
            keywords_with_metrics.append({
                "keyword": metric['keyword'],
                "match_type": match_type,
                "avg_monthly_searches": avg_searches,
                "competition": metric.get('competition', 'UNKNOWN')
            })
        
        # Sort by search volume
        keywords_with_metrics.sort(key=lambda x: x.get("avg_monthly_searches", 0), reverse=True)
        
        logger.info(f"✅ Successfully retrieved REAL metrics for {len(keywords_with_metrics)} keywords")
        
    except Exception as e:
        logger.error(f"❌ Failed to get real keyword metrics: {str(e)}")
        return {
            "status": "error",
            "message": f"❌ Failed to get keyword metrics from Google Ads API: {str(e)}"
        }
    
    # Save to state
    state_manager.set(conv_id, "keywords", keywords_with_metrics)
    
    # Group keywords by theme
    themes = {}
    for kw_data in keywords_with_metrics:
        keyword = kw_data["keyword"]
        
        # Theme detection
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
        "data_source": "Google Ads API (REAL DATA)",
        "keyword_groups": {
            theme: {
                "count": len(kws),
                "top_keywords": kws[:5]
            }
            for theme, kws in themes.items()
        },
        "all_keywords": keywords_with_metrics
    }

@mcp.tool()
async def generate_ad_copy(
    conversation_id: str = Field(description="Conversation ID from keyword_research"),
    themes: Optional[List[str]] = Field(default=None, description="Specific themes to generate for")
) -> Dict:
    """
    Generate ad copy variations using AI models.
    NO FALLBACKS - requires at least one AI service configured.
    """
    logger.info("📝 Starting ad copy generation (NO FALLBACKS)")
    
    # Check if we have AI services
    if not openai_client and not anthropic_client:
        return {
            "status": "error",
            "message": "❌ No AI services configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
        }
    
    # Get keywords from state
    keywords = state_manager.get(conversation_id, "keywords", [])
    if not keywords:
        return {
            "status": "error",
            "message": "❌ No keywords found. Please run keyword_research first."
        }
    
    # Group keywords by theme
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
        
        # Try OpenAI first
        if openai_client:
            try:
                logger.debug(f"Calling OpenAI GPT-4 for {theme}...")
                prompt = f"""
                Create Google Ads copy for {theme} theme.
                Keywords: {', '.join(theme_keywords[:10])}
                
                Requirements:
                - 15 headlines (max 30 chars each)
                - 4 descriptions (max 90 chars each)
                
                Return ONLY valid JSON with this exact structure:
                {{"headlines": ["headline1", "headline2", ...], "descriptions": ["desc1", "desc2", ...]}}
                """
                
                response = openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert Google Ads copywriter. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                variations["gpt4"] = {
                    "headlines": [h[:30] for h in result.get("headlines", [])[:15]],
                    "descriptions": [d[:90] for d in result.get("descriptions", [])[:4]]
                }
                logger.info(f"✅ GPT-4 generated ad copy for {theme}")
            except Exception as e:
                logger.error(f"❌ OpenAI error for {theme}: {str(e)}")
        
        # Try Claude if available
        if anthropic_client:
            try:
                logger.debug(f"Calling Claude for {theme}...")
                prompt = f"""
                Create Google Ads copy for {theme} theme.
                Keywords: {', '.join(theme_keywords[:10])}
                
                Return ONLY valid JSON with 15 headlines (max 30 chars) and 4 descriptions (max 90 chars).
                Format: {{"headlines": [...], "descriptions": [...]}}
                """
                
                response = anthropic_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                text = response.content[0].text
                logger.debug(f"Claude response preview: {text[:200]}...")
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    variations["claude"] = {
                        "headlines": [h[:30] for h in result.get("headlines", [])[:15]],
                        "descriptions": [d[:90] for d in result.get("descriptions", [])[:4]]
                    }
                    logger.info(f"✅ Claude generated ad copy for {theme}")
                else:
                    logger.error(f"❌ Could not extract JSON from Claude response")
            except Exception as e:
                logger.error(f"❌ Claude error for {theme}: {str(e)}")
        
        # NO FALLBACK - if no variations generated, report error
        if not variations:
            return {
                "status": "error",
                "message": f"❌ Failed to generate ad copy for {theme}. Check AI service configuration and logs."
            }
        
        ad_copies[theme] = variations
    
    # Save to state
    state_manager.set(conversation_id, "ad_copy", ad_copies)
    
    return {
        "status": "success",
        "conversation_id": conversation_id,
        "themes_generated": len(ad_copies),
        "ad_copies": ad_copies,
        "ai_models_used": list(set([model for theme in ad_copies.values() for model in theme.keys()]))
    }

@mcp.tool()
async def list_accounts() -> str:
    """Lists all accessible Google Ads accounts."""
    try:
        if not oauth_credentials:
            return "❌ Error: OAuth credentials not configured. Please set GOOGLE_ADS_OAUTH_TOKENS_BASE64"
        
        headers = get_headers(oauth_credentials)
        
        url = f"https://googleads.googleapis.com/{API_VERSION}/customers:listAccessibleCustomers"
        logger.debug(f"Requesting accessible customers...")
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Failed to list accounts: {response.text}")
            return f"❌ Error accessing accounts: {response.text}"
        
        customers = response.json()
        if not customers.get('resourceNames'):
            return "No accessible accounts found."
        
        result_lines = ["✅ Accessible Google Ads Accounts:"]
        result_lines.append("-" * 50)
        
        for resource_name in customers['resourceNames']:
            customer_id = resource_name.split('/')[-1]
            formatted_id = format_customer_id(customer_id)
            result_lines.append(f"Account ID: {formatted_id}")
        
        logger.info(f"Found {len(customers['resourceNames'])} accessible accounts")
        return "\n".join(result_lines)
    
    except Exception as e:
        logger.error(f"Exception listing accounts: {str(e)}")
        return f"❌ Error listing accounts: {str(e)}"

@mcp.tool()
async def execute_gaql_query(
    customer_id: str = Field(description="Google Ads customer ID (10 digits, no dashes)"),
    query: str = Field(description="Valid GAQL query string")
) -> str:
    """Execute a custom GAQL (Google Ads Query Language) query."""
    return execute_gaql_query_internal(customer_id, query)

# --- MCP Resources ---

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
            "api_version": API_VERSION,
            "mode": "NO_FALLBACKS - Real API calls only"
        },
        "configuration": {
            "oauth_configured": bool(oauth_credentials),
            "developer_token": bool(GOOGLE_ADS_DEVELOPER_TOKEN),
            "login_customer_id": GOOGLE_ADS_LOGIN_CUSTOMER_ID or "Not set",
            "openai_configured": bool(OPENAI_API_KEY),
            "anthropic_configured": bool(ANTHROPIC_API_KEY),
            "sheets_configured": bool(SERVICE_ACCOUNT_KEY_SHEETS)
        }
    }, indent=2)

# --- Main Execution ---

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Google Ads MCP Server (NO FALLBACKS MODE)")
    logger.info(f"📍 Port: {port}")
    logger.info(f"🌐 Base URL: {base_url}")
    logger.info("=" * 60)
    logger.info("⚠️  NO MOCK DATA - Real API calls only!")
    logger.info("⚠️  NO FALLBACKS - Fails if services unavailable!")
    logger.info("=" * 60)
    logger.info("🔧 Service Status:")
    logger.info(f"  Google Ads OAuth: {'✅' if oauth_credentials else '❌ NOT CONFIGURED'}")
    logger.info(f"  Developer Token: {'✅' if GOOGLE_ADS_DEVELOPER_TOKEN else '❌ NOT SET'}")
    logger.info(f"  Login Customer ID: {GOOGLE_ADS_LOGIN_CUSTOMER_ID or '❌ NOT SET'}")
    logger.info(f"  Google Sheets: {'✅' if sheets_client else '❌'}")
    logger.info(f"  OpenAI: {'✅' if openai_client else '❌'}")
    logger.info(f"  Anthropic: {'✅' if anthropic_client else '❌'}")
    logger.info("=" * 60)
    
    if not oauth_credentials:
        logger.error("⚠️  WARNING: Google Ads OAuth not configured!")
        logger.error("⚠️  Set GOOGLE_ADS_OAUTH_TOKENS_BASE64 for real API access")
    
    if not openai_client and not anthropic_client:
        logger.error("⚠️  WARNING: No AI services configured!")
        logger.error("⚠️  Set OPENAI_API_KEY or ANTHROPIC_API_KEY for ad copy generation")
    
    logger.info("=" * 60)
    
    # Run FastMCP server
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port
    )
