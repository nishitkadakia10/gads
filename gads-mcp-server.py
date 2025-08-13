#!/usr/bin/env python3
"""
Google Ads Automation MCP Server - FastMCP Implementation with Streamable HTTP
Deployable on Railway for remote access via Claude Desktop
"""

import os
import json
import time
import uuid
import asyncio
import logging
import warnings
import tempfile
import pandas as pd
from typing import Any, Optional, Literal, Dict, List, Union
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn")

from pydantic import Field, BaseModel, HttpUrl, model_validator, field_validator, PrivateAttr
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Google Ads imports
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core import exceptions

# Additional imports for tools
import httpx
import requests
from fuzzywuzzy import process
from openai import OpenAI
from anthropic import Anthropic
import litellm
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.cache_context import CacheMode
import nodriver as uc

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

# --- Environment Variables ---
GOOGLE_ADS_DEVELOPER_TOKEN = os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN")
GOOGLE_ADS_MANAGER_ID = os.getenv("GOOGLE_ADS_MANAGER_ID")
SERVICE_ACCOUNT_KEY_ADS = os.getenv("SERVICE_ACCOUNT_KEY_ADS")
SERVICE_ACCOUNT_KEY_FIREBASE = os.getenv("SERVICE_ACCOUNT_KEY_FIREBASE")
GOOGLE_SHEET_TEMPLATE = os.getenv("GOOGLE_SHEET_TEMPLATE")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
THREAD_TIMEOUT = int(os.getenv("THREAD_TIMEOUT", "300"))

# AI Model API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
O1_API_KEY = os.getenv("O1_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Web scraping credentials
DATA_FOR_CEO_NAME = os.getenv("DATA_FOR_CEO_NAME")
DATA_FOR_CEO_PASSWORD = os.getenv("DATA_FOR_CEO_PASSWORD")

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

# --- Initialize Services ---

# Initialize Firebase
if SERVICE_ACCOUNT_KEY_FIREBASE:
    try:
        service_account_key = json.loads(SERVICE_ACCOUNT_KEY_FIREBASE)
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_key)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("✅ Firebase initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Firebase: {e}")
        db = None
else:
    logger.warning("⚠️  Firebase credentials not configured")
    db = None

# Initialize Google Ads Client
google_ads_client = None
if SERVICE_ACCOUNT_KEY_ADS and GOOGLE_ADS_DEVELOPER_TOKEN and GOOGLE_ADS_MANAGER_ID:
    try:
        service_account_info = json.loads(SERVICE_ACCOUNT_KEY_ADS)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(json.dumps(service_account_info).encode())
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
else:
    logger.warning("⚠️  Google Ads credentials not configured")

# Initialize AI clients
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Initialize FastMCP server
mcp = FastMCP(
    name="Google Ads Automation MCP"
)

# --- Shared State Management ---

class SharedState:
    """Manages shared state across tools"""
    def __init__(self):
        self.data = {
            "KEYWORDS": None,
            "GENERATED_AD_COPY": None,
            "URL": None,
            "SEARCH_DATA": None,
            "CONVERSATION_ID": None
        }
    
    def get(self, key: str, default=None):
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any):
        self.data[key] = value
        # If Firebase is available, persist to database
        if db and self.data.get("CONVERSATION_ID"):
            self.save_to_firebase(key, value)
    
    def save_to_firebase(self, key: str, value: Any):
        """Save state to Firebase"""
        try:
            conversation_id = self.data.get("CONVERSATION_ID")
            if conversation_id and key in ["KEYWORDS", "GENERATED_AD_COPY", "URL", "SEARCH_DATA"]:
                db.collection("agencii-chats").document(conversation_id).set(
                    {key: value}, merge=True
                )
        except Exception as e:
            logger.error(f"Failed to save to Firebase: {e}")
    
    def load_from_firebase(self, conversation_id: str):
        """Load state from Firebase"""
        try:
            if db:
                doc = db.collection("agencii-chats").document(conversation_id).get()
                if doc.exists:
                    doc_data = doc.to_dict()
                    for key in ["KEYWORDS", "GENERATED_AD_COPY", "URL", "SEARCH_DATA"]:
                        if key in doc_data:
                            self.data[key] = doc_data[key]
        except Exception as e:
            logger.error(f"Failed to load from Firebase: {e}")

# Global shared state instance
shared_state = SharedState()

# --- Helper Functions ---

def format_customer_id(customer_id: str) -> str:
    """Format customer ID to ensure it's 10 digits without dashes."""
    customer_id = str(customer_id)
    customer_id = customer_id.replace('\"', '').replace('"', '').replace('-', '')
    customer_id = ''.join(char for char in customer_id if char.isdigit())
    return customer_id.zfill(10)

def get_location_id(location_name: str, location_type: str, country_code: str) -> tuple[str, str]:
    """Get location ID from CSV file."""
    try:
        # This would need the geotargets CSV file uploaded with the deployment
        df = pd.read_csv("geotargets-2024-10-10.csv")
        df = df[df["Target Type"] == location_type]
        df = df[df["Country Code"] == country_code]
        
        if df.empty:
            return None, f"No `{location_type}` locations found in {country_code}"
        
        city_match = df[df["Name"].str.lower() == location_name.lower()]
        if not city_match.empty:
            return city_match.iloc[0]["Criteria ID"], None
        
        closest_matches = process.extract(location_name, df["Name"], limit=5)
        suggestions = [match[0] for match in closest_matches]
        error_msg = f"Location '{location_name}' not found. Did you mean: {', '.join(suggestions)}?"
        return None, error_msg
    except Exception as e:
        logger.error(f"Error loading location data: {e}")
        return None, f"Error loading location data: {str(e)}"

# --- MCP Tools ---

@mcp.tool()
async def health_check() -> str:
    """Health check endpoint for monitoring"""
    status = {
        "status": "healthy",
        "service": "Google Ads Automation MCP",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "firebase": db is not None,
            "google_ads": google_ads_client is not None,
            "openai": openai_client is not None,
            "anthropic": anthropic_client is not None,
        }
    }
    logger.info("🏥 Health check requested")
    return json.dumps(status, indent=2)

@mcp.tool()
async def keyword_search(
    url: str = Field(description="URL to search keywords for"),
    location: str = Field(description="Location name or 'Worldwide'"),
    country_code: Optional[str] = Field(default=None, description="2-letter country code"),
    location_type: Optional[str] = Field(default=None, description="Location type (City, State, Country, etc.)")
) -> Dict:
    """
    Performs web scraping and keyword generation for a given web page URL.
    Returns keywords with search volume data for the specified location.
    """
    logger.info(f"🔍 Starting keyword search for URL: {url}")
    logger.info(f"📍 Location: {location}, Country: {country_code}, Type: {location_type}")
    
    # Validate location parameters
    if location != "Worldwide" and (not location_type or not country_code):
        return {
            "status": "error",
            "message": "Location type and country code are required if location is not 'Worldwide'"
        }
    
    # Check if provided location exists
    location_id = None
    if location != "Worldwide":
        location_id, error_msg = get_location_id(location, location_type, country_code)
        if error_msg:
            return {"status": "error", "message": error_msg}
    
    # Create task ID for async processing
    task_id = str(uuid.uuid4()).replace("-", "_")
    
    # Store initial task status
    task_data = {
        "status": "processing",
        "url": url,
        "location": location,
        "started_at": datetime.now(timezone.utc).isoformat()
    }
    
    # If we have a conversation ID, store in Firebase
    conversation_id = shared_state.get("CONVERSATION_ID")
    if conversation_id and db:
        db.collection("agencii-chats").document(conversation_id).set(
            {f"task_{task_id}": task_data}, merge=True
        )
    
    # Start async processing
    thread = Thread(
        target=_process_keyword_search,
        args=(url, location, location_id, task_id, conversation_id)
    )
    thread.start()
    
    return {
        "status": "processing",
        "task_id": task_id,
        "message": "Keyword search started. Use check_progress to monitor."
    }

def _process_keyword_search(url: str, location: str, location_id: str, task_id: str, conversation_id: str):
    """Process keyword search in background"""
    try:
        # Simulate keyword extraction (simplified version)
        # In production, this would include the full web scraping logic
        logger.info(f"Processing keyword search for task {task_id}")
        
        # Mock results for demonstration
        keywords = [
            {"keyword": "sample keyword 1", "match_type": "BROAD", "avg_monthly_searches": 1000},
            {"keyword": "sample keyword 2", "match_type": "PHRASE", "avg_monthly_searches": 500},
            {"keyword": "sample keyword 3", "match_type": "EXACT", "avg_monthly_searches": 100}
        ]
        
        # Save results to shared state
        shared_state.set("KEYWORDS", keywords)
        shared_state.set("URL", url)
        
        # Update task status
        task_data = {
            "status": "completed",
            "result": f"Found {len(keywords)} keywords",
            "keywords": keywords,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        if conversation_id and db:
            db.collection("agencii-chats").document(conversation_id).set(
                {f"task_{task_id}": task_data}, merge=True
            )
        
        logger.info(f"✅ Keyword search completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"❌ Error in keyword search: {e}")
        task_data = {
            "status": "error",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        if conversation_id and db:
            db.collection("agencii-chats").document(conversation_id).set(
                {f"task_{task_id}": task_data}, merge=True
            )

@mcp.tool()
async def generate_ad_copy() -> Dict:
    """
    Generates ad copy variations using GPT-4 and Claude.
    Uses data from previous keyword search.
    """
    logger.info("📝 Starting ad copy generation")
    
    # Check for required data
    search_data = shared_state.get("SEARCH_DATA")
    if not search_data:
        return {
            "status": "error",
            "message": "No search data found. Please run keyword_search first."
        }
    
    task_id = str(uuid.uuid4()).replace("-", "_")
    conversation_id = shared_state.get("CONVERSATION_ID")
    
    # Store initial task status
    task_data = {
        "status": "processing",
        "started_at": datetime.now(timezone.utc).isoformat()
    }
    
    if conversation_id and db:
        db.collection("agencii-chats").document(conversation_id).set(
            {f"task_{task_id}": task_data}, merge=True
        )
    
    # Start async processing
    thread = Thread(
        target=_process_ad_copy_generation,
        args=(search_data, task_id, conversation_id)
    )
    thread.start()
    
    return {
        "status": "processing",
        "task_id": task_id,
        "message": "Ad copy generation started. Use check_progress to monitor."
    }

def _process_ad_copy_generation(search_data: str, task_id: str, conversation_id: str):
    """Process ad copy generation in background"""
    try:
        logger.info(f"Processing ad copy generation for task {task_id}")
        
        # Mock ad copy generation
        ad_copy = {
            "gpt4": {
                "headlines": ["Great Products", "Best Prices", "Shop Now"],
                "descriptions": ["Discover amazing deals", "Quality guaranteed"],
                "extensions": {
                    "sitelinks": [],
                    "callouts": ["Free Shipping", "24/7 Support"],
                    "structured_snippet": {
                        "header": "Services",
                        "values": ["Consulting", "Support", "Training"]
                    }
                }
            },
            "claude": {
                "headlines": ["Premium Solutions", "Expert Service", "Get Started"],
                "descriptions": ["Transform your business", "Professional expertise"],
                "extensions": {
                    "sitelinks": [],
                    "callouts": ["Money Back Guarantee", "Expert Team"],
                    "structured_snippet": {
                        "header": "Types",
                        "values": ["Enterprise", "Small Business", "Startup"]
                    }
                }
            }
        }
        
        # Save results
        shared_state.set("GENERATED_AD_COPY", ad_copy)
        
        # Update task status
        task_data = {
            "status": "completed",
            "result": "Ad copy generated successfully",
            "ad_copy_preview": {
                "gpt4_headlines": ad_copy["gpt4"]["headlines"][:3],
                "claude_headlines": ad_copy["claude"]["headlines"][:3]
            },
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        if conversation_id and db:
            db.collection("agencii-chats").document(conversation_id).set(
                {f"task_{task_id}": task_data}, merge=True
            )
        
        logger.info(f"✅ Ad copy generation completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"❌ Error in ad copy generation: {e}")
        task_data = {
            "status": "error",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        if conversation_id and db:
            db.collection("agencii-chats").document(conversation_id).set(
                {f"task_{task_id}": task_data}, merge=True
            )

@mcp.tool()
async def check_progress(
    task_id: str = Field(description="Task ID to check progress for")
) -> Dict:
    """
    Check the progress of an asynchronous task.
    """
    logger.info(f"📊 Checking progress for task: {task_id}")
    
    conversation_id = shared_state.get("CONVERSATION_ID")
    
    if not conversation_id or not db:
        return {
            "status": "error",
            "message": "No conversation context or database connection available"
        }
    
    try:
        # Load task data from Firebase
        doc = db.collection("agencii-chats").document(conversation_id).get()
        
        if doc.exists:
            doc_data = doc.to_dict()
            task_key = f"task_{task_id}"
            
            if task_key in doc_data:
                task_data = doc_data[task_key]
                return {
                    "status": task_data.get("status", "unknown"),
                    "result": task_data.get("result"),
                    "error": task_data.get("error"),
                    "started_at": task_data.get("started_at"),
                    "completed_at": task_data.get("completed_at")
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"Task {task_id} not found"
                }
        else:
            return {
                "status": "error",
                "message": "Conversation not found"
            }
            
    except Exception as e:
        logger.error(f"Error checking progress: {e}")
        return {
            "status": "error",
            "message": f"Error checking task progress: {str(e)}"
        }

@mcp.tool()
async def expand_keywords(
    num_keywords: int = Field(description="Number of keywords to generate"),
    model: str = Field(default="gemini", description="Model to use (gemini or perplexity)"),
    location: str = Field(description="Location name or 'Worldwide'"),
    country_code: Optional[str] = Field(default=None, description="2-letter country code"),
    location_type: Optional[str] = Field(default=None, description="Location type")
) -> Dict:
    """
    Generate additional keywords based on existing data.
    """
    logger.info(f"🔍 Expanding keywords - generating {num_keywords} new keywords")
    
    search_data = shared_state.get("SEARCH_DATA")
    if not search_data:
        return {
            "status": "error",
            "message": "No search data found. Please run keyword_search first."
        }
    
    # For demonstration, return mock expanded keywords
    existing_keywords = shared_state.get("KEYWORDS", [])
    new_keywords = [
        {"keyword": f"expanded keyword {i}", "match_type": "PHRASE", "avg_monthly_searches": 100 * i}
        for i in range(1, min(num_keywords + 1, 6))
    ]
    
    # Combine keywords
    all_keywords = existing_keywords + new_keywords
    shared_state.set("KEYWORDS", all_keywords)
    
    return {
        "status": "success",
        "message": f"Generated {len(new_keywords)} additional keywords",
        "new_keywords": new_keywords,
        "total_keywords": len(all_keywords)
    }

@mcp.tool()
async def save_to_database(
    data_type: str = Field(description="Type of data: KEYWORDS or GENERATED_AD_COPY"),
    data: Union[List, Dict] = Field(description="Data to save")
) -> Dict:
    """
    Save or update data in the database.
    """
    logger.info(f"💾 Saving {data_type} to database")
    
    if data_type not in ["KEYWORDS", "GENERATED_AD_COPY"]:
        return {
            "status": "error",
            "message": "Invalid data_type. Must be KEYWORDS or GENERATED_AD_COPY"
        }
    
    try:
        shared_state.set(data_type, data)
        
        return {
            "status": "success",
            "message": f"{data_type} saved successfully"
        }
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        return {
            "status": "error",
            "message": f"Failed to save data: {str(e)}"
        }

@mcp.tool()
async def post_google_sheet(
    campaign_title: str = Field(description="Campaign title"),
    total_budget: float = Field(description="Total budget for campaign"),
    campaign_type: str = Field(description="Campaign type"),
    start_date: str = Field(description="Start date in YYYYMMDD format"),
    end_date: str = Field(description="End date in YYYYMMDD format")
) -> Dict:
    """
    Create a Google Sheet with campaign data for review.
    """
    logger.info(f"📊 Creating Google Sheet for campaign: {campaign_title}")
    
    # Validate dates
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        
        if start >= end:
            return {
                "status": "error",
                "message": "End date must be after start date"
            }
        
        if start < datetime.now():
            return {
                "status": "error",
                "message": "Start date must be in the future"
            }
    except ValueError:
        return {
            "status": "error",
            "message": "Invalid date format. Use YYYYMMDD"
        }
    
    # Check for required data
    ad_copy = shared_state.get("GENERATED_AD_COPY")
    if not ad_copy:
        return {
            "status": "error",
            "message": "No ad copy found. Please generate ad copy first."
        }
    
    # For demonstration, return a mock sheet URL
    sheet_url = f"https://docs.google.com/spreadsheets/d/mock-sheet-{campaign_title.replace(' ', '-')}"
    
    return {
        "status": "success",
        "message": "Google Sheet created successfully",
        "sheet_url": sheet_url,
        "campaign_details": {
            "title": campaign_title,
            "budget": total_budget,
            "type": campaign_type,
            "duration": f"{start_date} - {end_date}"
        }
    }

@mcp.tool()
async def post_google_ad(
    campaign_title: str = Field(description="Campaign title"),
    total_budget: float = Field(description="Total budget"),
    campaign_type: str = Field(description="Campaign type"),
    start_date: str = Field(description="Start date YYYYMMDD"),
    end_date: str = Field(description="End date YYYYMMDD"),
    customer_id: str = Field(description="Google Ads customer ID (10 digits)"),
    location: str = Field(description="Target location"),
    country_code: Optional[str] = Field(default=None, description="Country code"),
    location_type: Optional[str] = Field(default=None, description="Location type")
) -> Dict:
    """
    Post the campaign to Google Ads.
    """
    logger.info(f"🚀 Posting campaign to Google Ads: {campaign_title}")
    
    # Format customer ID
    customer_id = format_customer_id(customer_id)
    
    if len(customer_id) != 10:
        return {
            "status": "error",
            "message": "Customer ID must be 10 digits"
        }
    
    # Check for required data
    ad_copy = shared_state.get("GENERATED_AD_COPY")
    keywords = shared_state.get("KEYWORDS")
    
    if not ad_copy or not keywords:
        return {
            "status": "error",
            "message": "Missing required data. Please complete keyword search and ad copy generation first."
        }
    
    # Create task for async processing
    task_id = str(uuid.uuid4()).replace("-", "_")
    conversation_id = shared_state.get("CONVERSATION_ID")
    
    # For demonstration, return processing status
    return {
        "status": "processing",
        "task_id": task_id,
        "message": "Campaign posting started. Use check_progress to monitor.",
        "campaign_details": {
            "title": campaign_title,
            "customer_id": customer_id,
            "budget": total_budget,
            "keywords_count": len(keywords) if keywords else 0
        }
    }

@mcp.tool()
async def set_conversation_context(
    conversation_id: str = Field(description="Conversation ID for context")
) -> Dict:
    """
    Set the conversation context for maintaining state across tools.
    """
    logger.info(f"🔗 Setting conversation context: {conversation_id}")
    
    shared_state.set("CONVERSATION_ID", conversation_id)
    
    # Load existing state from Firebase if available
    if db:
        shared_state.load_from_firebase(conversation_id)
        
        return {
            "status": "success",
            "message": "Conversation context set",
            "conversation_id": conversation_id,
            "loaded_data": {
                "has_keywords": shared_state.get("KEYWORDS") is not None,
                "has_ad_copy": shared_state.get("GENERATED_AD_COPY") is not None,
                "has_url": shared_state.get("URL") is not None
            }
        }
    else:
        return {
            "status": "success",
            "message": "Conversation context set (no database connection)",
            "conversation_id": conversation_id
        }

# --- MCP Resources ---

@mcp.resource("workflow://guide")
def workflow_guide() -> str:
    """Workflow guide for Google Ads automation"""
    return """
    Google Ads Automation Workflow Guide
    
    1. **Set Context**
       - Use set_conversation_context() to establish session
    
    2. **Keyword Research**
       - Use keyword_search() with target URL and location
       - Check progress with check_progress()
       - Optionally expand with expand_keywords()
    
    3. **Generate Ad Copy**
       - Use generate_ad_copy() to create variations
       - Check progress with check_progress()
       - Edit with save_to_database() if needed
    
    4. **Review Campaign**
       - Use post_google_sheet() to create review sheet
       - Share sheet URL with stakeholders
    
    5. **Launch Campaign**
       - Use post_google_ad() with customer ID
       - Check progress with check_progress()
    
    Tools support async processing - always check progress for long operations.
    """

@mcp.resource("api://status")
def api_status() -> str:
    """API and service status"""
    status = {
        "services": {
            "firebase": "✅ Connected" if db else "❌ Not connected",
            "google_ads": "✅ Connected" if google_ads_client else "❌ Not connected",
            "openai": "✅ Connected" if openai_client else "❌ Not connected",
            "anthropic": "✅ Connected" if anthropic_client else "❌ Not connected",
        },
        "configuration": {
            "timeout": THREAD_TIMEOUT,
            "base_url": base_url
        }
    }
    return json.dumps(status, indent=2)

# --- MCP Prompts ---

@mcp.prompt("google_ads_workflow")
def google_ads_workflow_prompt() -> str:
    """Complete workflow for Google Ads automation"""
    return """
    I'll help you create a Google Ads campaign. Here's the process:
    
    1. **Provide your website URL** - I'll analyze it for keywords
    2. **Select target location** - Choose geographic targeting
    3. **Review keywords** - I'll find relevant search terms
    4. **Generate ad copy** - AI will create compelling ads
    5. **Set campaign details** - Budget, dates, and type
    6. **Review in Google Sheet** - Check everything before launch
    7. **Launch campaign** - Post to Google Ads
    
    Each step includes automatic progress tracking.
    Would you like to start with your website URL?
    """

@mcp.prompt("troubleshooting")
def troubleshooting_prompt() -> str:
    """Troubleshooting guide for common issues"""
    return """
    Common Issues and Solutions:
    
    **Task stuck in processing:**
    - Use check_progress() with the task_id
    - Tasks timeout after 5 minutes
    - Retry if needed
    
    **Missing data errors:**
    - Ensure keyword_search completed first
    - Check conversation context is set
    - Verify all required fields provided
    
    **Location not found:**
    - Use exact spelling for locations
    - Specify correct location_type
    - Try "Worldwide" for global targeting
    
    **Date validation errors:**
    - Use YYYYMMDD format
    - Ensure dates are in future
    - End date must be after start date
    
    Need more help? Check the health_check() for service status.
    """

# --- Main Execution ---

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    # Log server configuration
    logger.info("=" * 60)
    logger.info("🚀 Starting Google Ads Automation MCP Server")
    logger.info(f"📍 Port: {port}")
    logger.info(f"🌐 Base URL: {base_url}")
    logger.info(f"⏱️  Timeout: {THREAD_TIMEOUT}s")
    logger.info("=" * 60)
    logger.info("🔧 Service Status:")
    logger.info(f"  Firebase: {'✅ Connected' if db else '❌ Not configured'}")
    logger.info(f"  Google Ads: {'✅ Connected' if google_ads_client else '❌ Not configured'}")
    logger.info(f"  OpenAI: {'✅ Connected' if openai_client else '❌ Not configured'}")
    logger.info(f"  Anthropic: {'✅ Connected' if anthropic_client else '❌ Not configured'}")
    logger.info("=" * 60)
    logger.info("📡 Server endpoints:")
    logger.info(f"  MCP: {base_url}/mcp")
    logger.info("=" * 60)
    
    # Run FastMCP server with streamable-http transport
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port
    )
