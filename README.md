# Google Ads Automation MCP Server with Firecrawl

A Model Context Protocol (MCP) server for automating Google Ads campaigns with integrated Firecrawl web scraping capabilities.

## Features

### 🚀 Firecrawl-Powered Web Scraping
- Industrial-strength web scraping and content extraction
- No custom scraping code to maintain
- AI-powered content understanding
- Web search capabilities
- Handles JavaScript-heavy sites

### 🔍 Intelligent Keyword Research
- Automatic keyword extraction from any URL
- Google Ads API integration for search volume data
- Smart match type assignment (BROAD, PHRASE, EXACT)
- Theme-based keyword grouping for ad groups
- Keyword expansion with AI suggestions

### 📝 Multi-Model Ad Copy Generation
- Generate variations using GPT-4 and Claude
- Theme-based ad copy for each keyword group
- Automatic character limit enforcement
- Headlines (max 30 chars) and descriptions (max 90 chars)

### 📊 Campaign Management
- Google Sheets integration for campaign review
- Direct Google Ads API posting
- Budget and schedule management
- Location targeting

## Quick Start

### Prerequisites

1. **API Keys Required:**
   - `FIRECRAWL_API_KEY` - Get from [Firecrawl](https://firecrawl.dev)
   - `GOOGLE_ADS_DEVELOPER_TOKEN` - From Google Ads API Center
   - `GOOGLE_ADS_MANAGER_ID` - Your Google Ads Manager Account ID
   - `SERVICE_ACCOUNT_KEY_ADS` - Google service account JSON for Ads
   - `SERVICE_ACCOUNT_KEY_FIREBASE` - Firebase service account JSON (optional)
   
2. **Optional AI Keys:**
   - `OPENAI_API_KEY` - For GPT-4 ad copy generation
   - `ANTHROPIC_API_KEY` - For Claude ad copy generation
   - `GEMINI_API_KEY` - For additional keyword suggestions

### Deployment on Railway

1. Fork this repository
2. Connect your GitHub to Railway
3. Create a new project from the repository
4. Add environment variables in Railway dashboard
5. Deploy!

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/gads-mcp-server.git
cd gads-mcp-server

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the server
python gads-mcp-server.py
```

### Configure Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "google-ads": {
      "url": "https://your-railway-url.railway.app",
      "transport": "streamable-http",
      "name": "Google Ads Automation"
    }
  }
}
```

## Workflow

### 1. Keyword Research
Start by analyzing any URL to extract relevant keywords:

```python
keyword_research(
    url="https://example.com",
    location="United States",
    location_type="Country"
)
```

### 2. Expand Keywords (Optional)
Add more related keywords using AI and web search:

```python
expand_keywords(
    conversation_id="...",
    num_keywords=20,
    use_ai_suggestions=True
)
```

### 3. Generate Ad Copy
Create themed ad copy variations:

```python
generate_ad_copy(
    conversation_id="..."
)
```

### 4. Create Campaign Sheet
Export to Google Sheets for review:

```python
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
Deploy to Google Ads:

```python
launch_campaign(
    conversation_id="...",
    customer_id="1234567890",
    campaign_title="My Campaign",
    total_budget=1000,
    campaign_type="SEARCH",
    start_date="20250201",
    end_date="20250228",
    location="United States"
)
```

## Environment Variables

### Required
- `FIRECRAWL_API_KEY` - Firecrawl API key for web scraping
- `GOOGLE_ADS_DEVELOPER_TOKEN` - Google Ads API developer token
- `GOOGLE_ADS_MANAGER_ID` - Google Ads manager account ID
- `SERVICE_ACCOUNT_KEY_ADS` - Google service account JSON (stringified)

### Optional
- `FIRECRAWL_API_URL` - Custom Firecrawl API URL (for self-hosted)
- `SERVICE_ACCOUNT_KEY_FIREBASE` - Firebase service account JSON
- `OPENAI_API_KEY` - OpenAI API key for GPT-4
- `ANTHROPIC_API_KEY` - Anthropic API key for Claude
- `GEMINI_API_KEY` - Google Gemini API key
- `GOOGLE_SHEET_TEMPLATE` - Template sheet ID
- `GOOGLE_DRIVE_FOLDER_ID` - Folder for saving sheets
- `THREAD_TIMEOUT` - Timeout for async operations (default: 300)

## Files Structure

```
gads-mcp-server/
├── gads-mcp-server.py      # Main server file
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── railway.json            # Railway deployment config
├── .env.example            # Environment variables template
├── geotargets-2024-10-10.csv  # Location data
└── README.md               # This file
```

## Benefits of Firecrawl Integration

1. **Reliability**: Industrial-strength web scraping that handles complex sites
2. **Speed**: Optimized content extraction with caching
3. **AI-Powered**: Smart content understanding and keyword extraction
4. **Maintenance-Free**: No scraping code to maintain or update
5. **Scale**: Handle any website complexity without custom code

## Troubleshooting

### Common Issues

1. **Firecrawl API Key Missing**
   - Ensure `FIRECRAWL_API_KEY` is set in environment variables
   - Get your key from [https://firecrawl.dev](https://firecrawl.dev)

2. **Google Ads API Errors**
   - Verify your developer token is approved
   - Check that the manager account ID is correct
   - Ensure service account has proper permissions

3. **No Keywords Extracted**
   - Check if the URL is accessible
   - Verify Firecrawl API is working
   - Try a different URL with more content

## Support

For issues or questions:
- Open an issue on GitHub
- Check Firecrawl documentation at [docs.firecrawl.dev](https://docs.firecrawl.dev)
- Review Google Ads API docs at [developers.google.com/google-ads/api](https://developers.google.com/google-ads/api)

## License

MIT License - see LICENSE file for details
