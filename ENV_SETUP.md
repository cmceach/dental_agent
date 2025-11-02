# .env File Setup Guide

## Basic Setup (Required)

Create a `.env` file in the project root with your API keys:

```bash
# Google Gemini API Key
GOOGLE_API_KEY=your_google_api_key_here

# Exa API Key
EXA_API_KEY=your_exa_api_key_here
```

## Domain Configuration

### Option 1: Use Default Domains (Recommended)

**You don't need to do anything!** The agent automatically uses all 15 default domains:

**Professional Dental Associations (6):**
- ada.org, aapd.org, aae.org, aaop.org, aaoms.org, cdc.gov

**Medical Associations (2):**
- aap.org, publications.aap.org

**Evidence-Based Research Sources (7):**
- pubmed.ncbi.nlm.nih.gov, nidcr.nih.gov, nih.gov, cochranelibrary.com, who.int, iadr.org, journals.ada.org

### Option 2: Override with Custom Domains

If you want to use a custom subset of domains, add this to your `.env`:

```bash
# Custom domain list (comma-separated)
DENTAL_GUIDELINE_DOMAINS=ada.org,aapd.org,cdc.gov,pubmed.ncbi.nlm.nih.gov,aap.org
```

**Important Notes:**
- Use only domain names (e.g., `cdc.gov`), not paths (e.g., `cdc.gov/oralhealth`)
- Separate multiple domains with commas
- No spaces around commas (or spaces will be trimmed automatically)

### Option 3: Optional Model Override

To use a different Gemini model:

```bash
MODEL=gemini-2.0-flash-exp
```

## Search Configuration

### Search Results Configuration

Control how many results are returned per search:

```bash
# Number of search results per query (default: 8)
SEARCH_RESULTS_COUNT=8

# Maximum characters per search result (default: 3000)
MAX_CHARACTERS_PER_RESULT=3000
```

### Agent Iteration Limits

Control how many times the agent can call tools:

```bash
# Maximum recursion limit for agent iterations (default: 25)
# This controls how many tool calls the agent can make per query
# Formula: typically allows ~12 search iterations (2 * max_iterations + 1)
RECURSION_LIMIT=25
```

### PDF Auto-Upload Configuration

Automatically download and upload PDFs from search results:

```bash
# Enable automatic PDF upload from search results (default: false)
AUTO_UPLOAD_PDFS=true

# Maximum PDF size in MB for auto-upload (default: 25)
MAX_PDF_SIZE_MB=25
```

**How it works:**
- When enabled, PDFs found in search results are automatically downloaded and uploaded to Gemini File API
- The agent can then reference PDF content directly instead of just URLs
- Checks for existing files in Gemini API before uploading to avoid duplicates
- If download/upload fails, falls back to URL-only citation

**Recommendations:**
- `SEARCH_RESULTS_COUNT`: 5-10 results is usually sufficient (default: 8)
- `MAX_CHARACTERS_PER_RESULT`: 2000-5000 characters (default: 3000)
- `RECURSION_LIMIT`: 15-50 iterations (default: 25 allows ~12 search calls)
- `AUTO_UPLOAD_PDFS`: Enable if you want PDFs from search results to be directly accessible (default: false)
- `MAX_PDF_SIZE_MB`: 10-50 MB (default: 25) - larger files take longer to download/upload

## Complete .env Example

```bash
# Required API Keys
GOOGLE_API_KEY=your_google_api_key_here
EXA_API_KEY=your_exa_api_key_here

# Optional: Override model
# MODEL=gemini-2.5-flash

# Optional: Override domains (uncomment to use custom list)
# DENTAL_GUIDELINE_DOMAINS=ada.org,aapd.org,cdc.gov,pubmed.ncbi.nlm.nih.gov

# Optional: Search configuration
# SEARCH_RESULTS_COUNT=8
# MAX_CHARACTERS_PER_RESULT=3000
# RECURSION_LIMIT=25

# Optional: PDF auto-upload from search results
# AUTO_UPLOAD_PDFS=true  # Enable to auto-upload PDFs from search results
# MAX_PDF_SIZE_MB=25     # Maximum PDF size for auto-upload

# Optional: Date filtering (prioritize recent guidelines)
# MIN_DATE_YEARS_AGO=5  # Set to 0 to disable
```

## Quick Start

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys

3. If you want custom domains, uncomment and modify `DENTAL_GUIDELINE_DOMAINS`

4. That's it! The agent will use defaults if `DENTAL_GUIDELINE_DOMAINS` is not set.
