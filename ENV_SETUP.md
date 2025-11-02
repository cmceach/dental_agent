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

## Complete .env Example

```bash
# Required API Keys
GOOGLE_API_KEY=your_google_api_key_here
EXA_API_KEY=your_exa_api_key_here

# Optional: Override model
# MODEL=gemini-2.5-flash

# Optional: Override domains (uncomment to use custom list)
# DENTAL_GUIDELINE_DOMAINS=ada.org,aapd.org,cdc.gov,pubmed.ncbi.nlm.nih.gov
```

## Quick Start

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys

3. If you want custom domains, uncomment and modify `DENTAL_GUIDELINE_DOMAINS`

4. That's it! The agent will use defaults if `DENTAL_GUIDELINE_DOMAINS` is not set.
