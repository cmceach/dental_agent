# 🦷 Dental Guideline Agent

A Streamlit-based chat interface for querying authoritative dental guidelines using LangGraph and Google Gemini. The agent searches only trusted dental association websites to provide accurate, evidence-based information.

## Features

- 🤖 **ReAct Agent**: Uses LangGraph's prebuilt ReAct agent for reasoning and tool execution
- 🔍 **Domain-Restricted Search**: Searches only authoritative dental sources (ADA, AAPD, AAE, AAOP, AAOMS, CDC)
- 💬 **Streamlit Chat UI**: Interactive chat interface with streaming responses
- 📊 **Tool Visualization**: See tool calls and reasoning steps in real-time
- 💾 **Conversation Memory**: Maintains conversation context throughout the session

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- Exa API key ([Get one here](https://exa.ai/))

## Setup

### 1. Clone or navigate to the repository

```bash
cd dental_agent
```

### 2. Create a conda environment (recommended)

```bash
conda create -n dental_agent python=3.11 -y
conda activate dental_agent
pip install -r requirements.txt
```

**Alternative: Using venv**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
GOOGLE_API_KEY=your_google_api_key_here
EXA_API_KEY=your_exa_api_key_here
```

**Optional Configuration:**

The agent uses **15 default domains** by default (see [Authoritative Sources](#authoritative-sources) section below). You don't need to configure anything unless you want to override the defaults.

**To override domains**, add to your `.env`:

```
# Optional: Override model (default: gemini-2.5-flash)
MODEL=gemini-2.5-flash

# Optional: Custom domain list (comma-separated)
# If not set, all 15 default domains are used automatically
DENTAL_GUIDELINE_DOMAINS=ada.org,aapd.org,cdc.gov,pubmed.ncbi.nlm.nih.gov,aap.org
```

**Additional Configuration:**

To prioritize recent guidelines (default: last 5 years):
```
MIN_DATE_YEARS_AGO=5  # Set to 0 to disable date filtering
```

**Notes:** 
- When specifying domains, use only domain names (e.g., `cdc.gov`), not paths (e.g., `cdc.gov/oralhealth`)
- Separate multiple domains with commas
- If `DENTAL_GUIDELINE_DOMAINS` is not set, the agent automatically uses all default domains
- Publication dates are displayed in search results to verify recency

## Usage

### Run the Streamlit app

**With conda environment:**
```bash
conda activate dental_agent
streamlit run streamlit_app.py
```

**With venv:**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`.

### Using the Agent

1. Enter your question in the chat input at the bottom
2. The agent will:
   - Analyze your question
   - Search authoritative dental sources
   - Synthesize a response based on the findings
3. View tool execution details in the expandable section
4. Check recent tool calls in the sidebar

### Example Questions

- "What is the ADA guideline for using amalgam?"
- "What are the recommendations for pediatric dental x-rays?"
- "What does the CDC say about oral health and systemic disease?"

## Project Structure

```
dental_agent/
├── agent.py              # Core agent logic and initialization
├── streamlit_app.py      # Streamlit UI application
├── requirements.txt      # Python dependencies
├── .env.example         # Example environment variables
├── .gitignore           # Git ignore rules
└── README.md             # This file
```

## Architecture

### Agent Components

- **LLM**: Google Gemini 2.5 Flash (via LangChain)
- **Tool**: Exa API search restricted to dental guideline domains
- **Graph**: LangGraph ReAct agent (prebuilt)

### Authoritative Sources

The agent searches these domains by default:

**Professional Dental Associations:**
- `ada.org` - American Dental Association
- `aapd.org` - American Academy of Pediatric Dentistry
- `aae.org` - American Association of Endodontists
- `aaop.org` - American Academy of Periodontology
- `aaoms.org` - American Association of Oral and Maxillofacial Surgeons
- `cdc.gov` - CDC (includes oral health division)

**Medical Associations:**
- `aap.org` - American Academy of Pediatrics (pediatric oral health guidelines)
- `publications.aap.org` - AAP Publications (clinical guidelines and recommendations)

**Government Regulatory Agencies:**
- `fda.gov` - FDA (Food and Drug Administration - dental products, devices, and safety)

**Evidence-Based Research Sources:**
- `pubmed.ncbi.nlm.nih.gov` - PubMed biomedical literature database
- `nidcr.nih.gov` - National Institute of Dental and Craniofacial Research
- `nih.gov` - National Institutes of Health (oral health resources)
- `cochranelibrary.com` - Cochrane Library (systematic reviews)
- `who.int` - World Health Organization (oral health)
- `iadr.org` - International Association for Dental Research
- `journals.ada.org` - Journal of the American Dental Association

**Customizing Domains:**

To override the default domains, set `DENTAL_GUIDELINE_DOMAINS` in your `.env` file:
```
DENTAL_GUIDELINE_DOMAINS=ada.org,aapd.org,cdc.gov,pubmed.ncbi.nlm.nih.gov
```

## Troubleshooting

### API Key Errors

If you see errors about missing API keys:
1. Ensure `.env` file exists in the project root
2. Verify keys are correctly named (`GOOGLE_API_KEY`, `EXA_API_KEY`)
3. Restart the Streamlit app after adding keys

### Import Errors

If you encounter import errors:
1. Ensure virtual environment is activated
2. Run `pip install -r requirements.txt` again
3. Check Python version (3.8+ required)

## License

This project is provided as-is for educational and research purposes.

