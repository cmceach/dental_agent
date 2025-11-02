import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangGraph and LangChain Imports
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, AnyMessage
from langchain_core.runnables import RunnableConfig

# Exa API Import
from exa_py import Exa

# Load environment variables
load_dotenv()


def get_dental_guideline_domains() -> List[str]:
    """
    Get the list of approved dental guideline domains from environment variables.
    Falls back to default domains if not specified in .env file.
    
    Returns:
        List[str]: List of domain names (without paths)
    """
    # Default domains if not configured
    default_domains = [
        # Professional dental associations
        "ada.org",        # American Dental Association
        "aapd.org",       # American Academy of Pediatric Dentistry
        "aae.org",        # American Association of Endodontists
        "aaop.org",       # American Academy of Periodontology
        "aaoms.org",      # American Association of Oral and Maxillofacial Surgeons
        "cdc.gov",        # CDC (includes oral health division)
        # Medical associations (pediatric and oral health)
        "aap.org",                    # American Academy of Pediatrics
        "publications.aap.org",       # AAP Publications (pediatric oral health guidelines)
        # Government regulatory agencies
        "fda.gov",                     # FDA (Food and Drug Administration - dental products and devices)
        # Evidence-based research sources
        "pubmed.ncbi.nlm.nih.gov",    # PubMed - biomedical literature database
        "nidcr.nih.gov",               # National Institute of Dental and Craniofacial Research
        "nih.gov",                     # National Institutes of Health (oral health resources)
        "cochranelibrary.com",         # Cochrane Library - systematic reviews
        "who.int",                     # World Health Organization (oral health)
        "iadr.org",                    # International Association for Dental Research
        "journals.ada.org",            # Journal of the American Dental Association
    ]
    
    # Try to get domains from environment variable
    env_domains = os.environ.get("DENTAL_GUIDELINE_DOMAINS")
    
    if env_domains:
        # Parse comma-separated or newline-separated domains
        domains = [
            domain.strip() 
            for domain in env_domains.replace('\n', ',').split(',') 
            if domain.strip()
        ]
        # Filter out empty strings and validate domains don't contain paths
        domains = [d for d in domains if d and '/' not in d]
        if domains:
            return domains
    
    # Return default domains if none configured
    return default_domains


# Get domains (cached after first call)
DENTAL_GUIDELINE_DOMAINS = get_dental_guideline_domains()


def initialize_exa_client() -> Exa:
    """
    Initialize and return the Exa client.
    
    Returns:
        Exa: Initialized Exa client
        
    Raises:
        EnvironmentError: If EXA_API_KEY is not found in environment variables
    """
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise EnvironmentError("EXA_API_KEY not found in environment variables.")
    
    try:
        return Exa(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Exa client: {e}")


def create_dental_tool(exa_client: Exa):
    """
    Create the dental guideline search tool.
    
    Args:
        exa_client: Initialized Exa client
        
    Returns:
        Tool: The dental_guideline_search tool
    """
    @tool
    def dental_guideline_search(query: str) -> str:
        """
        Searches for information and clinical guidelines from authoritative dental sources.
        
        Use this tool to find:
        - Clinical guidelines and recommendations
        - Best practices for dental procedures
        - Official position statements
        - Evidence-based treatment protocols
        
        The query should be specific and include relevant terms. Examples:
        - "ADA fluoride recommendations for children"
        - "AAPD guidelines for pediatric dental x-rays"
        - "CDC oral health prevention strategies"
        
        Args:
            query: A specific search query about dental guidelines, procedures, or recommendations
            
        Returns:
            Formatted search results with URLs, titles, and content snippets from trusted dental sources
        """
        try:
            logger.info(f"Searching for query: '{query}'")
            logger.info(f"Domains: {DENTAL_GUIDELINE_DOMAINS}")
            
            # Get date filter from environment (default: prioritize last 5 years)
            # Set MIN_DATE_YEARS_AGO in .env to change (e.g., "3" for 3 years, "0" to disable)
            min_date_years = int(os.environ.get("MIN_DATE_YEARS_AGO", "5"))
            start_date = None
            if min_date_years > 0:
                start_date = (datetime.now() - timedelta(days=365 * min_date_years)).strftime("%Y-%m-%d")
                logger.info(f"Prioritizing results from last {min_date_years} years (since {start_date})")
            
            # Use Exa's search_and_contents method with improved parameters
            # - include_domains: Restricts to authoritative dental sources only
            # - num_results: Get more results to improve relevance
            # - type="auto": Lets Exa choose between neural and keyword search
            # - start_published_date: Prioritize recent content (if configured)
            search_params = {
                "query": query,
                "include_domains": DENTAL_GUIDELINE_DOMAINS,
                "num_results": 8,  # Increased to get more relevant results
                "type": "auto",  # Automatically chooses best search type
                "text": {
                    "include_html_tags": False,
                    "max_characters": 3000  # Increased for more context
                }
            }
            
            # Add date filter if configured
            if start_date:
                search_params["start_published_date"] = start_date
            
            search_results = exa_client.search_and_contents(**search_params)
            
            logger.info(f"Found {len(search_results.results) if search_results.results else 0} results")

            # Format the results for the LLM with citation numbers
            formatted_results = ""
            if not search_results.results:
                return f"No relevant guidelines or information was found from the trusted dental sources for the query: '{query}'. Try rephrasing with more specific terms or including organization names (e.g., 'ADA', 'CDC', 'AAPD')."

            formatted_results += "SEARCH RESULTS WITH CITATIONS:\n"
            formatted_results += "When referencing information from these sources, use the citation number [1], [2], etc.\n\n"
            
            for i, result in enumerate(search_results.results, 1):
                formatted_results += f"[{i}] {result.title}\n"
                formatted_results += f"    URL: {result.url}\n"
                
                # Display publication date if available
                if hasattr(result, 'published_date') and result.published_date:
                    try:
                        # Parse and format the date
                        pub_date = result.published_date
                        if isinstance(pub_date, str):
                            # Try to parse ISO format dates
                            try:
                                date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                                formatted_date = date_obj.strftime("%B %d, %Y")
                                formatted_results += f"    Published: {formatted_date}\n"
                            except:
                                formatted_results += f"    Published: {pub_date}\n"
                        else:
                            formatted_results += f"    Published: {pub_date}\n"
                    except Exception as e:
                        logger.debug(f"Could not parse publication date: {e}")
                
                if result.text:
                    formatted_results += f"    Content: {result.text}\n"
                else:
                    formatted_results += f"    Content: (No text content available)\n"
                formatted_results += "\n"
            
            formatted_results += "\nIMPORTANT: When you reference information from these sources in your response, you MUST include inline citations using the format [1], [2], etc. corresponding to the source numbers above.\n"
            
            return formatted_results

        except Exception as e:
            return f"Error: Could not perform search. {str(e)}"
    
    return dental_guideline_search


def create_citation_prompt(state: AgentState, config: RunnableConfig) -> List[AnyMessage]:
    """
    Create a system prompt that instructs the agent to include inline citations.
    
    Args:
        state: The agent state
        config: Runnable configuration
        
    Returns:
        List of messages including system message and state messages
    """
    logger.debug(f"Creating citation prompt. State has {len(state.get('messages', []))} messages")
    
    citation_instructions = """You are a dental guideline assistant that provides evidence-based information from authoritative sources.

CRITICAL CITATION REQUIREMENTS:
- When you reference information from search results, you MUST include inline citations using the format [1], [2], [3], etc.
- These citation numbers correspond to the numbered sources provided in the search results.
- Always cite your sources immediately after referencing information from them.
- Use multiple citations if information comes from multiple sources: [1][2]
- Include a "Sources" section at the end of your response listing all cited sources with their titles and URLs.
- IMPORTANT: In the Sources section, put each citation on its own separate line for readability.

Example citation format:
"The American Dental Association recommends fluoride use [1]. Research shows it reduces tooth decay by 25% [2]."

Example Sources section (each citation as a bullet point on its own line):
## Sources
- [1] Title of Source 1 - URL
- [2] Title of Source 2 - URL
- [3] Title of Source 3 - URL

Format each citation as: - [number] Title - URL (use a bullet point "- " before each citation, and put each citation on its own separate line)

DISCLAIMER REQUIREMENT:
- If the search results are unable to confirm a definitive diagnosis, provide insufficient information to reach a conclusion, or the evidence is inconclusive, you MUST include a disclaimer at the end of your response.
- The disclaimer should direct users to post their inquiry in a forum for further discussion and professional consultation.
- Use this format for the disclaimer:

**Disclaimer:** The information provided is based on available guidelines and research. If the search results are unable to confirm a definitive diagnosis or reach a clear conclusion regarding your specific situation, please post your inquiry in the forum for further discussion and professional consultation.

Be thorough, accurate, and always cite your sources. Include the disclaimer when appropriate."""
    
    # Return system message followed by all messages from state
    return [SystemMessage(content=citation_instructions)] + state["messages"]


def create_agent():
    """
    Initialize the LLM, tools, and create the LangGraph ReAct agent.
    
    Returns:
        CompiledGraph: The compiled LangGraph agent
        
    Raises:
        EnvironmentError: If GOOGLE_API_KEY is not found in environment variables
    """
    # Validate API key
    if "GOOGLE_API_KEY" not in os.environ:
        raise EnvironmentError("GOOGLE_API_KEY not found in environment variables.")
    
    # Initialize Exa client and create tool
    exa_client = initialize_exa_client()
    dental_tool = create_dental_tool(exa_client)
    
    # List of all tools the agent has access to
    tools = [dental_tool]
    
    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("MODEL", "gemini-2.5-flash"),
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        # This helps Gemini models better understand system prompts
        convert_system_message_to_human=True 
    )
    
    # Create the LangGraph ReAct Agent with citation system prompt
    # `create_react_agent` is a prebuilt graph factory.
    # It automatically creates the ReAct (Reason-Act) loop:
    # 1. Agent (LLM) decides to act
    # 2. Tool is executed
    # 3. Tool output is sent back to the Agent
    # 4. Agent synthesizes the final answer with citations
    app = create_react_agent(
        llm, 
        tools,
        prompt=create_citation_prompt
    )
    
    return app

