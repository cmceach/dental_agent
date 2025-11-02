import os
import logging
import time
import tempfile
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import requests
import google.generativeai as genai

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


def detect_pdf_url(url: str) -> bool:
    """
    Check if a URL points to a PDF file.
    
    Args:
        url: URL to check
        
    Returns:
        bool: True if URL appears to be a PDF
    """
    url_lower = url.lower()
    return url_lower.endswith('.pdf') or '.pdf?' in url_lower or 'application/pdf' in url_lower


def check_existing_gemini_file(url: str, gemini_files_cache: Optional[List] = None) -> Optional[Any]:
    """
    Check if a PDF with the same URL already exists in Gemini File API.
    
    Args:
        url: Source URL of the PDF
        gemini_files_cache: Optional cached list of Gemini files to avoid repeated API calls
        
    Returns:
        Optional gemini_file object if found, None otherwise
    """
    try:
        # Initialize Gemini if not already configured
        try:
            # Check if already configured by trying to access api_key
            _ = genai.api_key
        except (AttributeError, Exception):
            # Not configured, try to configure it
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            else:
                return None
        
        # Get list of files (use cache if provided, otherwise fetch)
        if gemini_files_cache is None:
            try:
                files = list(genai.list_files())
            except Exception as e:
                logger.debug(f"Could not list Gemini files: {e}")
                return None
        else:
            files = gemini_files_cache
        
        # Extract filename from URL for matching
        url_filename = os.path.basename(url.split('?')[0])  # Remove query params
        
        # Check each file for URL match
        for file in files:
            # Check if file name matches or if we can match by URL in metadata
            if hasattr(file, 'name'):
                file_name = file.name.lower() if hasattr(file, 'name') else ''
                if url_filename.lower() in file_name or url in str(file_name):
                    logger.info(f"Found existing Gemini file for URL: {url}")
                    return file
        
        return None
    except Exception as e:
        logger.debug(f"Error checking existing Gemini file: {e}")
        return None


def download_pdf_from_url(url: str, max_size_mb: int = 25, timeout: int = 30) -> Optional[bytes]:
    """
    Download a PDF from a URL with size and timeout limits.
    
    Args:
        url: URL to download PDF from
        max_size_mb: Maximum file size in MB (default: 25)
        timeout: Request timeout in seconds (default: 30)
        
    Returns:
        PDF bytes if successful, None on failure
    """
    try:
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # Make request with timeout and stream to check size
        response = requests.get(url, timeout=timeout, stream=True, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; DentalAgent/1.0)'
        })
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
            logger.debug(f"URL does not appear to be PDF: {url}")
            return None
        
        # Check content length if available
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > max_size_bytes:
            logger.warning(f"PDF too large ({content_length} bytes > {max_size_bytes} bytes): {url}")
            return None
        
        # Download with size check
        pdf_bytes = b''
        for chunk in response.iter_content(chunk_size=8192):
            pdf_bytes += chunk
            if len(pdf_bytes) > max_size_bytes:
                logger.warning(f"PDF exceeds size limit during download: {url}")
                return None
        
        logger.info(f"Successfully downloaded PDF ({len(pdf_bytes)} bytes) from: {url}")
        return pdf_bytes
        
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout downloading PDF: {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error downloading PDF from {url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error downloading PDF from {url}: {e}")
        return None


def get_pdf_text_from_gemini(gemini_file_uri: str) -> Optional[str]:
    """
    Retrieve the extracted text content from a PDF file in Gemini File API.
    This is useful for debugging what text Gemini is actually extracting from PDFs.
    
    Args:
        gemini_file_uri: The URI of the file in Gemini File API
        
    Returns:
        Extracted text content as string, or None on failure
    """
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found")
            return None
        
        genai.configure(api_key=api_key)
        
        # Find the file by URI
        files = list(genai.list_files())
        gemini_file = None
        for file in files:
            if hasattr(file, 'uri') and file.uri == gemini_file_uri:
                gemini_file = file
                break
        
        if not gemini_file:
            logger.warning(f"File not found for URI: {gemini_file_uri}")
            return None
        
        # Use Gemini to extract text from the PDF
        # Create a simple model instance to extract text
        # Use the same model as configured in the agent
        model_name = os.environ.get("MODEL", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)
        
        # Use the file in a simple prompt to get its text content
        # Request detailed table extraction with structure preservation
        response = model.generate_content([
            "Extract and return the full text content of this PDF document, preserving tables and formatting as much as possible. "
            "For tables, include: (1) Table title/heading, (2) Column headers, (3) All row data with proper column alignment, "
            "(4) Any footnotes or notes. Format tables clearly showing which items belong in which columns. "
            "Include all text exactly as it appears in the document.",
            genai.get_file(gemini_file.name)
        ])
        
        if response and response.text:
            return response.text
        else:
            logger.warning("No text content returned from Gemini")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting text from Gemini file: {e}")
        return None


def upload_pdf_to_gemini(pdf_bytes: bytes, filename: str, source_url: str, gemini_files_cache: Optional[List] = None) -> Optional[Any]:
    """
    Upload a PDF to Gemini File API, checking for existing files first.
    
    Args:
        pdf_bytes: PDF file content as bytes
        filename: Name for the PDF file
        source_url: Original URL where PDF was downloaded from
        gemini_files_cache: Optional cached list of Gemini files
        
    Returns:
        gemini_file object if successful, None on failure
    """
    try:
        # Initialize Gemini if not already configured
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found")
            return None
        
        genai.configure(api_key=api_key)
        
        # Check if file already exists
        existing_file = check_existing_gemini_file(source_url, gemini_files_cache)
        if existing_file:
            logger.info(f"Reusing existing Gemini file for: {source_url}")
            return existing_file
        
        # Save to temporary file (required by genai.upload_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Upload to Gemini File API
            gemini_file = genai.upload_file(
                path=tmp_path,
                mime_type="application/pdf"
            )
            
            # Wait for file to be processed
            while gemini_file.state.name != "ACTIVE":
                time.sleep(1)
                gemini_file = genai.get_file(gemini_file.name)
            
            logger.info(f"Successfully uploaded PDF to Gemini: {filename}")
            return gemini_file
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error uploading PDF to Gemini: {e}")
        return None


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
            Formatted search results with URLs, titles, and content snippets from trusted dental sources.
            May include auto-uploaded PDF references encoded at the end.
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
            
            # Get configurable search parameters from environment variables
            # Set SEARCH_RESULTS_COUNT in .env to change (default: 8)
            num_results = int(os.environ.get("SEARCH_RESULTS_COUNT", "8"))
            # Set MAX_CHARACTERS_PER_RESULT in .env to change (default: 3000)
            max_characters = int(os.environ.get("MAX_CHARACTERS_PER_RESULT", "3000"))
            
            # Check if auto-upload PDFs is enabled
            auto_upload_pdfs = os.environ.get("AUTO_UPLOAD_PDFS", "false").lower() == "true"
            max_pdf_size_mb = int(os.environ.get("MAX_PDF_SIZE_MB", "25"))
            
            # Use Exa's search_and_contents method with improved parameters
            # - include_domains: Restricts to authoritative dental sources only
            # - num_results: Configurable number of results per search
            # - type="auto": Lets Exa choose between neural and keyword search
            # - start_published_date: Prioritize recent content (if configured)
            search_params = {
                "query": query,
                "include_domains": DENTAL_GUIDELINE_DOMAINS,
                "num_results": num_results,
                "type": "auto",  # Automatically chooses best search type
                "text": {
                    "include_html_tags": False,
                    "max_characters": max_characters
                }
            }
            
            # Add date filter if configured
            if start_date:
                search_params["start_published_date"] = start_date
            
            search_results = exa_client.search_and_contents(**search_params)
            
            logger.info(f"Found {len(search_results.results) if search_results.results else 0} results")

            # Format the results for the LLM with citation numbers
            formatted_results = ""
            auto_uploaded_pdfs = []
            
            if not search_results.results:
                return f"No relevant guidelines or information was found from the trusted dental sources for the query: '{query}'. Try rephrasing with more specific terms or including organization names (e.g., 'ADA', 'CDC', 'AAPD')."

            formatted_results += "SEARCH RESULTS WITH CITATIONS:\n"
            formatted_results += "When referencing information from these sources, use the citation number [1], [2], etc.\n\n"
            
            # Cache Gemini files list if auto-upload is enabled (to avoid repeated API calls)
            gemini_files_cache = None
            if auto_upload_pdfs:
                try:
                    api_key = os.environ.get("GOOGLE_API_KEY")
                    if api_key:
                        genai.configure(api_key=api_key)
                        gemini_files_cache = list(genai.list_files())
                        logger.info(f"Cached {len(gemini_files_cache)} existing Gemini files")
                except Exception as e:
                    logger.debug(f"Could not cache Gemini files list: {e}")
            
            for i, result in enumerate(search_results.results, 1):
                formatted_results += f"[{i}] {result.title}\n"
                formatted_results += f"    URL: {result.url}\n"
                
                # Check if this is a PDF URL and auto-upload is enabled
                pdf_uploaded = False
                if auto_upload_pdfs and detect_pdf_url(result.url):
                    try:
                        # Check if already uploaded
                        existing_file = check_existing_gemini_file(result.url, gemini_files_cache)
                        was_reused = existing_file is not None
                        
                        if existing_file:
                            gemini_file = existing_file
                            pdf_uploaded = True
                            logger.info(f"Reusing existing Gemini file for PDF: {result.url}")
                        else:
                            # Download PDF
                            pdf_bytes = download_pdf_from_url(result.url, max_size_mb=max_pdf_size_mb)
                            if pdf_bytes:
                                # Extract filename from URL
                                filename = os.path.basename(result.url.split('?')[0]) or f"document_{i}.pdf"
                                
                                # Upload to Gemini
                                gemini_file = upload_pdf_to_gemini(pdf_bytes, filename, result.url, gemini_files_cache)
                                if gemini_file:
                                    pdf_uploaded = True
                                    logger.info(f"Auto-uploaded PDF to Gemini: {filename}")
                        
                        if pdf_uploaded:
                            # Store PDF reference
                            auto_uploaded_pdfs.append({
                                "url": result.url,
                                "filename": os.path.basename(result.url.split('?')[0]) or f"document_{i}.pdf",
                                "gemini_file": gemini_file,
                                "is_auto_uploaded": True,
                                "was_reused": was_reused,
                                "citation_number": i
                            })
                            
                            # Add note in formatted results
                            if was_reused:
                                formatted_results += f"    📄 PDF available for direct reference (reused from previous upload)\n"
                                formatted_results += f"    ⚠️ NOTE: This PDF was auto-uploaded. If you cite both the search result URL [{i}] and the PDF [PDF], only list the PDF in Sources section (not the URL separately).\n"
                            else:
                                formatted_results += f"    📄 PDF auto-uploaded and available for direct reference\n"
                                formatted_results += f"    ⚠️ NOTE: This PDF was auto-uploaded. If you cite both the search result URL [{i}] and the PDF [PDF], only list the PDF in Sources section (not the URL separately).\n"
                                
                    except Exception as e:
                        logger.warning(f"Failed to auto-upload PDF from {result.url}: {e}")
                        # Continue with normal URL citation
                
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
            
            formatted_results += "\nIMPORTANT CITATION RULES:\n"
            formatted_results += "- When you reference information from these sources in your response, you MUST include inline citations using the format [1], [2], etc. corresponding to the source numbers above.\n"
            formatted_results += "- In your Sources section, ONLY list the sources you actually cited/referenced in your response text.\n"
            formatted_results += "- Do NOT list all available sources - only include the ones you used.\n"
            formatted_results += "- CRITICAL: In the Sources section, you MUST renumber citations sequentially starting from [1], even if your inline citations used different numbers.\n"
            formatted_results += "- Example: If you cited [1], [2], [7], [11] in your text, list them in Sources as [1], [2], [3], [4] respectively.\n"
            formatted_results += "- The Sources section should always start at [1] and continue sequentially [2], [3], [4], etc.\n"
            
            # Encode auto-uploaded PDFs info at the end of the return string
            # Format: <AUTO_UPLOADED_PDFS>JSON</AUTO_UPLOADED_PDFS>
            if auto_uploaded_pdfs:
                import json
                # Store only essential info (can't serialize gemini_file object, store URI)
                pdf_info = []
                for pdf in auto_uploaded_pdfs:
                    gemini_file = pdf.get("gemini_file")
                    uri = None
                    if gemini_file and hasattr(gemini_file, 'uri'):
                        uri = gemini_file.uri
                    
                    pdf_info.append({
                        "url": pdf["url"],
                        "filename": pdf["filename"],
                        "uri": uri,
                        "citation_number": pdf["citation_number"],
                        "was_reused": pdf["was_reused"]
                    })
                pdf_json = json.dumps(pdf_info)
                formatted_results += f"\n<AUTO_UPLOADED_PDFS>{pdf_json}</AUTO_UPLOADED_PDFS>"
            
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
    
    # Check if any messages contain PDF/media files and extract filenames
    has_pdf = False
    pdf_filenames = []
    auto_uploaded_pdfs = []
    for message in state.get("messages", []):
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'media':
                        has_pdf = True
                        # Try to get filename from file_part metadata or from text message
                        filename = item.get('filename') or "uploaded PDF document"
                        is_auto_uploaded = item.get('is_auto_uploaded', False)
                        source_url = item.get('source_url', '')
                        
                        # Also check if filename is mentioned in text content
                        if filename == "uploaded PDF document":
                            # Look for PDF filenames in text messages
                            for msg in state.get("messages", []):
                                if hasattr(msg, 'content'):
                                    msg_content = msg.content
                                    if isinstance(msg_content, list):
                                        for msg_item in msg_content:
                                            if isinstance(msg_item, dict) and msg_item.get('type') == 'text':
                                                text = msg_item.get('text', '')
                                                # Look for "PDF document(s) are attached: filename"
                                                if "PDF document(s) are attached:" in text:
                                                    import re
                                                    match = re.search(r'attached:\s*([^\n]+)', text, re.IGNORECASE)
                                                    if match:
                                                        filenames_text = match.group(1).strip()
                                                        # Parse both user and auto-uploaded PDFs
                                                        if "Auto-uploaded PDF(s)" in filenames_text:
                                                            # Extract auto-uploaded PDF names
                                                            auto_match = re.search(r'Auto-uploaded PDF\(s\) from search results:\s*([^;]+)', filenames_text)
                                                            if auto_match:
                                                                auto_names = [f.strip() for f in auto_match.group(1).split(',')]
                                                                pdf_filenames.extend(auto_names)
                                                                # Mark as auto-uploaded
                                                                for name in auto_names:
                                                                    auto_uploaded_pdfs.append({"filename": name, "url": ""})
                                                        # Extract user-uploaded PDF names
                                                        if "User-uploaded PDF(s)" in filenames_text:
                                                            user_match = re.search(r'User-uploaded PDF\(s\):\s*([^;]+)', filenames_text)
                                                            if user_match:
                                                                user_names = [f.strip() for f in user_match.group(1).split(',')]
                                                                pdf_filenames.extend(user_names)
                                                    break
                                        if pdf_filenames:
                                            break
                        if filename != "uploaded PDF document":
                            pdf_filenames.append(filename)
                            if is_auto_uploaded and source_url:
                                auto_uploaded_pdfs.append({"filename": filename, "url": source_url})
    
    # Remove duplicates while preserving order
    pdf_filenames = list(dict.fromkeys(pdf_filenames))
    
    pdf_reminder = ""
    if has_pdf:
        pdf_names_str = ", ".join(pdf_filenames) if pdf_filenames else "uploaded PDF document"
        pdf_reminder = f"\n\nPDF DETECTED: You have received PDF document(s): {pdf_names_str}\n"
        pdf_reminder += "- If you reference information from the PDF in your response, you MUST cite it as [PDF] in your response text.\n"
        
        # Check if any PDFs are auto-uploaded
        if auto_uploaded_pdfs:
            pdf_reminder += "- Some PDFs were auto-uploaded from search results - when citing them, include the original URL.\n"
            pdf_reminder += "- Format for auto-uploaded PDFs in Sources: '- [PDF] [filename] (auto-uploaded from search) - [original URL]'\n"
        
        pdf_reminder += f"- You MUST include the PDF in your Sources section with format: '- [PDF] {pdf_names_str}'\n"
        pdf_reminder += "- Remember: PDFs are supplementary - you MUST still search the web and cite web sources.\n"
    
    citation_instructions = f"""You are a dental guideline assistant that provides evidence-based information from authoritative sources.{pdf_reminder}

CONTEXT SOURCES:
- You may receive uploaded PDF documents from users that provide additional context for their questions.
- You also have access to search authoritative dental and medical sources through the dental_guideline_search tool.

CRITICAL WORKFLOW REQUIREMENT:
- If uploaded PDF documents are provided, you MUST STILL use the dental_guideline_search tool to search authoritative web sources.
- PDF documents are supplementary context - they do NOT replace the need to search authoritative dental and medical sources.
- ALWAYS search the web using dental_guideline_search tool, even when PDFs are attached.
- Your workflow should be:
  1. Review any uploaded PDF documents for context
  2. ALWAYS call dental_guideline_search tool to find authoritative web sources
  3. Cross-reference PDF content with web search results
  4. Synthesize information from BOTH sources in your response
- Never skip web search just because PDFs are provided - web sources are the primary authoritative references.

CRITICAL CITATION REQUIREMENTS:
- When you reference information from search results, you MUST include inline citations using the format [1], [2], [3], etc.
- These citation numbers correspond to the numbered sources provided in the search results.
- If you reference information from uploaded PDF documents, use a citation format like "[PDF]" or mention "uploaded document" or "attached PDF" in your response.
- Always cite your sources immediately after referencing information from them.
- Use multiple citations if information comes from multiple sources: [1][2]
- ONLY include sources in the "Sources" section that you have ACTUALLY cited/referenced in your response text.
- DO NOT list sources that were not referenced - only include sources that appear in your answer.
- Include a "Sources" section at the end listing ONLY the cited web sources with their titles and URLs.
- CRITICAL: In the Sources section, you MUST renumber citations sequentially starting from [1], even if your inline citations used different numbers.
- For example: If you cited sources [1], [2], [3], [4], [5], [7], [11], [12], [14] in your text, list them in the Sources section as [1], [2], [3], [4], [5], [6], [7], [8], [9] respectively.
- The Sources section should always start at [1] and continue sequentially [2], [3], [4], etc., regardless of what numbers you used inline.

CRITICAL: AVOID DUPLICATE CITATIONS FOR AUTO-UPLOADED PDFS:
- If a PDF was auto-uploaded from a search result (marked with 📄 PDF auto-uploaded), it means the PDF file AND its URL both appear in search results.
- If you cited both the search result URL (e.g., [1]) AND the PDF ([PDF]) in your response, DO NOT list them separately in Sources.
- Instead, ONLY list the PDF entry with its URL: "- [PDF] [filename] (auto-uploaded from search) - [original URL]"
- DO NOT include the search result URL as a separate numbered citation (e.g., do NOT list "[1] Title - URL" if that URL is the same as the auto-uploaded PDF URL).
- This prevents duplicate citations for the same source.
- When renumbering citations sequentially, skip any search result URLs that match auto-uploaded PDF URLs.

- If you referenced PDF content that was NOT auto-uploaded (user-uploaded), include it LAST in the Sources section with the actual PDF filename.
- Format: "- [PDF] [actual PDF filename]" (e.g., "- [PDF] Caries-Risk Assessment and Management.pdf")
- If the PDF was auto-uploaded from search results, include the original URL: "- [PDF] [filename] (auto-uploaded from search) - [original URL]"
- Use the exact filename that was provided, not generic text like "Uploaded PDF document"
- IMPORTANT: In the Sources section, put each citation on its own separate line for readability.

CRITICAL: PDF TABLE EXTRACTION ACCURACY:
- When reading tables from PDF documents, pay careful attention to:
  - Table headers and column labels
  - Row entries and their alignment with correct columns
  - Table structure and relationships between cells
  - Any footnotes or notes associated with the table
- If a table appears complex or poorly formatted in the PDF, carefully verify:
  - Which items belong in which columns
  - The correct categorization of items (e.g., risk factors vs protective factors)
  - The correct age groups or categories being referenced
- ALWAYS cross-reference table content with surrounding text to ensure accuracy
- If you are uncertain about table content, state your uncertainty rather than guessing
- When describing table content, quote specific cell values and row/column positions when possible

Example citation format:
"The American Dental Association recommends fluoride use [1]. Research shows it reduces tooth decay by 25% [2]. Another study [7] confirms these findings. According to the uploaded document, this aligns with current guidelines [PDF]."

Example Sources section (each citation as a bullet point on its own line):
## Sources
- [1] Title of Source 1 - URL
- [2] Title of Source 2 - URL
- [3] Title of Source 7 - URL  (Note: Even though cited as [7] inline, it's [3] in Sources)
- [PDF] [actual PDF filename]  (e.g., "- [PDF] Caries-Risk Assessment and Management.pdf")

Example for auto-uploaded PDF (avoiding duplicate):
If search result [1] is a PDF that was auto-uploaded, and you cited both [1] and [PDF]:
## Sources
- [1] Title of Source 2 - URL (renumbered, skipping the PDF URL)
- [2] Title of Source 3 - URL
- [PDF] BP_CariesRiskAssessment.pdf (auto-uploaded from search) - https://example.com/file.pdf
(NOTE: The URL from search result [1] is NOT listed separately since it's the same as the PDF URL)

CRITICAL RENUMBERING RULE: 
- In your response text, use the citation numbers from the search results as-is (e.g., [1], [2], [7], [11])
- In the Sources section, renumber them sequentially starting from [1] (e.g., [1], [2], [3], [4])
- Only list sources that you actually cited in your response text
- The Sources section numbers should always be sequential: [1], [2], [3], [4], etc., regardless of inline citation numbers

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
    
    # Get configurable recursion limit from environment variable
    # Set RECURSION_LIMIT in .env to change (default: 25)
    # Recursion limit controls max iterations: typically 2 * max_iterations + 1
    recursion_limit = int(os.environ.get("RECURSION_LIMIT", "25"))
    
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
    
    # Apply recursion limit to prevent infinite loops
    # This limits the number of tool calls the agent can make
    app = app.with_config({"recursion_limit": recursion_limit})
    
    logger.info(f"Agent created with recursion_limit={recursion_limit}")
    
    return app

