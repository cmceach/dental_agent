import os
import time
import tempfile
import json
import re
import logging
from typing import List, Dict
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import File

from agent import create_agent, get_pdf_text_from_gemini

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🦷 Dental Guideline Agent",
    page_icon="🦷",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "agent" not in st.session_state:
    try:
        st.session_state["agent"] = create_agent()
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.stop()

if "tool_calls_history" not in st.session_state:
    st.session_state["tool_calls_history"] = []

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []  # Store uploaded file references

if "auto_uploaded_pdfs" not in st.session_state:
    st.session_state["auto_uploaded_pdfs"] = []  # Store auto-uploaded PDFs from search results


def validate_api_keys():
    """Validate that required API keys are present."""
    missing_keys = []
    if not os.environ.get("GOOGLE_API_KEY"):
        missing_keys.append("GOOGLE_API_KEY")
    if not os.environ.get("EXA_API_KEY"):
        missing_keys.append("EXA_API_KEY")
    
    if missing_keys:
        st.error(f"❌ Missing required environment variables: {', '.join(missing_keys)}")
        st.info("💡 Please create a `.env` file with your API keys. See `.env.example` for reference.")
        return False
    return True


def reset_conversation():
    """Reset the conversation history."""
    # Clean up uploaded files from Gemini
    try:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        
        # Clean up user-uploaded files
        if st.session_state.get("uploaded_files"):
            for file_info in st.session_state["uploaded_files"]:
                try:
                    genai.delete_file(file_info['gemini_file'].name)
                except:
                    pass
        
        # Clean up auto-uploaded PDFs (only delete if uploaded in this session, not reused)
        if st.session_state.get("auto_uploaded_pdfs"):
            for pdf_info in st.session_state["auto_uploaded_pdfs"]:
                # Only delete if we uploaded it (not reused from previous session)
                if not pdf_info.get("was_reused", False):
                    try:
                        if pdf_info.get("gemini_file") and hasattr(pdf_info["gemini_file"], 'name'):
                            genai.delete_file(pdf_info['gemini_file'].name)
                    except:
                        pass
    except:
        pass
    
    st.session_state["messages"] = []
    st.session_state["tool_calls_history"] = []
    st.session_state["uploaded_files"] = []
    st.session_state["auto_uploaded_pdfs"] = []
    st.rerun()


def extract_auto_uploaded_pdfs(tool_result: str) -> List[Dict]:
    """
    Extract auto-uploaded PDF information from tool result string.
    
    Args:
        tool_result: Tool result string that may contain <AUTO_UPLOADED_PDFS>JSON</AUTO_UPLOADED_PDFS>
        
    Returns:
        List of PDF info dictionaries, empty list if none found
    """
    try:
        # Look for encoded PDF info
        pattern = r'<AUTO_UPLOADED_PDFS>(.*?)</AUTO_UPLOADED_PDFS>'
        match = re.search(pattern, tool_result, re.DOTALL)
        if match:
            pdf_json = match.group(1)
            pdf_info_list = json.loads(pdf_json)
            
            # Get gemini_file objects by URI lookup
            try:
                genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
                files = list(genai.list_files())
                
                for pdf_info in pdf_info_list:
                    uri = pdf_info.get("uri")
                    if uri:
                        # Find matching file by URI
                        for file in files:
                            if hasattr(file, 'uri') and file.uri == uri:
                                pdf_info["gemini_file"] = file
                                break
            except Exception as e:
                logger.debug(f"Could not retrieve Gemini file objects: {e}")
            
            return pdf_info_list
    except Exception as e:
        logger.debug(f"Error extracting auto-uploaded PDFs: {e}")
    
    return []


def renumber_inline_citations(text: str) -> str:
    """
    Post-process response text to replace inline citations with sequential numbers.
    
    Extracts citations from Sources section and creates a mapping, then replaces
    inline citations to match the Sources numbering. Also removes the 
    "(from search result [X])" annotations from the Sources section.
    
    Args:
        text: The full response text including Sources section
        
    Returns:
        str: Text with renumbered inline citations and cleaned Sources
    """
    import re
    
    # Extract Sources section
    sources_match = re.search(r'## Sources\s*\n(.*?)(?:\n\n|$)', text, re.DOTALL)
    if not sources_match:
        return text
    
    sources_text = sources_match.group(1)
    
    # Extract mapping from Sources: [1] ... (from search result [X])
    # Pattern: [sequential_num] ... (from search result [original_num])
    mapping = {}  # original_num -> sequential_num
    for line in sources_text.split('\n'):
        match = re.search(r'- \[(\d+)\].*\(from search result \[(\d+)\]\)', line)
        if match:
            sequential_num = match.group(1)
            original_num = match.group(2)
            mapping[original_num] = sequential_num
    
    # If no mapping found, still clean the sources section
    if not mapping:
        # Just remove "(from search result [X])" annotations
        sources_text_cleaned = re.sub(r'\s*\(from search result \[\d+\]\)', '', sources_text)
        sources_start = text.find('## Sources')
        if sources_start != -1:
            text_before_sources = text[:sources_start]
            return text_before_sources + '## Sources\n' + sources_text_cleaned
        return text
    
    # Split text into before Sources and Sources sections
    sources_start = text.find('## Sources')
    if sources_start == -1:
        return text
    
    text_before_sources = text[:sources_start]
    sources_section = text[sources_start:]
    
    # Replace inline citations in the text before Sources
    # Sort by original number (descending) to avoid issues with overlapping replacements
    for original_num in sorted(mapping.keys(), key=int, reverse=True):
        sequential_num = mapping[original_num]
        if original_num != sequential_num:
            # Replace [original_num] with [sequential_num] in text before Sources
            text_before_sources = re.sub(
                r'\[' + original_num + r'\]',
                f'[{sequential_num}]',
                text_before_sources
            )
    
    # Remove "(from search result [X])" annotations from Sources section
    sources_section_cleaned = re.sub(r'\s*\(from search result \[\d+\]\)', '', sources_section)
    
    return text_before_sources + sources_section_cleaned


def extract_text_content(message_content):
    """
    Extract plain text from message content, handling both string and structured formats.
    
    Args:
        message_content: Content from a message, can be str, list, or dict
        
    Returns:
        str: Plain text content
    """
    if isinstance(message_content, str):
        return message_content
    elif isinstance(message_content, list):
        # Handle list of content blocks (e.g., [{'type': 'text', 'text': '...'}, {'type': 'media', ...}])
        text_parts = []
        file_info = []
        for item in message_content:
            if isinstance(item, dict):
                if item.get('type') == 'text' and 'text' in item:
                    text_parts.append(item['text'])
                elif item.get('type') == 'media':
                    # Extract file name if available, otherwise use URI
                    file_uri = item.get('file_uri', '')
                    file_info.append(f"PDF document")
                elif 'text' in item:
                    text_parts.append(str(item['text']))
            elif isinstance(item, str):
                text_parts.append(item)
        
        result = '\n'.join(text_parts)
        if file_info:
            result += f"\n\n📎 Attached: {', '.join(file_info)}"
        return result
    elif isinstance(message_content, dict):
        # Handle dict format
        if 'text' in message_content:
            return str(message_content['text'])
        else:
            return str(message_content)
    else:
        return str(message_content)


# Main UI
st.title("🦷 Dental Guideline ReAct Agent")
st.markdown("Ask questions about dental procedures, clinical guidelines, or official recommendations from authoritative sources.")

# Validate API keys
if not validate_api_keys():
    st.stop()

# Sidebar with controls
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        reset_conversation()
    
    st.markdown("---")
    st.markdown("### 📄 Upload PDF Document")
    st.markdown("Upload a PDF document to provide additional context for the agent.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document that the agent can reference when answering questions"
    )
    
    if uploaded_file is not None:
        # Check if file is already uploaded
        file_name = uploaded_file.name
        file_already_uploaded = any(f.get('name') == file_name for f in st.session_state["uploaded_files"])
        
        if not file_already_uploaded:
            with st.spinner(f"Uploading {file_name} to Gemini..."):
                try:
                    # Initialize Gemini client for file upload
                    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
                    
                    # Read file content and upload to Gemini File API
                    # Save to temporary file first (required by genai.upload_file)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    # Upload to Gemini File API with display name
                    gemini_file = genai.upload_file(
                        path=tmp_path,
                        mime_type="application/pdf",
                        display_name=file_name
                    )
                    
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    
                    # Wait for file to be processed
                    while gemini_file.state.name != "ACTIVE":
                        time.sleep(1)
                        gemini_file = genai.get_file(gemini_file.name)
                    
                    # Store file reference
                    st.session_state["uploaded_files"].append({
                        "name": file_name,
                        "gemini_file": gemini_file,
                        "uri": gemini_file.uri
                    })
                    
                    st.success(f"✅ {file_name} uploaded successfully!")
                    st.info(f"📄 The agent can now reference this document when answering questions.")
                    
                except Exception as e:
                    st.error(f"❌ Error uploading file: {str(e)}")
        else:
            st.info(f"📄 {file_name} is already uploaded and available.")
    
    # Show uploaded files
    if st.session_state["uploaded_files"]:
        with st.expander("📚 Uploaded Documents", expanded=False):
            for i, file_info in enumerate(st.session_state["uploaded_files"], 1):
                st.markdown(f"**{i}. {file_info['name']}**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Remove", key=f"remove_{i}", use_container_width=True):
                        # Remove file from Gemini
                        try:
                            genai.delete_file(file_info['gemini_file'].name)
                        except:
                            pass
                        st.session_state["uploaded_files"].pop(i-1)
                        st.rerun()
                with col2:
                    if st.button(f"Debug Text", key=f"debug_{i}", use_container_width=True):
                        # Show extracted text for debugging
                        try:
                            text = get_pdf_text_from_gemini(file_info['gemini_file'].uri)
                            if text:
                                with st.expander(f"Extracted Text from {file_info['name']}", expanded=True):
                                    st.text_area("PDF Text Content", text, height=400, key=f"debug_text_{i}")
                            else:
                                st.warning("Could not extract text from PDF")
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    # Show auto-uploaded PDFs with debug option
    if st.session_state.get("auto_uploaded_pdfs"):
        with st.expander("📄 Auto-Uploaded PDFs (from search)", expanded=False):
            for i, pdf_info in enumerate(st.session_state["auto_uploaded_pdfs"], 1):
                filename = pdf_info.get("filename", f"PDF {i}")
                st.markdown(f"**{i}. {filename}**")
                if pdf_info.get("gemini_file") and hasattr(pdf_info["gemini_file"], 'uri'):
                    if st.button(f"Debug Text", key=f"debug_auto_{i}", use_container_width=True):
                        try:
                            text = get_pdf_text_from_gemini(pdf_info["gemini_file"].uri)
                            if text:
                                with st.expander(f"Extracted Text from {filename}", expanded=True):
                                    st.text_area("PDF Text Content", text, height=400, key=f"debug_auto_text_{i}")
                            else:
                                st.warning("Could not extract text from PDF")
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    This agent searches authoritative dental and medical sources:
    
    **Professional Dental Associations:**
    - ADA (American Dental Association)
    - AAPD (American Academy of Pediatric Dentistry)
    - AAE (American Association of Endodontists)
    - AAOP (American Academy of Periodontology)
    - AAOMS (American Association of Oral and Maxillofacial Surgeons)
    - CDC Division of Oral Health
    
    **Medical Associations:**
    - AAP (American Academy of Pediatrics)
    - AAP Publications
    
    **Government Regulatory Agencies:**
    - FDA (Food and Drug Administration)
    
    **Evidence-Based Research Sources:**
    - PubMed (biomedical literature)
    - NIDCR (National Institute of Dental and Craniofacial Research)
    - NIH (National Institutes of Health)
    - Cochrane Library (systematic reviews)
    - WHO (World Health Organization)
    - IADR (International Association for Dental Research)
    - JADA (Journal of the American Dental Association)
    """)
    
    # Show tool calls history if available
    if st.session_state["tool_calls_history"]:
        with st.expander("🔍 Recent Tool Calls", expanded=False):
            for i, tool_call in enumerate(reversed(st.session_state["tool_calls_history"][-5:]), 1):
                st.markdown(f"**{i}. {tool_call.get('tool_name', 'Unknown')}**")
                st.text(f"Query: {tool_call.get('query', 'N/A')}")
                if tool_call.get('result_preview'):
                    st.text(f"Result: {tool_call['result_preview'][:200]}...")
                st.markdown("---")

# Display chat history
for message in st.session_state["messages"]:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        content = extract_text_content(message.content)
        st.markdown(content)

# Handle user input
if prompt := st.chat_input("Ask a question about dental guidelines..."):
    # Build message content - include text and any uploaded PDF files (user + auto-uploaded)
    file_parts = []
    pdf_filenames = []
    auto_uploaded_names = []
    
    # Add user-uploaded PDFs
    for file_info in st.session_state["uploaded_files"]:
        file_part = {
            "type": "media",
            "mime_type": "application/pdf",
            "file_uri": file_info["gemini_file"].uri
        }
        file_part["filename"] = file_info["name"]
        file_parts.append(file_part)
        pdf_filenames.append(file_info["name"])
    
    # Add auto-uploaded PDFs from search results
    for pdf_info in st.session_state["auto_uploaded_pdfs"]:
        if pdf_info.get("gemini_file") and hasattr(pdf_info["gemini_file"], 'uri'):
            file_part = {
                "type": "media",
                "mime_type": "application/pdf",
                "file_uri": pdf_info["gemini_file"].uri
            }
            file_part["filename"] = pdf_info.get("filename", "search_result.pdf")
            file_part["is_auto_uploaded"] = True
            file_part["source_url"] = pdf_info.get("url", "")
            file_parts.append(file_part)
            auto_uploaded_names.append(pdf_info.get("filename", "search_result.pdf"))
    
    # Build prompt text with PDF info
    if pdf_filenames or auto_uploaded_names:
        pdf_text_parts = []
        if pdf_filenames:
            pdf_text_parts.append(f"User-uploaded PDF(s): {', '.join(pdf_filenames)}")
        if auto_uploaded_names:
            pdf_text_parts.append(f"Auto-uploaded PDF(s) from search results: {', '.join(auto_uploaded_names)}")
        
        pdf_names_text = f"\n\nNote: The following PDF document(s) are attached: {'; '.join(pdf_text_parts)}"
        prompt_with_pdf_info = prompt + pdf_names_text
        message_content = [{"type": "text", "text": prompt_with_pdf_info}] + file_parts
    else:
        # Simple text message if no files
        message_content = prompt
    
    # Add user message to history and display
    user_message = HumanMessage(content=message_content)
    st.session_state["messages"].append(user_message)
    
    with st.chat_message("user"):
        st.markdown(prompt)
        if st.session_state["uploaded_files"]:
            file_names = [f["name"] for f in st.session_state["uploaded_files"]]
            st.caption(f"📎 Attached: {', '.join(file_names)}")
    
    # Prepare graph input
    graph_input = {"messages": st.session_state["messages"]}
    
    # Stream the agent's response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        tool_expander = None
        
        # Collect streaming chunks
        full_response = ""
        current_tool_calls = []
        
        try:
            # Stream the agent's steps
            for s in st.session_state["agent"].stream(graph_input, stream_mode="values"):
                # Get the last message added to the state
                if not s.get("messages") or len(s["messages"]) == 0:
                    continue
                    
                last_message = s["messages"][-1]
                
                # Handle tool calls - explicitly check for AIMessage with tool_calls
                # This must be checked FIRST before checking for final AI response
                if isinstance(last_message, AIMessage):
                    # Check if this AIMessage has tool calls
                    has_tool_calls = (
                        hasattr(last_message, 'tool_calls') and 
                        last_message.tool_calls and 
                        len(last_message.tool_calls) > 0
                    )
                    
                    if has_tool_calls:
                        # Extract tool call information
                        tool_call = last_message.tool_calls[0]
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})
                        query = tool_args.get('query', 'N/A')
                        
                        # Store tool call info
                        tool_info = {
                            'tool_name': tool_name,
                            'query': query,
                            'result_preview': None
                        }
                        current_tool_calls.append(tool_info)
                        
                        # Create expander for tool calls if not already created
                        if tool_expander is None:
                            tool_expander = st.expander("🔍 Tool Execution Details", expanded=True)
                        
                        with tool_expander:
                            st.info(f"🤖 **Agent Action**: Calling tool `{tool_name}`")
                            st.text(f"📝 Query: {query}")
                    else:
                        # This is a final AI response (no tool calls)
                        # Extract text content properly, handling structured formats
                        raw_content = last_message.content
                        full_response = extract_text_content(raw_content)
                        # Update display with streaming effect
                        response_placeholder.markdown(full_response)
                
                # Handle tool output
                elif isinstance(last_message, ToolMessage):
                    tool_result = last_message.content
                    
                    # Extract auto-uploaded PDFs from tool result
                    auto_uploaded_pdfs = extract_auto_uploaded_pdfs(tool_result)
                    if auto_uploaded_pdfs:
                        # Store auto-uploaded PDFs in session state
                        for pdf_info in auto_uploaded_pdfs:
                            # Check if already stored (avoid duplicates)
                            url = pdf_info.get("url")
                            if url and not any(p.get("url") == url for p in st.session_state["auto_uploaded_pdfs"]):
                                st.session_state["auto_uploaded_pdfs"].append(pdf_info)
                        
                        if tool_expander:
                            with tool_expander:
                                pdf_count = len(auto_uploaded_pdfs)
                                st.info(f"📄 Auto-uploaded {pdf_count} PDF(s) from search results")
                    
                    # Update the last tool call with result preview
                    if current_tool_calls:
                        current_tool_calls[-1]['result_preview'] = tool_result
                    
                    if tool_expander:
                        with tool_expander:
                            st.success("✅ Tool executed successfully")
                            st.text(f"📄 Result preview: {tool_result[:300]}...")
            
            # Display final response if we have one
            if full_response:
                # Post-process: renumber inline citations to match Sources section
                full_response_display = renumber_inline_citations(full_response)
                response_placeholder.markdown(full_response_display)
                
                # Add to message history (store the original content structure)
                ai_message = AIMessage(content=full_response)
                st.session_state["messages"].append(ai_message)
                
                # Update tool calls history
                if current_tool_calls:
                    st.session_state["tool_calls_history"].extend(current_tool_calls)
            else:
                # Fallback if no final response was captured
                st.warning("Agent completed but no final response was generated.")
        
        except Exception as e:
            error_msg = f"❌ Error occurred: {str(e)}"
            response_placeholder.error(error_msg)
            st.exception(e)

