import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent import create_agent

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
    st.session_state["messages"] = []
    st.session_state["tool_calls_history"] = []
    st.rerun()


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
        # Handle list of content blocks (e.g., [{'type': 'text', 'text': '...'}])
        text_parts = []
        for item in message_content:
            if isinstance(item, dict):
                if item.get('type') == 'text' and 'text' in item:
                    text_parts.append(item['text'])
                elif 'text' in item:
                    text_parts.append(str(item['text']))
            elif isinstance(item, str):
                text_parts.append(item)
        return '\n'.join(text_parts)
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
    # Add user message to history and display
    user_message = HumanMessage(content=prompt)
    st.session_state["messages"].append(user_message)
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
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
                last_message = s["messages"][-1]
                
                # Handle tool calls
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    tool_name = last_message.tool_calls[0]['name']
                    tool_args = last_message.tool_calls[0]['args']
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
                
                # Handle tool output
                elif isinstance(last_message, ToolMessage):
                    tool_result = last_message.content
                    
                    # Update the last tool call with result preview
                    if current_tool_calls:
                        current_tool_calls[-1]['result_preview'] = tool_result
                    
                    if tool_expander:
                        with tool_expander:
                            st.success("✅ Tool executed successfully")
                            st.text(f"📄 Result preview: {tool_result[:300]}...")
                
                # Handle final AI response
                elif isinstance(last_message, AIMessage) and not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                    # Extract text content properly, handling structured formats
                    raw_content = last_message.content
                    full_response = extract_text_content(raw_content)
                    # Update display with streaming effect
                    response_placeholder.markdown(full_response)
            
            # Display final response if we have one
            if full_response:
                response_placeholder.markdown(full_response)
                
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

