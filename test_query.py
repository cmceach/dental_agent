#!/usr/bin/env python3
"""
Test script for the dental agent query functionality.
Run this to test a query without starting the Streamlit app.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Load environment variables
load_dotenv()

# Import agent creation function
from agent import create_agent

def test_query(query: str):
    """Test a query with the dental agent."""
    print(f"\n{'='*60}")
    print(f"Testing query: {query}")
    print(f"{'='*60}\n")
    
    try:
        # Create the agent
        print("Initializing agent...")
        agent = create_agent()
        print("✓ Agent initialized successfully\n")
        
        # Prepare input
        messages = [HumanMessage(content=query)]
        graph_input = {"messages": messages}
        
        print("Executing query...")
        print("-" * 60)
        
        # Stream the response
        final_response = None
        all_messages = []
        
        for step, s in enumerate(agent.stream(graph_input, stream_mode="values"), 1):
            last_message = s["messages"][-1]
            all_messages.append(last_message)
            
            # Print message type and content preview
            msg_type = type(last_message).__name__
            
            print(f"\n[Step {step}] {msg_type}:")
            
            # Handle different message types
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                print(f"  Tool call: {last_message.tool_calls[0]['name']}")
                print(f"  Args: {last_message.tool_calls[0]['args']}")
            elif isinstance(last_message, HumanMessage):
                print(f"  Content: {last_message.content[:100]}...")
            elif isinstance(last_message, ToolMessage):
                print(f"  Tool result preview: {str(last_message.content)[:150]}...")
            elif hasattr(last_message, 'content'):
                # This is likely the final AIMessage
                content_preview = str(last_message.content)[:150]
                print(f"  Content preview: {content_preview}...")
                # Check if this is a final response (no tool calls)
                if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                    final_response = last_message.content
        
        print("\n" + "-" * 60)
        print("\nFINAL RESPONSE:")
        print("=" * 60)
        
        # Try to find the final AIMessage response
        if not final_response:
            # Look for the last AIMessage that doesn't have tool calls
            for msg in reversed(all_messages):
                if hasattr(msg, 'content') and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                    final_response = msg.content
                    break
        
        if final_response:
            # Extract text if it's structured
            if isinstance(final_response, list):
                for item in final_response:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        print(item.get('text', ''))
                    elif isinstance(item, str):
                        print(item)
            elif isinstance(final_response, str):
                print(final_response)
            else:
                print(str(final_response))
        else:
            print("No final response captured")
            print("\nAll messages:")
            for i, msg in enumerate(all_messages, 1):
                print(f"  {i}. {type(msg).__name__}: {str(msg)[:100]}...")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Test query (fixing typo in user's query)
    query_text = "is fluoride safe to recommend?"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    
    test_query(query_text)

