import re
from dotenv import load_dotenv

# --- New LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

try:
    from search import search_the_web
    from web_scraper import scrape_web_page
    from image_gen import generate_image
    from rag import search_my_documents
except ImportError:
    print("Error: Could not import 'search_the_web'.")
    print("Make sure 'components/search.py' is a LangChain tool (@tool).")
    exit()

# --- Load API Keys ---
load_dotenv()

# --- 1. Set up the Model (LLM) ---
# We'll use Mixtral, which is stable for tool use.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0
)

# --- 2. Define the Tools ---
# We just put our imported search_the_web function in a list.
tools = [search_the_web, scrape_web_page, generate_image, search_my_documents]
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(
    content=(
        "You are a helpful and concise search assistant.\n"
        "You have access to the following tools:\n"
        "1. search_the_web - use this to find information or URLs.\n"
        "2. scrape_web_page - use this to fetch webpage content.\n"
        "3. generate_image - use this to create images.\n"
        "4. search_my_documents - ALWAYS use this tool first when the user mentions "
        " 'uploaded file', 'my document', 'the PDF', a topic present in their files, "
        " or any question that could be answered from their uploaded documents.\n\n"
        "Rules:\n"
        "- If user asks anything related to uploaded PDFs/documents, call search_my_documents.\n"
        "- If user asks for general knowledge, use search_the_web.\n"
        "- If user asks for page content, use scrape_web_page (but only after getting URL).\n"
        "- If user asks to draw/create something, call generate_image.\n"
        "- NEVER reply with 'I cannot find uploaded file' unless search_my_documents returns empty.\n"
        "- When in doubt, PREFER calling search_my_documents.\n"
    )
)

def agent_node(state: MessagesState):
    """
    The 'Brain' node. It takes the current state (messages), 
    appends the system prompt if needed, and calls the LLM.
    """
    messages = state["messages"]
    
    # We prepend the system message if it's not already there
    # (Simple check to ensure we don't stack system messages)
    if len(messages) == 0 or not isinstance(messages[0], SystemMessage):
        messages = [sys_msg] + messages

    user_text = messages[-1].content.lower()

    document_keywords = [
        "pdf", "document", "file", "my file", "my pdf",
        "uploaded", "knowledge base", "kb", "rag", "from my files"
    ]

    if any(k in user_text for k in document_keywords):
        # Inject a hint BEFORE the model decides tool usage
        messages.append(SystemMessage(content="Use search_my_documents now."))        
    # Call the model
    response = llm_with_tools.invoke(messages)
    
    # Return the update to the state (LangGraph automagically appends this)
    return {"messages": [response]}

tool_node = ToolNode(tools, handle_tool_errors=True)

builder = StateGraph(MessagesState)

# Add Nodes
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

# Add Edges
# Start -> Agent
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)

# Tools -> Agent
# After a tool runs, we always go back to the agent to "read" the result.
builder.add_edge("tools", "agent")

memory_conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(memory_conn)
react_graph = builder.compile(checkpointer=memory)

def get_all_thread_ids():
    """Returns a list of all conversation IDs from the database."""
    try:
        cursor = memory_conn.cursor()
        # Query the checkpoints table for distinct thread_ids
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"Error fetching threads: {e}")
        return []

def get_thread_history(thread_id):
    """Fetches the message history for a specific thread."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = react_graph.get_state(config)
        if not snapshot.values:
            return []
        
        messages = snapshot.values.get("messages", [])
        formatted_msgs = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_msgs.append({"role": "user", "content": msg.content})
            
            elif isinstance(msg, AIMessage):
                # Filter out empty tool calls (which have no content)
                if msg.content and isinstance(msg.content, str) and msg.content.strip():
                    content = msg.content
                    img_path = None
                    
                    # Robust Image Regex Check
                    try:
                        # Look for .jpg or .png in the response text
                        match = re.search(r"(generated_image_\d+\.(jpg|png))", content)
                        if match: 
                            img_path = match.group(1)
                    except Exception as re_error:
                        print(f"Regex error parsing image: {re_error}")

                    formatted_msgs.append({
                        "role": "assistant", 
                        "content": content,
                        "image": img_path
                    })
                    
        return formatted_msgs
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []

def run_llm_agent(user_query: str, thread_id: str = "session_1") -> str:
    """
    Runs the LangGraph agent.
    
    Args:
        user_query: The text input from the user.
        thread_id: A unique ID for the conversation session.
    """
    print(f"\n--- 🚀 [LangGraph Agent] New Query: '{user_query}' ---")
    
    # Config for the memory (keeps track of this specific conversation thread)
    config = {"configurable": {"thread_id": thread_id}}
    
    # Invoke the graph
    # We pass the new user message into the state
    input_message = HumanMessage(content=user_query)
    
    try:
        # Stream or Invoke? Invoke is simpler for now.
        # The graph handles the loop (Agent -> Tool -> Agent -> Final Answer) internally.
        final_state = react_graph.invoke({"messages": [input_message]}, config=config)
        
        # Extract the last message (the AI's final response)
        last_message = final_state["messages"][-1]
        content = last_message.content
        
        print(f"--- 🚀 [LangGraph Agent] Final Answer: {content[:100]}... ---")
        return content

    except Exception as e:
        print(f"An error occurred in the LangGraph agent: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    print("Testing LangGraph Agent...")
    
    # Test 1: Simple memory test
    print("\n--- Test 1: Context ---")
    print(run_llm_agent("My name is John."))
    print(run_llm_agent("What is my name?")) # Should remember "John"
    
    # Test 2: Tool use
    print("\n--- Test 2: Tool Use ---")
    print(run_llm_agent("What is the weather in Paris?"))




    