import os
from serpapi import GoogleSearch
from dotenv import load_dotenv
from langchain_core.tools import tool

# Load API keys
load_dotenv()

@tool
def search_the_web(query: str) -> str:
    """
    Performs a Google search using SerpAPI and returns a string 
    of the most relevant results. Use this for questions about 
    current events, weather, or specific facts.
    """
    print(f"--- 🔎 Searching for: '{query}' ---")

    serpapi_key = os.environ.get("SERPAPI_API_KEY")
    if not serpapi_key:
        return "Error: SERPAPI_API_KEY not set."

    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_key,
            "num": 5 # Request top 5 results
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        # --- Process the results ---
        # We will extract and simplify the results to send to the LLM.

        output_parts = []

        # 1. Check for a direct answer box (e.g., "What is 2+2?")
        if "answer_box" in results:
            answer = results["answer_box"].get("result")
            if answer:
                output_parts.append(f"Direct Answer: {answer}")

        # 2. Check for snippets from top organic results
        if "organic_results" in results:
            for result in results["organic_results"][:3]: # Look at top 3
                title = result.get("title", "No Title")
                snippet = result.get("snippet", "No Snippet")
                output_parts.append(f"Title: {title}\nSnippet: {snippet}\n---")

        if not output_parts:
            return "No relevant search results found."

        print("--- 🔎 Search complete. ---")
        return "\n".join(output_parts)

    except Exception as e:
        print(f"--- 🔎 Search error: {e} ---")
        return f"Error during search: {e}"

# --- This part is for testing ---
# We can run this file directly to test it.
if __name__ == "__main__":
    print("Testing Search Module...")

    # Test 1: A general query
    results1 = search_the_web("What is the weather in New York?")
    print("\n--- Test 1 Results ---")
    print(results1)

    # Test 2: A knowledge query
    results2 = search_the_web("Who is the CEO of Groq?")
    print("\n--- Test 2 Results ---")
    print(results2)