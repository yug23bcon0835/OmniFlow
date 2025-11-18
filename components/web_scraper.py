import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

@tool
def scrape_web_page(url: str) -> str:
    """
    Scrapes the text content from a given URL.
    Use this tool when you need to get the full content, summary, 
    or specific details from a specific web page link. 
    Only use this if you have a URL.
    """
    print(f"--- 🛠️ [Tool] Scraping URL: {url} ---")

    try:
        # Set a user-agent to pretend to be a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)

        # Check for successful response
        if response.status_code != 200:
            return f"Error: Failed to retrieve page (Status code: {response.status_code})"

        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # --- Extract text ---
        # This is a basic text extraction. It strips out <script>, <style>,
        # and other non-text tags.

        # Remove non-content tags
        for script_or_style in soup(["script", "style", "nav", "footer", "aside"]):
            script_or_style.decompose()

        # Get the text, strip whitespace, and join lines
        text_lines = (line.strip() for line in soup.get_text().splitlines())
        # Re-join lines into a single coherent block of text
        text_content = '\n'.join(line for line in text_lines if line)

        # Truncate to a reasonable length to not overwhelm the LLM context
        max_length = 8000 # ~8000 characters

        if len(text_content) > max_length:
            print(f"--- 🛠️ [Tool] Content truncated (was {len(text_content)} chars) ---")
            return text_content[:max_length] + "... (content truncated)"
        else:
            print(f"--- 🛠️ [Tool] Scrape successful ({len(text_content)} chars) ---")
            return text_content

    except Exception as e:
        return f"Error during scraping: {e}"

# --- This part is for testing ---
if __name__ == "__main__":
    print("Testing Web Scraper Module...")

    # Test 1: A blog post
    test_url_1 = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    print(f"\n--- Scraping: {test_url_1} ---")
    content1 = scrape_web_page.invoke(test_url_1)
    print(content1[:500] + "...") # Print first 500 chars