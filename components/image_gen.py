import os
import requests
import base64
import time
from dotenv import load_dotenv
from langchain_core.tools import tool

# Load API keys
load_dotenv()

# Cloudflare Workers AI Configuration
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")

if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
    print("WARNING: CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID not found in .env.")
    print("Image generation will not work without these credentials.")

# This is the model recommended for text-to-image on Workers AI
IMAGE_MODEL_NAME = "@cf/lykon/dreamshaper-8-lcm"
# IMAGE_MODEL_NAME = "@cf/runwayml/stable-diffusion-v1-5" # Alternative, slower model

@tool
def generate_image(prompt: str) -> str:
    """
    Generates an image from a text description using Cloudflare Workers AI.
    Use this tool when the user explicitly asks to "create an image", "draw something",
    "generate a picture", or any similar request where a visual output is desired.
    
    The prompt should be a clear and concise description of the desired image.
    """
    print(f"--- 🖼️ [Image Gen Tool] Generating image for: '{prompt}' ---")
    
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        return "Error: Cloudflare API credentials not configured. Cannot generate image."

    api_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{IMAGE_MODEL_NAME}"
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "num_steps": 20 # Lower for faster generation, higher for better quality
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        # Cloudflare Workers AI returns a JPEG byte stream directly
        image_bytes = response.content
        
        # Save the image to a unique file
        output_filename = f"generated_image_{int(time.time())}.jpg"
        with open(output_filename, "wb") as f:
            f.write(image_bytes)
        
        print(f"--- 🖼️ [Image Gen Tool] Image saved to: {output_filename} ---")
        return f"The image has been generated and saved as: {output_filename}"

    except requests.exceptions.RequestException as e:
        print(f"Error calling Cloudflare Workers AI: {e}")
        # Attempt to parse error message from response if available
        try:
            error_data = response.json()
            return f"Error: Failed to generate image. Cloudflare API said: {error_data.get('errors', [{}])[0].get('message', 'Unknown error.')}"
        except:
            return f"Error: Failed to generate image. Network or API issue: {e}"
    except Exception as e:
        return f"An unexpected error occurred during image generation: {e}"

# --- This part is for testing ---
if __name__ == "__main__":
    print("Testing Image Generation Module...")
    
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        print("\n--- Skipping test: Cloudflare credentials not configured. ---")
        print("Please set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID in your .env file.")
    else:
        test_prompt = "A futuristic city at sunset, highly detailed, cyberpunk style"
        print(f"\n--- Generating image for: '{test_prompt}' ---")
        
        # --- Using .invoke() because it's a LangChain tool ---
        result = generate_image.invoke(test_prompt)
        print(f"\nResult:\n{result}")
        
        if "Image generated successfully" in result:
            image_path = result.split(": ")[1]
            if os.path.exists(image_path):
                print(f"Image successfully created at {image_path}. You can view it.")
                # You might want to delete it after checking, or manually.
                # os.remove(image_path)
            else:
                print("Image file not found despite success message.")
        else:
            print("Image generation failed.")