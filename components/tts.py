import asyncio
import edge_tts
import os

# A good, fast, natural-sounding voice
VOICE = "en-US-GuyNeural" 

async def _generate_speech_async(text: str, file_path: str):
    """
    Internal async function to generate and save speech.
    """
    print(f"--- 🔊 Generating speech for: '{text[:30]}...' ---")
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        # Use the file_path variable passed to the function
        await communicate.save(file_path)
        print(f"--- 🔊 Audio saved to: {file_path} ---")
        
    except Exception as e:
        print(f"An error occurred during speech generation: {e}")

def generate_speech(text_to_speak: str, output_file_path: str) -> str:
    """
    Synchronous wrapper for the async speech generation.
    Takes text and saves it to the specified output_file_path.
    
    Returns:
        The path to the saved audio file.
    """
    try:
        # Pass the unique output_file_path to the async function
        asyncio.run(_generate_speech_async(text_to_speak, output_file_path))
        return output_file_path
    except Exception as e:
        if "cannot run current event loop" in str(e):
            print("Error: Asyncio loop already running.")
            return f"Error: Could not run asyncio: {e}"
        else:
            print(f"An error occurred: {e}")
            return f"Error: {e}"

# --- This part is for testing ---
if __name__ == "__main__":
    print("Testing TTS Module...")
    
    test_text = "Hello, this is a test of the Edge TTS text-to-speech generation."
    
    # --- UPDATED TEST ---
    # We now must provide a unique filename for the test
    test_output_file = "test_audio_output.mp3"
    output_path = generate_speech(test_text, test_output_file)
    
    if "Error" not in output_path:
        print(f"\n--- Test Successful ---")
        print(f"Audio file generated at: {output_path}")
        if os.path.exists(test_output_file):
            os.remove(test_output_file) # Clean up the test file
            print("Cleaned up test file.")
    else:
        print(f"\n--- Test Failed ---")
        print(output_path)