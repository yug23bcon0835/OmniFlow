import os
import io
from groq import Groq
from dotenv import load_dotenv

# --- Load API Keys ---
load_dotenv()

try:
    # Initialize the Groq client
    client = Groq()
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    print("Please make sure your .env file is set up with GROQ_API_KEY.")
    exit()

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribes audio bytes using the Groq Whisper API.

    Args:
        audio_bytes: The raw audio data as bytes.

    Returns:
        The transcribed text as a string.
    """
    print("--- 🗣️ Transcribing audio... ---")

    try:
        # We must wrap the bytes in a file-like object (BytesIO)
        # and provide a filename for the API.
        audio_file = ("mic_audio.wav", audio_bytes)

        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="text"
        )

        print(f"--- 🗣️ Transcription: '{transcription}' ---")
        return transcription

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return f"Error: {e}"

# --- This part is for testing ---
# We can run this file directly to test it.

if __name__ == "__main__":
    print("Testing Transcriber Module...")

    # We'll test this by reading the sample audio file
    # you downloaded earlier.

    # IMPORTANT: Make sure 'sample_audio.mp3' is in the
    # *root* folder (voice-search-agent), not in 'modules'.

    sample_file_path = "Recording (2).mp3" # Or whatever you named it

    try:
        with open(sample_file_path, "rb") as audio_file:
            sample_bytes = audio_file.read()

        print(f"Loaded sample file: {sample_file_path}")

        # Run the transcription
        transcribed_text = transcribe_audio(sample_bytes)

        print("\n--- Test Result ---")
        print(transcribed_text)

    except FileNotFoundError:
        print(f"\n--- Test Failed ---")
        print(f"Error: Could not find the test file '{sample_file_path}'.")
        print("Please make sure you have a sample audio file in your main project folder.")

    except Exception as e:
        print(f"An error occurred during the test: {e}")