import subprocess
import webbrowser
from tts import TTS

tts = TTS()

def execute_command(transcript_or_result):
    """
    Handles executing commands based on either:
    - A plain text transcript (string), or
    - A result JSON from Gemini or other LLM.
    """
    if isinstance(transcript_or_result, dict):
        # Handle JSON result from Gemini or another LLM agent
        intent = transcript_or_result.get("intent")
        params = transcript_or_result.get("parameters", {})
        speak_text = transcript_or_result.get("response_text", "")

        if speak_text:
            tts.speak(speak_text)

        if intent == "open_app":
            app = params.get("app_name")
            if app:
                subprocess.run(app, shell=True)
        elif intent == "search_web":
            query = params.get("query")
            if query:
                webbrowser.open(f"https://www.google.com/search?q={query}")
        else:
            tts.speak("Sorry, I don't know how to do that yet.")

    elif isinstance(transcript_or_result, str):
        # Handle raw transcript (fallback)
        # Here you'd implement naive keyword matching
        lower_text = transcript_or_result.lower()

        if "open notepad" in lower_text:
            subprocess.run("notepad", shell=True)
            tts.speak("Opening Notepad.")
        elif "google" in lower_text:
            query = transcript_or_result
            webbrowser.open(f"https://www.google.com/search?q={query}")
            tts.speak("Searching Google.")
        else:
            tts.speak("Sorry, I don't know how to handle that yet.")

    else:
        tts.speak("Invalid command input.")
