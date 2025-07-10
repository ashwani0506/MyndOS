import subprocess
import webbrowser

from tts import TTS

tts = TTS()

def dispatch_intent(result_json):
    """
    Execute commands based on Gemini response.
    """
    intent = result_json.get("intent")
    params = result_json.get("parameters", {})
    speak_text = result_json.get("response_text", "")

    if speak_text:
        speak(speak_text)

    if intent == "open_app":
        app = params.get("app_name")
        if app:
            subprocess.run(app)
    elif intent == "search_web":
        query = params.get("query")
        if query:
            webbrowser.open(f"https://www.google.com/search?q={query}")
    else:
        speak("Sorry, I don't know how to do that yet.")
