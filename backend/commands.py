import subprocess
import urllib.parse
import webbrowser

from tts import tts


def execute_command(transcript: str):
    """
    Placeholder command dispatch: naive keyword matching on the raw transcript.

    Superseded in step 2/3 of the build by the planner + trusted execution layer
    (risk tiers, confirmation, action log). Nothing here should grow.
    """
    lower_text = transcript.lower()

    if "open notepad" in lower_text:
        subprocess.run(["notepad"])
        tts.speak("Opening Notepad.")
    elif "google" in lower_text:
        query = urllib.parse.quote_plus(transcript)
        webbrowser.open(f"https://www.google.com/search?q={query}")
        tts.speak("Searching Google.")
    else:
        tts.speak("Sorry, I don't know how to handle that yet.")
