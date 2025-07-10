import os
import google.generativeai as genai

# Load from env automatically if you've done dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash-latest")

def call_gemini(user_text: str) -> dict:
    """
    Calls Gemini with strict JSON enforcement.
    """
    system_prompt = """
    You are an AI assistant.
    When given a user command, analyze it and respond ONLY in JSON.
    Use this exact JSON structure:

    {
      "intent": "<string>",
      "parameters": { "<key>": "<value>", ... },
      "response_text": "<string>"
    }

    Do not add any text before or after the JSON block.
    """

    final_prompt = f"{system_prompt}\n\nUser command: {user_text}"

    try:
        response = model.generate_content(final_prompt)
        raw_text = response.text.strip()

        # Parse JSON
        import json
        result = json.loads(raw_text)

        return result

    except Exception as e:
        print(f"[llm_client] Gemini call failed: {e}")
        return {
            "intent": None,
            "parameters": {},
            "response_text": "Sorry, I could not understand your request."
        }
