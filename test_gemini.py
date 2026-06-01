import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
keys = [
    part.strip()
    for part in (os.getenv('GEMINI_API_KEYS') or '').split(',')
    if part.strip()
]
if not keys and os.getenv('GEMINI_API_KEY'):
    keys = [os.getenv('GEMINI_API_KEY', '').strip()]

client = genai.Client(api_key=keys[0] if keys else None)

print("Gemini client initialized.")
print("Configured key slots:", len(keys))
