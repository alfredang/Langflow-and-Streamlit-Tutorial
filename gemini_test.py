from google import genai
from dotenv import load_dotenv
load_dotenv()
import os

gemini_api_key = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=gemini_api_key)


messages = ["What is 2+2?"]
response = client.models.generate_content(
    model="gemini-2.0-flash", contents=messages
)

print(response.text)