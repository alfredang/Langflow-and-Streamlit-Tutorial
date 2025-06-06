from groq import Groq
from dotenv import load_dotenv
load_dotenv()
import os

groq_api_key = os.getenv('GROQ_API_KEY')

groq = Groq()
messages = [{"role": "user", "content": "What is 2+2?"}]

response = groq.chat.completions.create(model='llama-3.3-70b-versatile', messages=messages)
print(response.choices[0].message.content)