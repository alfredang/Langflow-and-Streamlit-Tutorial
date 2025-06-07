from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

openai_api_key=os.getenv("OPENAI_API_KEY")

openai=OpenAI()
message = [{'role':'user','content':"what is 2+3?"}]

response = openai.chat.completions.create(model="gpt-4o-mini",messages=message)
print(response.choices[0].message.content)