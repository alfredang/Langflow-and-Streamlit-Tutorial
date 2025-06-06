from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

deepseek_api_key=os.getenv("DEEPSEEK_API_KEY")

openai=OpenAI()
message = [{'role':'user','content':"what is 2+3?"}]


deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
model_name = "deepseek-chat"

response = deepseek.chat.completions.create(model=model_name, messages=message)
print(response.choices[0].message.content)