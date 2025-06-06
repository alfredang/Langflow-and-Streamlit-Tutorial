import os
from dotenv import load_dotenv
from agents import Agent, Runner
import asyncio


# Load environment variables from .env file
load_dotenv()

# Access your API key
api_key = os.getenv("OPENAI_API_KEY")

async def test_installation():
    agent = Agent(
        name="Test Agent",
        instructions="You are a helpful assistant that provides concise responses."
    )
    result = await Runner.run(agent, "Hello! Are you working correctly?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(test_installation())


   