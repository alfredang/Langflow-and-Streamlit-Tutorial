from dotenv import load_dotenv
from agents import Agent, Runner, trace


load_dotenv(override=True)

agent = Agent(name="Jokester", instructions="You are a joke teller", model="gpt-4o-mini")

with trace("Telling a joke"):
    result = await Runner.run(agent, "Tell a joke about Autonomous AI Agents")
    print(result.final_output)