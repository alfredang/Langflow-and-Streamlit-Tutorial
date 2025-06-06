from agents import Agent, Runner

# Define the agent
agent = Agent(
    name="Basic Agent",
    instructions="You are a helpful assistant. Respond in all caps.",
    # Optional: Define tools
)

# Create a runner
runner = Runner()

# Run the agent
result = await runner.run(agent, ["Hello"])

print(result)