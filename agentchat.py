from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import UserMessage
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from typing import Literal
from pydantic import BaseModel

class AgentReponse(BaseModel):
    thoughts: str
    response: Literal["happy", "sad","neutral"]

model_client = OpenAIChatCompletionClient(
    model="llama3.2:latest",
    base_url="http://localhost:11434/v1",
    api_key="placeholder",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
    },
    # response_format=AgentReponse
)

# Define a tool
async def web_search(query: str) -> str:
    """Find Information on the web"""
    return "Autogen is a programming framework for building multi-agent applications"

async def get_weather(city: str) -> str:
    return f"The weather in {city} is 73 degrees and Sunny."

async def get_message(input: str) -> str:
    result = await model_client.create([UserMessage(content=input)])
    return result



async def main() -> None:
    # Define an agent
    weather_agent = AssistantAgent(
        name="weather_agent",
        model_client=model_client,
        tools=[get_weather],
    )

    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        # tools=[web_search],
        system_message="Categorize the input as happy, sad, or neutral following the JSON format.",
    )

    # Define a termination condition that stops the task if the critic approves.
    text_termination = TextMentionTermination("APPROVE")

    # await Console(agent.run_stream(task="I am happy"))

    # print(response.inner_messages)
    # print(response.chat_message)

    # # Define a team with a single agent and maximum auto-gen turn of 1
    agent_team = RoundRobinGroupChat([agent, weather_agent],termination_condition=text_termination, max_turns=1)

    while True:
        # get user input from console.
        user_input = input("Enter a message (type 'exit' to leave): ")
        if user_input.strip().lower() == 'exit':
            break
        # run the team and stream message to the console

        stream = agent_team.run_stream(task=user_input)
        await Console(stream)

asyncio.run(main())

