import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer


def calculator(a: float, b: float, operator: str) -> str:
    try:
        if operator == '+':
            return str(a + b)
        elif operator == '-':
            return str(a - b)
        elif operator == '*':
            return str(a * b)
        elif operator == "/":
            if b == 0:
                return 'Error: Division by zero'
            return str(a/b)
        else:
            return 'Error: Invalid operator. Please use +, -, *, or /'
    except Exception as e:
        return f'Error: {str(e)}'
    
async def web_search(query: str) -> str:
    """Find Information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."
    
async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    termination = MaxMessageTermination(max_messages=10) | TextMentionTermination("TERMINATE")
    assistant = AssistantAgent(
        "assistant",
        model_client=model_client,
        tools=[calculator]
    )

    agent = AssistantAgent(
        name="web_surfer",
        model_client=model_client,
        tools=[web_search],
        system_message="Use tools to solve taks"
    )

    web_surfer_agent = MultimodalWebSurfer(
        name="MultiModalWebSurfer",
        model_client=model_client,

    )
    team = RoundRobinGroupChat([web_surfer_agent], max_turns=3)
    input_user = input("Ask Anything: ")


    if input_user.lower() in ['exit', 'quit']:
        return

    await Console(team.run_stream(task=input_user))

asyncio.run(main())