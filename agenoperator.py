# create a virtual env e.g conda create -n autogen python=3.12
# pip install -U autogen-agentchat autogen-ext[openai,web-surfer] 
# playwright install 

import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer


async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o-2024-11-20")
    termination = MaxMessageTermination(
        max_messages=20) | TextMentionTermination("TERMINATE")
    
    webmodal = MultimodalWebSurfer(
        downloads_folder="D:\Lab\llm\autogen",
        name="webmodal",
        description="an agent that solves tasks by browsing the web",
        model_client=model_client,
        headless=False,
    )
    assistant = AssistantAgent(
        name="assistant",
        description="an agent that verifies and summarizes information",
        system_message="You are a task verification assistant who is working with a web surfer agent to solve tasks. At each point, check if the task has been completed as requested by the user. If the webmodal responds and the task has not yet been completed, respond with what is left to do and then say 'keep going'. If and only when the task has been completed, summarize and present a final answer that directly addresses the user task in detail and then respond with  TERMINATE."
        "assistant", model_client=model_client)

    selector_prompt = """You are the cordinator of role play game. The following roles are available:
    {roles}. Given a task, the webmodal will be tasked to address it by browsing the web and providing information.  The assistant will be tasked with verifying the information provided by the webmodal and summarizing the information to present a final answer to the user. 
    If the task  needs assistance from a human user (e.g., providing feedback, preferences, or the task is stalled), you should select the user_proxy role to provide the necessary information.

    Read the following conversation. Then select the next role from {participants} to play. Only return the role.

    {history}

    Read the above conversation. Then select the next role from {participants} to play. Only return the role.
    """
    user_proxy = UserProxyAgent(name="user_proxy", description="a human user that should be consulted only when the assistant is unable to verify the information provided by the webmodal")
    
    team = SelectorGroupChat(
        [webmodal, assistant, user_proxy],
        selector_prompt=selector_prompt,
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"), termination_condition=termination)
    user_input = input("What the topic you want to search: ")
    await Console(team.run_stream(task=user_input))

    await webmodal.close()

asyncio.run(main())