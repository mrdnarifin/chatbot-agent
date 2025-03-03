import asyncio
from dataclasses import dataclass
from typing import List, Optional

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler
)
from autogen_core.models import ChatCompletionClient
from autogen_core.tool_agent import ToolAgent
from autogen_core.tools import ToolSchema, FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

@dataclass
class Message:
    content: str

# Simple calculator tool
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
    
class AssistantAgent(RoutedAgent):
    def __init__(
            self,
            model_client: ChatCompletionClient,
            tool_schema: List[ToolSchema],
            tool_agent_type: str,
            max_messages: Optional[int] = None
    ) -> None:
        super().__init__("Assistant with calculator")
        # inititalize agent properties
        # Implementation details omitted

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> Optional[Message]:
        # MEssage handling logic
        # Implementation details omitted ..
        pass

async def main() -> None:
    # Create runtime
    runtime = SingleThreadedAgentRuntime()

    # Setup calculator tool
    calculator_tool = FunctionTool(calculator, description="Basic Calculator")
    tools = [calculator_tool]

    # create OpenAI client
    model_client = OpenAIChatCompletionClient(model="gpt-4-turbo")

    # Register assistant agent with runtime
    await AssistantAgent.register(
        runtime,
        "assistant",
        lambda: AssistantAgent(
            model_client=model_client,
            max_messages=10
        )
    )

    # Example usage
    assistant = AgentId("assistant", "default")
    response = await runtime.send_message(
        Message("What is the result of 545.34567 * 34555.34"),
        assistant
    )

    if response:
        print(f"Response: {response.content}")

    await runtime.stop_when_idle()

asyncio.run(main())