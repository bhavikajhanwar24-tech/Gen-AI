from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print

# Tool 1
@tool
def calculator(expression: str) -> float:
    """Use this tool to solve mathematical expressions."""
    return eval(expression)

# Tool 2
@tool
def get_text_length(text: str) -> int:
    """Use this tool to count the number of characters in a text."""
    return len(text)

tools = {
    "calculator": calculator,
    "get_text_length": get_text_length
}

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# bind both tools
llm_with_tool = llm.bind_tools([calculator, get_text_length])

messages = []
prompt = input("You: ")
messages.append(HumanMessage(content=prompt))

result = llm_with_tool.invoke(messages)

print("Tool Calls:", result.tool_calls)

messages.append(result)

while result.tool_calls:
    for tool_call in result.tool_calls:
        tool_name = tool_call["name"]
        tool_result = tools[tool_name].invoke(tool_call["args"])

        print(f"{tool_name} Output:", tool_result)
        messages.append(ToolMessage(content=str(tool_result),tool_call_id=tool_call["id"]))

    result = llm_with_tool.invoke(messages)
    messages.append(result)

print(result.content)