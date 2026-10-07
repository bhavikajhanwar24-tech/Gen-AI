from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool 
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print 

#1 creating a tool 

@tool
def get_text_length(text: str) -> int:
    """
    Calculate and return the number of characters in the given text.
    Use this tool whenever the user asks for text length or character count.
    """
    return len(text)

tools = {"get_text_length" : get_text_length}

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

#tool binding 
llm_with_tool = llm.bind_tools([get_text_length])

message = []
prompt = input("You: ")
query = HumanMessage(content = prompt)
message.append(query)

result = llm_with_tool.invoke(message)

message.append(result)

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_result = tools[tool_name].invoke(result.tool_calls[0]["args"])

    message.append(ToolMessage(content=str(tool_result),tool_call_id=result.tool_calls[0]["id"]))

    result = llm_with_tool.invoke(message)
    print(result.content)

else:
    print(result.content)