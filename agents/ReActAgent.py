from email import message
import os
from dotenv import load_dotenv
from typing import Type, TypedDict, Sequence, Annotated
from langgraph.graph import StateGraph, START,END
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool 
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import AIMessage

load_dotenv()

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int,b:int)->int:
    """ this is and addition function which adds the numbers """
    return a + b 

@tool
def multiply(a:int, b:int)->int:
    """ this function will multiply  """
    return a*b

@tool
def subtract(a:int,b:int)->int:
    """ this function will subtract """
    return a-b

tools = [add,multiply,subtract]

model = ChatMistralAI(
    model = "mistral-small-latest",
    api_key = os.getenv('MISTRAL_API_KEY') 
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant. Please answer the query as best of your ability."
    )

    try:
        response = model.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}
    
    except Exception as e:
        # Handle rate limits or API errors
        if "429" in str(e) or "capacity" in str(e).lower():
            friendly_msg = AIMessage(content="⚠️ API is at capacity (rate limit). Don't worry, please try again later.")
        else:
            friendly_msg = AIMessage(content=f"⚠️ API error: {str(e)}")
        
        return {"messages": [friendly_msg]}

def should_continue(state:AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return 'end'
    else:
        return 'continue'

graph = StateGraph(AgentState)

graph.add_node('agent',model_call)
tool_node = ToolNode(tools=tools)
graph.add_node('tools',tool_node)

graph.set_entry_point('agent')

graph.add_conditional_edges(
    'agent',
    should_continue,
    {
        'continue':'tools',
        'end':END
    }
)

graph.add_edge('tools','agent')
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message  = s['messages'][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

input = {'messages':[("user","add 40+12.and then multiply the answer  with 6 and also tell me a joke")]}
print_stream(app.stream(input,stream_mode='values'))
