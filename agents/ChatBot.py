from typing import TypedDict,List
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
import os 
from dotenv import load_dotenv 

load_dotenv()

class AgentState(TypedDict):
    message : List[HumanMessage]

# llm = ChatOpenAI(model = 'gpt-4o')

llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

def process(state:AgentState)->AgentState:
    response = llm.invoke(state['message'])
    print(f"\nAI : {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node('process',process)
graph.add_edge(START,'process')
graph.add_edge('process',END)

agent = graph.compile()

user_input = input("Enter :")
while user_input!= "exit":
    agent.invoke({'message': [HumanMessage(content=user_input)]})
    user_input = input("Enter :")