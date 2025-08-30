# Here We Creating ChatBot with Memory
import os 
from typing import TypedDict,Dict,List,Union
from langgraph.graph import StateGraph,START,END
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage,AIMessage
from dotenv import load_dotenv
from openai import conversations
from urllib3 import response

load_dotenv()

class AgentState(TypedDict):
    message : List[Union[HumanMessage,AIMessage]]

llm = ChatMistralAI(
    model = 'mistral-small-latest',
    api_key = os.getenv('MISTRAL_API_KEY')
)

def process(state:AgentState)->AgentState:
    """ This node will solve the request you input """
    response = llm.invoke(state['message'])
    state['message'].append(AIMessage(content=response.content))
    print(f'\nAI : {response.content}')

    return state

graph = StateGraph(AgentState)
graph.add_node('process',process)
graph.add_edge(START,'process')
graph.add_edge('process',END)
Bot = graph.compile()

conversations_history = []
user_input = input("Enter :")
while user_input!= "exit":
    conversations_history.append(HumanMessage(content = user_input))
    result = Bot.invoke({'message':conversations_history})
    conversations_history = result['message']
    user_input = input("Enter :")

# Store the conversation in text file 
with open("conversation.txt","w") as file:
    file.write(f"Your conversation log :\n")

    for message in conversations_history:
        if isinstance(message, HumanMessage):
            file.write(f"You : {message.content}\n")
        if isinstance(message,AIMessage):
            file.write(f"AI : {message.content}\n\n")
    file.write(f"End of conversation")
print(f"Conversation saved to conversation.txt.")




