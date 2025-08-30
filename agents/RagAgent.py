import os
from dotenv import load_dotenv
from langchain_core import vectorstores
from langchain_core import messages
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Sequence
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,ToolMessage
from operator import add as add_messages
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatMistralAI(
    model = "mistral-large-latest",
    temperature=0,
    api_key=os.getenv("MISTRAL_API_KEY")
)

# Our Embeddings Model - has to compatible with the llm we are using
embeddings = MistralAIEmbeddings(
    model = "mistral-embed",
)

pdf_path = "agents/AIML Industry Analysis.pdf"

# for debugging purpose 
if not os.path.exists(pdf_path):
    raise FileExistsError(f"File not found {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

# Check pdf is there 
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error in loadig PDF {(e)}")
    raise

# Chunking process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap =  200
)

pages_split = text_splitter.split_documents(pages)

persist_directory = r"/Users/darshandhanwade/AI Agents/LangGraph/Exercise/agents"
collection_name = "My_collection"

# if our collection does not exists in the directory, then we create the directory using os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # Here we are creating the chroma vector database
    vectorstores = Chroma.from_documents(
        documents = pages_split,
        embedding = embeddings,
        persist_directory = persist_directory,
        collection_name = collection_name
    )
    print(f"Created ChromaDB vector store!")
except Exception as e:
    print(f"Error setting up ChromaDB {str(e)}")
    raise

# Now we create retriever
retriever = vectorstores.as_retriever(
    search_type = 'similarity',
    search_kwargs = {"k":5} # k is the amount of chunks to return 
)

@tool
def retriever_tool(query:str)->str:
    """ This tool searches and returns the information from the document uploaded by user."""

    docs = retriever.invoke(query)

    if not docs:
        return "I found no revelent information in the uploaded document."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}: \n {doc.page_content}")
    
    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages  : Annotated[Sequence[BaseException],add_messages]

def should_continue(state: AgentState):
    """Check if the last messages contain tool calls."""
    result = state['messages'][-1]
    return hasattr(result,'tool_calls') and len(result.tool_calls) > 0

system_prompt = system_prompt = """
You are an intelligent AI assistant who answers questions based strictly on the content of provided PDF documents.  
Use the retriever tool to find and analyze the most relevant sections of the PDFs before answering.  

If the user's question cannot be fully answered from the documents:  
- First, clearly say that the information is not found in the provided PDFs.  
- Then, offer the user a choice by asking: 
  "I have some relevant knowledge outside the document. Would you like me to include it in my answer?"  

Always cite the specific parts or sections of the documents you use in your answers.  
Never mix outside knowledge with document-based answers unless the user explicitly allows it.  
"""
tools_dict = {our_tool.name : our_tool for our_tool in tools } # create a dictionary of tools

# LLM Agents
def call_llm(state:AgentState)->AgentState:
    """ This function call the LLM with the current state. """
    messages = list(state['messages'])
    messages = [SystemMessage(content = system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages':[message]}

# Retriever agent
def take_actions(state:AgentState)->AgentState:
    """ Execute tool calls from LLM's response """
    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query : {t['args'].get('query','No query is provided')}")

        if not t['name'] in tools_dict: # Checks if the valid tool is present
            print(f"\nTool: {t['name']} does not exists")
            result = 'Incorrect tool name, please retry and select the avilable tool in list of tools'

        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query','No query is provided'))
            print(f"result length : {len(str(result))}")

        # Append the tool messages
        results.append(ToolMessage(tool_call_id = t['id'],name = t['name'],content=str(result)))

    print(f"Tool execution complete. Back to the model!")
    return {'messages':results}

graph = StateGraph(AgentState)

graph.add_node('llm',call_llm)
graph.add_node('retriever_agent',take_actions)

graph.add_conditional_edges(
    'llm',
    should_continue,
    {True : 'retriever_agent',False : END}
)
graph.add_edge('retriever_agent','llm')
graph.set_entry_point('llm')

rag_agent = graph.compile()

def running_agent():
    print(f"\n =========RAG AGENT==========")

    while True:
        user_input = input(f"What is your question: ")
        if user_input.lower() in ['exit','quit']:
            print(f"\nExiting Agent!")
            break

        messages = [HumanMessage(content=user_input)] # convert back to human messages type
        result = rag_agent.invoke({'messages':messages})

        print(f"\n=========ANSWER===========")
        print(result['messages'][-1].content)

running_agent()

    



