from dotenv import load_dotenv
load_dotenv()
import uuid

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from langgraph.checkpoint.memory import MemorySaver


model = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
memory = MemorySaver()

agent = create_agent(
    model = model,
    tools = [search.run],
    checkpointer = memory,                  
    system_prompt = "You are a search agent that can use Google Search to answer any question."
)

current_id = str(uuid.uuid4())  #random generated chat ID

while True:
    query = input("User: ")
        
    if query.lower() in ['bye', 'exit']:
        break;

    if query.lower() in ['new', 'new chat']:
        previous_id = current_id
        current_id = str(uuid.uuid4())          # generates new current_id, skips this iteration and runs the loop again asking for input
        continue;

    if query.lower() in ['previous', 'back']:
        current_id = previous_id                # adding a previous chat functionality 
        continue;

    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": current_id}})              # adding a chat ID to the to store memmory (local)
    
    print(response['messages'][-1].content)