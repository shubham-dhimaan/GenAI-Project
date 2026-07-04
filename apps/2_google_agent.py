from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from langgraph.checkpoint.memory import MemorySaver


model = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()


agent = create_agent(
    model = model,
    tools = [search.run],
    system_prompt = "You are a search agent that can use Google Search to answer any question."
)


while True:
    query = input("User: ")

    if query.lower() in ['bye', 'exit']:
        break;

    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    print(response['messages'][-1].content)