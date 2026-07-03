
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash-lite')

st.title("🤖 AskBuddy – AI QnA Bot")
st.markdown("My QnA Bot with LangChain and Google Gemini 2.5!")

query = st.chat_input("Ask anything here...")   # creates the chat input box, query is str type here

if "messages" not in st.session_state:  # checks if the key "messages" is stored in the session_sate (dict)
    st.session_state.messages = []      # begins to store an empty list by creating that key of "messages" {EMPTY LIST}

for message in st.session_state.messages:       # iterates through the .messages list (each is a dict) and prints out the role and content from that dict
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content) 


if query:

    st.session_state.messages.append({"role": "user", "content": query})       # adds dict objects, containg role and content to the empty list created above
    st.chat_message("User").markdown(query)                                     # shows the query written by the user 
    res = llm.invoke(query)
    st.session_state.messages.append({"role": "ai", "content": res.content})    # adds dict objects, for the response as well, stored in session_state
    st.chat_message("AI").markdown(res.content)                                 # shows the AI response given 



# Streamlit execution happens all over, once you hit enter or enter a new query
# That's why the messages needs to be stored and the be print up
# Even though it appears to be a static page, the stored messages (chat history) is being printed over again using the for loop
# hence, it's running the code all over again, based on its execution patterns
