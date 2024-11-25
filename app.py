import streamlit as st
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Load environment variables from .env
load_dotenv()


def vectordb_creator(urls):
    """Load data from URLs, split it, and create a vector database."""
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','])
    docs = text_splitter.split_documents(data)

    # Initialize embeddings and create a FAISS vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)

    return vectordb


def get_context_retriever_chain(vectordb):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3})
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
        )

    # Create a prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")])
    
    retriever_chain = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt)
    
    return retriever_chain


def get_conversational_rag_chain(retriever_chain): 
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "\n\n"
        "{context}"
    )
    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt),
         MessagesPlaceholder("chat_history"),
         ("human", "{input}")])
    
    stuff_documents_chain = create_stuff_documents_chain(model, qa_prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']


# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url1 = st.text_input("Webiste URL 1")
    website_url2 = st.text_input("Webiste URL 2")
    website_url3 = st.text_input("Webiste URL 3")

urls = [website_url1, website_url2, website_url3]
# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = vectordb_creator(urls)    

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
    
    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

