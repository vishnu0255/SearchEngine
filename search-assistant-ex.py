#Arxiv-search
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
## Custom tools[RAG Tool]
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.tools.retriever import create_retriever_tool

api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

tools=[wiki,arxiv]

loader=WebBaseLoader("https://docs.smith.langchain.com/")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vectordb=FAISS.from_documents(documents,HuggingFaceEmbeddings())
retriever=vectordb.as_retriever()
retriever_tool=create_retriever_tool(retriever,"langsmith-search","Search any information about Langsmith ")
tools=[wiki,arxiv,retriever_tool]
print(tools)

## Run all this tools with Agents and LLM Models
## Tools, LLM-->AgentExecutor
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import openai
load_dotenv()
import os

groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="openai/gpt-oss-20b",api_key=groq_api_key)

## Prompt Template
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad", optional=False)
])
## Agents
from langchain_classic.agents import create_openai_tools_agent
agent=create_openai_tools_agent(llm,tools,prompt)

## Agent Executer
from langchain_classic.agents import AgentExecutor
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
print(agent_executor.invoke({"input":"Tell me about Langsmith"}))
print(agent_executor.invoke({"input":"What is Machine Learning?"}))
print(agent_executor.invoke({"input":"What's the paper 1706.03762 about?"}))