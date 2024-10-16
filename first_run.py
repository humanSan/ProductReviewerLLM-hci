from langchain_google_genai import ChatGoogleGenerativeAI
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")

llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key = key)

#genai.configure(api_key = key) 


loader = JSONLoader(
    file_path='./ReviewJson/Headphones_text.json',
    jq_schema='.review',
    text_content=False,
    json_lines=True)

docs = loader.load()
#print(docs[100].page_content)

g_embed = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

vectorstore = Chroma.from_documents(collection_name = "headphones", documents=docs, embedding=g_embed, persist_directory="./rag_vectorstore")

#this function uses an existing Chroma vector database. The line above I think creates/appends the vector database on every run, so it keeps growing in size.
#vectordb = Chroma(collection_name = "headphones", persist_directory="./rag_vectorstore", embedding_function=g_embed)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 50})
#retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 40})

dfile = open("./ProductDescriptions/HeadphonesDescription.txt", "r")
description = dfile.read()

#model = genai.GenerativeModel("gemini-1.5-flash")
'''
system_prompt = (
    "You are a product analyst."
    "Based on the information in the following product reviews, create a useful "
    "answer to the question about the product. Consider different perspecives and different possibilities." 
    "Be very detailed and give descriptive answers. Write your response objectively and directly about the product."
    "Here is the product description for this product:" 
    + description +
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


st.title("🦜🔗 Test Streamlit-Langchain App")


with st.form("form"):
   user_input = st.text_area(
      "Enter prompt:",
   )
   submitted = st.form_submit_button("Enter")

if(submitted):
   
   response = rag_chain.invoke({'input': user_input})
   st.write(response["answer"])
'''



