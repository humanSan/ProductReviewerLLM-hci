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
import chromadb


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
print(docs[100].page_content)


web_chroma = chromadb.HttpClient(host='localhost', port=8000)
collection = web_chroma.get_or_create_collection("headphones")

goog_embed = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

vectorstore_web = Chroma(
    client=web_chroma,
    collection_name="headphones",
    embedding_function = goog_embed
)


#vectorstore = Chroma.from_documents(collection_name = "headphones", documents=docs, embedding=goog_embed, persist_directory="./rag_vectorstore")
retriever = vectorstore_web.as_retriever(search_type="similarity", search_kwargs={"k": 50})

retrieved_docs = retriever.invoke("What is a common cause of these headphones breaking?")
print(len(retrieved_docs))

#model = genai.GenerativeModel("gemini-1.5-flash")

system_prompt = (
    "You are an analyst for product reviews."
    "Use the following user reviews for the given product to answer "
    "the question about the product. If you don't know the answer, say that you "
    "don't know. Give useful and descriptive answers."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

'''
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
