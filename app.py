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
g_embed = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

#genai.configure(api_key = key) 


filenames = ["AlarmClock", "Headphones", "IceBucket", "WashingMachine"]

docs=[]
descriptions = []
vectordblist = []
retrievers = []
rag_chain = []

i = 0
for file in filenames:
    loader = JSONLoader(
        file_path='./ReviewJson/' + file + '.json',
        jq_schema='.review',
        text_content=False,
        json_lines=True)
    
    docs.append(loader.load())
    #print(docs[100].page_content)
    dfile = open("./ProductDescriptions/"+file+".txt", "r")
    descriptions.append(dfile.read())

    #vectorstore = Chroma.from_documents(collection_name = file, documents=docs[i], embedding=g_embed, persist_directory="./rag_vectorstore") #use to create new vector db or append to existing db
    vectordblist.append(Chroma(collection_name = filenames[i], persist_directory="./rag_vectorstore", embedding_function=g_embed)) #use to access preexisting vector db
    
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 50}) #use with new db to append
    retrievers.append( vectordblist[i].as_retriever(search_type="similarity", search_kwargs={"k": 50}) )
    
    i+=1



#model = genai.GenerativeModel("gemini-1.5-flash")


st.title("🦜🔗 Test Streamlit-Langchain App")


with st.form("form"):
   user_input = st.text_area(
      "Enter prompt:",
   )
   submitted = st.form_submit_button("Enter")

k = 0
product = st.sidebar.radio("Product", filenames)

if product == "AlarmClock":
    k = 0
if product == "Headphones":
    k = 1
if product == "IceBucket":
    k = 2
if product == "WashingMachine":
    k = 3

if st.button("General Product Summary"):
    # System prompt for automatic summary, using {context} as a placeholder for the reviews
    system_prompt = (
        "You are a product analyst. Summarize the following reviews by identifying key strengths, weaknesses, "
        "and areas for improvement. Highlight what customers liked most and where there were consistent complaints "
        "or suggestions for enhancement. Make sure to include any specific product features or experiences that "
        "were praised or criticized.\n\n"
        "Here is the product description for this product: "
        + descriptions[k] +
        "\n\n{context}"
    )
    
    # Create a prompt template that accepts the 'context' variable
    prompt = ChatPromptTemplate.from_template(
        system_prompt  # This template contains the {context} variable
    )

    question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
    rag_chain = create_retrieval_chain(retrievers[k], question_answer_chain)

    response = rag_chain.invoke({'input': user_input})
    st.write(response["answer"])



if(submitted):
    system_prompt = (
        "You are a product analyst."
        "Based on the information in the following product reviews, create a useful "
        "answer to the question about the product. Consider different perspecives and different possibilities." 
        "Be very detailed and give descriptive answers. Write your response objectively and directly about the product."
        "Here is the product description for this product:" 
        + descriptions[k] +
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
    rag_chain = create_retrieval_chain(retrievers[k], question_answer_chain)

    response = rag_chain.invoke({'input': user_input})
    st.write(response["answer"])

