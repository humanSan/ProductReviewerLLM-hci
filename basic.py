import google.generativeai as genai
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key = key) 


model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Write a story about a man in Estonia with a red motorcycle.")
st.write(response.text)