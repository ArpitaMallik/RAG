from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os


loader = CSVLoader(file_path='Social_Network_Ads.csv', encoding='utf-8')
data = loader.load()

print(data[0])