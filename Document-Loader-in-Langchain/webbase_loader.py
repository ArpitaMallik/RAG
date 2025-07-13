from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

url = 'https://www.kaggle.com/competitions'
loader = WebBaseLoader(url)
docs = loader.load()

prompt = PromptTemplate.from_template(
    "Based on the text below, answer the question:\n\nQuestion: {question}\n\nText:\n{text}"
)

parser = StrOutputParser()

# print(len(docs))
# print(docs[0].page_content)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.05)

chain = prompt | model | parser

print(chain.invoke({'question': 'What are the featured competitions?', 'text': docs[0].page_content}))