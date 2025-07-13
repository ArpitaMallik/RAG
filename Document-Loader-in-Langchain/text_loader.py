from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text: \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()


#document loader object
loader = TextLoader('cricket.txt', encoding='utf-8')

#load the text file as a document
docs = loader.load()

print(docs)
print(docs[0])  # Print the content of the first document
print(type(docs[0])) #prints document type
print(docs[0].page_content)  # Print the content of the first document
print(docs[0].metadata)  # Print the metadata of the first document

chain = prompt | model | parser

result = chain.invoke({'poem': docs[0].page_content})
print(result)