import os
import json
import PyPDF2
from flask import request
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI,AzureOpenAI,AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.document_loaders import PDFPlumberLoader
# from pypdf import PdfReader
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# from PyPDF2

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_ENDPOINT")  
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
OPENAI_API_TYPE = os.getenv("AZURE_OPENAI_TYPE")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")

os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_TYPE"] = OPENAI_API_TYPE
os.environ["OPENAI_DEPLOYMENT_NAME"] = OPENAI_DEPLOYMENT_NAME
os.environ["OPENAI_EMBEDDING_MODEL_NAME"] = OPENAI_EMBEDDING_MODEL_NAME


def load_vector_and_generate_qa(file_name):
    # Load FAISS Store
    file_path = f"{file_name}"
   
    embedding_azure = AzureOpenAIEmbeddings(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,
            openai_api_type=OPENAI_API_TYPE,
            model = "embed-dev",
        )

    vector_store = FAISS.load_local(file_path, embedding_azure,allow_dangerous_deserialization=True)
    print("vectordb file is loading-----")


    # Retrieving Q & A from the resume and Job Description----------------------------------------------------------
    query = """
            Generate a list of **100 interview questions and answers** based on the given resume and job description.

            - Include a mix of **technical questions, coding questions (with answers), framework-related questions, and concept-based questions**.
            - Ensure a **diverse range of difficulty levels**.
            - Each coding question must have a **clear and complete answer**.
            - The response **must be in valid JSON format** with this structure:

            {
            "interview_questions": [
                {
                "question": "string",
                "answer": "string"
                },
                ...
            ]
            }

            Ensure the JSON is **well-formatted** and **fully valid**.
            """


    search_results = vector_store.similarity_search(query, k=5)  

    context = "\n\n".join([doc.page_content for doc in search_results])

    # LLM Model
    llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME, model="gpt-4", temperature=0.7)

    # Send a string instead of a dictionary
    response = llm.invoke(f"{context}\n\n{query}")

    # Extract response text
    if response and hasattr(response, "content"):
        response_text = response.content  # Only if content attribute exists
    elif hasattr(response, "message"):
        response_text = response.message['content']  # Extract from message

    # print("Raw Response:", response_text)
    return response_text

def embedding (description, filename):
    try:
        if not description:
            return "Description not found"
        if not filename:
            return "File not found"
        # print("description---", description)
        # print("filename---", filename)
        filename_path = "C:/Users/Akhil/Documents/p_projects/Q_A_generate_resume/backend/python_Q_A/uploads/"+filename.filename
        # print("filename---", filename)

        print("file is loading -------")
        loader  = PDFPlumberLoader(file_path=filename_path)
        documents = loader.load()
        # print(documents)

        print("file is spliting -------")
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".",],
            chunk_size=1000,
            chunk_overlap=0,
        )
        docs_file = splitter.split_documents(documents)
        # docs_description = splitter.split_documents(description)

        description1 = Document(page_content=description)
        docs_file.append(description1)
        # docs = docs_file + description1

        # print("docs_file ---", docs_file)
      
        embedding_azure = AzureOpenAIEmbeddings(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,
            openai_api_type=OPENAI_API_TYPE,
            model = "embed-dev",
        )

        print("file is embedding -------")
        vector_store = FAISS.from_documents(docs_file, embedding_azure)
        print(vector_store)
        print(vector_store.index.ntotal)
        print(vector_store.index.d) 

        file_name = filename.filename.split(".")[0]
        vectorfile_folder = "vectorfiles"
        if not os.path.exists(vectorfile_folder):
            os.makedirs(vectorfile_folder)

        print("file is saving in vectordb folder -------")
        file_path = f"C:/Users/Akhil/Documents/p_projects/Q_A_generate_resume/backend/python_Q_A/vectorfiles/{file_name}.pkl"
        vector_store.save_local(file_path)

        response = load_vector_and_generate_qa(file_path)
        return response

    except Exception as e:
        print(e)
        return str(e)
    