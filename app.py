from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI,AzureOpenAI,AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from ai import embedding
import PyPDF2

app = Flask(__name__)
CORS(app, origins="http://localhost:5173")


UPLOAD_FOLDER = "C:/Users/Akhil/Documents/p_projects/Q_A_generate_resume/backend/python_Q_A/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  


@app.route("/analyze", methods=["POST"])
def analyse_text():
    try:
        description = request.form.get("description") 
        # print("description---", description)

        file = request.files.get("resume") 
        # print("fileName---", file)

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path) 

        result = embedding(description, file)
        print("result---", result)
        return jsonify(result), 200
        # return jsonify({"message": "Text analysed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == "__main__":
    app.run(debug=True, port=5000)
