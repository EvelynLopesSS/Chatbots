from flask import Flask, render_template, request
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llm import HuggingFaceHub
#from htmlTemplates import css, bot_template, user_template
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter 
from langchain_community.embeddings import OllamaEmbeddings

app = Flask(__name__)

class PDFProcessor:
    def __init__(self):
        self.ollama = Ollama(model="mistral")

    def extract_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

class TextProcessor:
    def __init__(self):
        pass

    def get_text_chunks(self, text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

class ConversationManager:
    def __init__(self):
        self.vectorstore = None

    def get_vectorstore(self, text_chunks):
        embeddings = embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text')
        self.vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return  self.vectorstore

    def get_conversation_chain(self, vectorstore):
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                             model_kwargs={"temperature": 0.5, "max_length": 512})

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain

pdf_processor = PDFProcessor()
text_processor = TextProcessor()
conversation_manager = ConversationManager()
conversation_chain =None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    #global conversation_chain 
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = pdf_processor.extract_text(pdf_docs)
    text_chunks = text_processor.get_text_chunks(raw_text)
    conversation_manager.get_vectorstore(text_chunks)
    #conversation_chain = conversation_manager.get_conversation_chain(vectorstore)
    return render_template('index.html', pdf_processed=True)

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.form['user_question']
    conversation_chain = conversation_manager.get_conversation_chain(conversation_manager.vectorstore)
    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']
    return render_template('index.html', user_question=user_question, chat_history=chat_history)

if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True)