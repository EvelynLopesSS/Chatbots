from flask import Flask, render_template, request
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time



app = Flask(__name__)

# Set up Streamlit session state
class SessionState:
    pass

st = SessionState()

# Initialize Streamlit session state variables
if not hasattr(st, 'template'):
    st.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
    
    Context: {context}
    History: {history}
    
    User: {question}
    Chatbot:"""

if not hasattr(st, 'prompt'):
    st.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.template,
    )

if not hasattr(st, 'memory'):
    st.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )

if not hasattr(st, 'vectorstore'):
    st.vectorstore = Chroma(persist_directory='jj',
                            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
                            )

if not hasattr(st, 'llm'):
    st.llm = Ollama(
        model="mistral",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

if not hasattr(st, 'chat_history'):
    st.chat_history = []

# Função para criar os diretórios
def create_directories():
    if not os.path.exists('files'):
        os.mkdir('files')
    if not os.path.exists('jj'):
        os.mkdir('jj')

# Chamando a função para criar os diretórios
create_directories()

@app.route('/', methods=['POST'])
def home():
    return render_template('chatpdf.html')
@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            # Save the file
            file.save(os.path.join('files', filename))
            # Further processing
            if not os.path.isfile("files/"+filename+".pdf"):
                bytes_data = file.read()
                f = open("files/"+filename+".pdf", "wb")
                f.write(bytes_data)
                f.close()
                loader = PyPDFLoader("files/"+filename+".pdf")
                data = loader.load()

                # Initialize text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len
                )
                all_splits = text_splitter.split_documents(data)

                # Create and persist the vector store
                vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=OllamaEmbeddings(model="mistral")
                )
                vectorstore.persist()

            retriever = vectorstore.as_retriever()
            # Initialize the QA chain
            global qa_chain
            if qa_chain is None:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type='stuff',
                    retriever=retriever,
                    verbose=True,
                    chain_type_kwargs={
                        "verbose": True,
                        "prompt": prompt,
                        "memory": memory,
                    }
                )

            return jsonify({'message': 'File uploaded successfully.'})
        else:
            return jsonify({'error': 'No file uploaded.'})
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    user_message = {"role": "user", "message": user_input}
    chat_history.append(user_message)
    response = qa_chain(user_input)
    chatbot_message = {"role": "assistant", "message": response['result']}
    chat_history.append(chatbot_message)
    return jsonify({"user_message": user_input, "chatbot_message": response['result']})

if __name__ == '__main__':
    app.run(debug=True)