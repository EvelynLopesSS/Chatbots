import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from pdfplumber import open as open_pdf

class DocumentReader:
    def __init__(self):
        self.ollama = Ollama(model="mistral")

    def extract_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            with open_pdf(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    text += page.extract_text()
        return text

    def generate_response(self, uploaded_file, query_text):
        # Load document if file is uploaded
        if uploaded_file is not None:
            documents = [self.extract_text([uploaded_file])]
            # Split documents into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.create_documents(documents)
            # Select embeddings
            embeddings = OllamaEmbeddings(model='nomic-embed-text')
            # Create a vectorstore from documents
            db = Chroma.from_documents(texts, embeddings)
            # Create retriever interface
            retriever = db.as_retriever()
            # Create QA chain
            qa = RetrievalQA.from_chain_type(self.ollama, chain_type='stuff', retriever=retriever)
            return qa.run(query_text)

# Page title
st.set_page_config(page_title='🦜🔗 Ask DocApp')
st.title('🦜🔗 Ask DocApp')

# Explanation of the App
st.header('About the App')
st.write(
""" 
The App is an advanced question-answering platform that allows users to upload pdf documents and receive answers to their queries based on the content of these documents. Utilizing Ollama, an open source model, the app provides insightful and contextually relevant answers.

### How It Works
- Upload a Document: You can upload any text document in `.pdf` format.
- Ask a Question: After uploading the document, type in your question related to the document's content.
- Get Answers: AI analyzes the document and provides answers based on the information contained in it.

""")
st.sidebar.write("""
### Get Started
Simply upload your document and start asking questions!
""")

# File upload
uploaded_file = st.sidebar.file_uploader('Upload files', type='pdf')

# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please write your question here.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted :
        with st.spinner('Processing...'):
            doc_reader = DocumentReader()
            response = doc_reader.generate_response(uploaded_file, query_text)
            result.append(response)

if len(result):
    st.info(response)