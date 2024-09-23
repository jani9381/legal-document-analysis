import streamlit as st 
pip install PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(r"C:\Users\JANI BASHA\Downloads\google.env.txt")
GOOGLE_API_KEY = "AIzaSyALTecTsQ_s8o4IXavgmxQ-Z1WovbcbhjM"

# Retrieve the Google API key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    st.error("Google API Key not found! Please check your environment variables.")
else:
    genai.configure(api_key=api_key)
    st.write("Google API Key loaded successfully.")

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Handle None return
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
            print(f"Error reading {pdf.name}: {e}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Smaller chunks for better context
    chunks = text_splitter.split_text(text)
    st.write(f"Text split into {len(chunks)} chunks.")
    return chunks

# Function to create and save FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.write('success')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.write("Vector store saved.")

# Function to get conversational chain using Google Generative AI
def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)  # Use the new model
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a legal advisor in India providing information about crime punishments according to Indian law. The Indian Penal Code (IPC) was replaced by the Bharatiya Nyaya Sanhita (BNS) on July 1, 2024.'),
        ('human', 'Based on the following context, answer the question. Context: {context} Question: {user_input}')
    ])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and perform similarity search
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        print("Loading FAISS vector store...")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded.")
        
        # Perform similarity search
        docs = new_db.similarity_search(user_question)
        print(f"Documents found: {len(docs)}")
        
        if not docs:
            st.write("No relevant documents found.")
            return
        
        # Combine all document text into context
        context = " ".join(doc.page_content for doc in docs)
        
        # If no text is found, inform the user
        if not context:
            st.write("No context found in the documents.")
            return

        # Display all found documents
        st.write("Relevant documents found:")
        for doc in docs:
            st.write(doc.page_content)
        
        # Proceed with the conversational chain
        chain = get_conversational_chain()
        
        # Invoke chain with the correct input structure
        response = chain.invoke({"input_documents": docs, "context": context, "user_input": user_question})
        
        # Check and display response
        print("Response:", response)
        st.write("Reply: ", response.get("output_text", "No output_text found"))
    
    except Exception as e:
        if "429" in str(e):
            # Only show this error when the quota is exceeded
            st.error("API Quota exceeded. This happens when too many requests are sent in a short time or the quota is reached for the day. Please try again later.")
            st.info("Note: To avoid this issue, you can reduce the number of requests made to the API or check your API quota on your Google Cloud account. If you need more usage, consider upgrading your plan.")
        else:
            st.error(f"Error while processing input: {e}")
        print(f"Error while processing input: {e}")

# Main function to run the app
def main():
    st.header("Legal Document Analysis ⚖")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        print(f"User question: {user_question}")
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if raw_text:
                            st.write("PDF text extraction successful.")
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.success("PDFs processed and vector store created.")
                        else:
                            st.error("No text extracted from the uploaded PDFs.")
                            print("No text extracted from the PDFs.")
                    except Exception as e:
                        st.error(f"Error during processing: {e}")
                        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()  # Call the main function
