# File: buddima/chatbot1/main.py

import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

app = FastAPI()

CSV_PATH = "./data/job_descriptions_large.csv"
CHROMA_DIR = "./chroma_db"

# Global variables to hold LLM and vector DB instances
vector_db = None
chain = None

# Define the request body structure for the /chat endpoint
class ChatRequest(BaseModel):
    user_input: str # This field will receive the user's message from the frontend

@app.on_event("startup")
async def startup_event():
    """
    Initializes the vector store (ChromaDB) and the LLM chain (Ollama)
    when the FastAPI application starts up. This ensures they are loaded
    only once, improving performance for subsequent requests.
    """
    global vector_db, chain # Declare global to modify the module-level variables

    print("Starting up FastAPI application...")

    # --- STEP 1: Prepare Vector Store (ChromaDB) ---
    # Check if the CSV data file exists. It's crucial for creating the ChromaDB.
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV data file not found at {CSV_PATH}.")
        print("Please ensure 'job_descriptions_large.csv' is in 'buddima/chatbot1/data/'.")
        raise FileNotFoundError(f"CSV data file not found at {CSV_PATH}. Please ensure it's in the correct location.")

    # Initialize the embedding model. This is used both for creating and loading ChromaDB.
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Check if the ChromaDB directory already exists to decide whether to create or load.
    if not os.path.exists(CHROMA_DIR):
        print("Creating new Chroma vector DB")
        # Load job descriptions from the CSV file into a pandas DataFrame.
        df = pd.read_csv(CSV_PATH)

        # Convert each row of the DataFrame into a LangChain Document.
        # The 'Description' column becomes the page_content, and 'Job Title' is metadata.
        documents = [
            Document(page_content=row["Description"], metadata={"title": row["Job Title"]})
            for _, row in df.iterrows()
        ]

        # Create the Chroma vector store from the documents and embeddings, then persist it to disk.
        vector_db = Chroma.from_documents(documents, embedding=embedding, persist_directory=CHROMA_DIR)
        vector_db.persist() # Save the vector database to the specified directory
        print("Chroma vector DB created")
    else:
        print("Loading existing Chroma vector DB")
        # Load the existing Chroma vector store from the specified directory.
        vector_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        print("Chroma vector DB loaded.")

    # --- STEP 2: Setup LLM (Ollama) & Prompt ---
    try:
        # Initialize the Ollama language model with the "llama3" model.
        llm = Ollama(model="llama3")
        # A small test invocation to ensure the Ollama server and model are accessible.
        llm.invoke("Hello")
        print("Ollama model 'llama3' loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load Ollama model: {e}")
        print("Please ensure the Ollama server is running and the 'llama3' model is downloaded.")
        # Re-raise the exception to prevent the FastAPI app from starting if LLM fails to load.
        raise HTTPException(status_code=500, detail=f"Ollama model 'llama3' could not be loaded: {e}")

    # Define the prompt template for the LLM.
    # It takes 'traits' (user input) and 'context' (relevant job descriptions from ChromaDB).
    prompt_template = PromptTemplate(
        input_variables=["traits", "context"],
        template="""
You are an intelligent and concise career advisor.

A candidate has shared these personality traits:
{traits}

Below are job role descriptions you may use as reference:
{context}

From the roles below, pick the single **best fit**:
- Software Engineer (SE)
- Quality Assurance Engineer (QA)
- Business Analyst (BA)

Your response must include:
1. The best-fit role (SE, QA, or BA)
2. 2-3 traits from the input that support the match
3. A brief explanation in 2–3 sentences

Respond in this exact format:

[Recommended Role]: <SE / QA / BA>
[Supporting Traits]: <trait1>, <trait2>, ...
[Reasoning]: <short explanation>
"""
    )

    # Create an LLMChain that combines the LLM and the prompt template.
    chain = LLMChain(llm=llm, prompt=prompt_template)
    print("LLM chain initialized.")

# --- API Endpoints ---

@app.get("/")
async def root():
    """
    Root endpoint to confirm the backend is running.
    """
    return {"message": "Job Recommendation Chatbot Backend is running!"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Receives user input via a POST request and returns a job recommendation.
    """
    # Ensure that the vector_db and chain are initialized before processing requests.
    if not vector_db or not chain:
        print("ERROR: Backend not fully initialized. Vector DB or LLM chain is missing.")
        raise HTTPException(status_code=503, detail="Backend not fully initialized. Please wait a moment or check server logs.")

    user_input = request.user_input # Get the user's input from the request body
    print(f"Received user input for recommendation: '{user_input}'")

    try:
        # Get similar job documents from the vector database based on user input.
        docs = vector_db.similarity_search(user_input, k=5)
        # Combine the content of the retrieved documents to form the context for the LLM.
        context = "\n\n".join([doc.page_content for doc in docs])
        print(f"Generated context from vector DB (first 200 chars): {context[:200]}...")

        # Run the LLM chain with the user's traits and the generated context.
        result = chain.run(traits=user_input, context=context)
        print(f"Generated recommendation: {result}")

        # Return the recommendation in a JSON response.
        return {"recommendation": result}
    except Exception as e:
        print(f"ERROR: An error occurred during recommendation generation: {e}")
        # Return a 500 Internal Server Error if something goes wrong during processing.
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")