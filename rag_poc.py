# Import required libraries
import PyPDF2  # For extracting text from PDF files
from sentence_transformers import SentenceTransformer  # For generating text embeddings
import faiss  # For building a similarity search index (FAISS)
import numpy as np  # For handling arrays and mathematical operations
from fastapi import FastAPI, Query  # FastAPI to create a web API and handle HTTP requests

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """
    This function reads the PDF file at the specified path and extracts the text from it.
    It processes all pages in the PDF and concatenates the text into one string.
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)  # Use PyPDF2 to read the PDF file
        text = ""  # Initialize an empty string to store the extracted text
        
        # Iterate through each page of the PDF and extract the text
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()  # Concatenate the extracted text from each page
    return text  # Return the concatenated text from all pages of the PDF

# Function to split the text into smaller chunks and embed them using the sentence transformer model
def embed_text_chunks(text, model):
    """
    This function splits the input text into chunks (512 characters each) and then generates 
    embeddings for these chunks using a pre-trained SentenceTransformer model.
    """
    text_chunks = [text[i:i+512] for i in range(0, len(text), 512)]  # Split the text into 512-character chunks
    embeddings = model.encode(text_chunks)  # Generate embeddings for each chunk using the SentenceTransformer model
    return text_chunks, embeddings  # Return the chunks and their corresponding embeddings

# Function to build a FAISS index from the text embeddings
def build_faiss_index(embeddings):
    """
    This function takes the embeddings of text chunks and creates a FAISS index for fast similarity search.
    """
    dimension = embeddings.shape[1]  # The dimension of each embedding vector (e.g., 384 for the 'paraphrase-MiniLM-L6-v2' model)
    faiss_index = faiss.IndexFlatL2(dimension)  # Initialize a FAISS index using the L2 distance (Euclidean distance)
    faiss_index.add(np.array(embeddings))  # Add the embeddings to the FAISS index
    
    return faiss_index  # Return the FAISS index containing the embeddings

# Function to handle queries and retrieve relevant text chunks from the FAISS index
def query_rag_system(query, model, faiss_index, text_chunks):
    """
    This function receives a query, generates its embedding, performs a similarity search on the FAISS index, 
    and retrieves the most relevant text chunks from the indexed documents.
    """
    query_embedding = model.encode([query])  # Generate the embedding for the query using the model
    
    # Perform a similarity search in the FAISS index, retrieving the top 5 most similar chunks
    distances, indices = faiss_index.search(np.array(query_embedding), k=5)
    
    # Retrieve the corresponding text chunks based on the indices returned by the search
    retrieved_text = [text_chunks[i] for i in indices[0]]
    
    # Join the retrieved text chunks into a single string to provide context for the response
    context = " ".join(retrieved_text)
    
    return context  # Return the combined context of the most relevant chunks

# Initialize a FastAPI application
app = FastAPI()

# Load a pre-trained SentenceTransformer model for embedding text
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Use a lightweight model optimized for paraphrase identification

# Extract text from the PDF document, generate embeddings, and build a FAISS index
pdf_text = extract_text_from_pdf("thinkstats2.pdf")  # Extract text from the provided PDF
text_chunks, embeddings = embed_text_chunks(pdf_text, model)  # Split the text into chunks and generate embeddings
faiss_index = build_faiss_index(embeddings)  # Build the FAISS index from the embeddings

# Define a FastAPI endpoint to handle RAG (Retrieval-Augmented Generation) queries
@app.get("/rag")
def rag_query(query: str = Query(..., description="Your question")):
    """
    This endpoint receives a user's query, processes it using the RAG system (using FAISS for retrieval), 
    and returns the most relevant context from the document.
    """
    response = query_rag_system(query, model, faiss_index, text_chunks)  # Use the RAG system to get relevant context
    
    # Return the response in a JSON format
    return {"response": response}

# Entry point for running the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Run the FastAPI app on all interfaces (0.0.0.0) and port 8000
