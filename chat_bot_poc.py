                                                               # Sample code For Review Purpose (Chat/ CALL Bot)
 # Import necessary libraries for the functionality
import asyncio  # Asynchronous programming for concurrent tasks
import threading  # For running tasks in separate threads
from PyPDF2 import PdfReader  # To read and extract text from PDF files
from langchain.embeddings import HuggingFaceEmbeddings  # For generating text embeddings using HuggingFace models
from langchain.text_splitter import CharacterTextSplitter  # To split large text into smaller chunks
from langchain.vectorstores import FAISS  # For storing embeddings in a vector database for fast similarity search
from langchain.chains.question_answering import load_qa_chain  # To load a question-answering model
from langchain import HuggingFaceHub  # To access models hosted on HuggingFace Hub
import speech_recognition as sr  # For recognizing speech input from a microphone
import pyttsx3  # For text-to-speech conversion
import os  # For environment variable management

# Set up HuggingFace API Key (required to access models from HuggingFace)
os.environ["HUGGINGFACE_API_KEY"] = "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"

# Path to the PDF document containing the text (e.g., a cold-calling script)
pdf_path = 'Cold-Calling-Script.pdf'

# Initialize the PDF reader and extract text from each page
reader = PdfReader(pdf_path)
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()  # Extract the text from the current page
    if text:
        raw_text += text  # Append the extracted text to the raw text

# Initialize the text splitter to split the text into smaller chunks for processing
text_splitter = CharacterTextSplitter(
    separator="\n",  # Split text based on newlines
    chunk_size=1000,  # Define the chunk size (maximum number of characters per chunk)
    chunk_overlap=0,  # Define the overlap between chunks
    length_function=len,  # Length function to count characters
)

# Split the raw text into smaller chunks using the splitter
texts = text_splitter.split_text(raw_text)

# Load HuggingFace Embeddings (pre-trained model to convert text into embeddings)
embeddings = HuggingFaceEmbeddings()

# Use FAISS (a library for fast similarity search) to create a vector store of the text chunks
db = FAISS.from_texts(texts, embeddings)

# Load the question-answering model from HuggingFace Hub
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # The model ID on HuggingFace Hub
    model_kwargs={"temperature": 0.5, "max_length": 512},  # Set parameters for the model (e.g., temperature, max length)
    huggingfacehub_api_token="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",  # API token for authentication
)

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Use threading to manage interruptions during speaking
interrupt_event = threading.Event()

# Initialize the text-to-speech engine
engine = pyttsx3.init()


# Function to recognize speech using the microphone
def recognize_speech():
    with sr.Microphone() as source:
        print("Say something:")  # Prompt user to speak
        recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        try:
            # Listen for speech input and recognize it using Google's speech API
            audio = recognizer.listen(source, timeout=5)
            print("Recognizing...")
            text = recognizer.recognize_google(audio)  # Convert speech to text using Google's API
            print("You said:", text)  # Print recognized text
            return text  # Return the recognized text
        except sr.UnknownValueError:
            print("Not understood")  # Handle cases where speech was not recognized
            return None
        except sr.RequestError as e:
            print(f"Error in recognizing speech: {e}")  # Handle API errors
            return None
        except TimeoutError:
            print("Timeout. No speech detected.")  # Handle timeout when no speech is detected
            return None


# Function to speak text synchronously (blocking function)
def speak_sync(text, stop_event):
    engine.say(text)  # Convert the provided text into speech
    engine.runAndWait()  # Wait until the speech has finished playing
    
    # Check if stop event is set to interrupt the speaking
    while not stop_event.is_set():
        pass  # Keep looping until the event is set or speaking completes
    engine.stop()  # Stop the ongoing speech


# Async function to listen for speech while speaking asynchronously
async def listen_while_speaking_async(stop_event):
    global interrupt_event
    while not interrupt_event.is_set():  # Keep listening until interrupted
        user_input = recognize_speech()  # Recognize user speech
        if user_input:
            print("You interrupted with:", user_input)
            # Use the question-answering chain to find the answer based on recognized speech
            chain = load_qa_chain(llm, chain_type="stuff")
            query = user_input
            docs = db.similarity_search(query)  # Find relevant documents from the vector store
            result = chain.run(input_documents=docs, question=query)  # Run the QA chain to get the result
            
            # Speak the result
            speak_sync(result, stop_event)
            print("Listening again...")  # Prepare to listen for the next input
            
            stop_event.set()  # Set the stop event to stop speaking
            interrupt_event.set()  # Set the interrupt event to end the listening loop
            break


# Main asynchronous function to manage speech recognition and question answering
async def main():
    global interrupt_event
    while True:
        user_input = recognize_speech()  # Recognize speech from the user

        if user_input:
            response = f"You said: {user_input}"  # Respond back to the user with what they said
            # Use the question-answering chain to find an appropriate answer to the user's input
            chain = load_qa_chain(llm, chain_type="stuff")
            query = user_input
            docs = db.similarity_search(query)  # Retrieve relevant documents from the vector database
            result = chain.run(input_documents=docs, question=query)  # Run the QA chain to get the answer
            
            # Create a new thread to speak the answer asynchronously
            stop_event = threading.Event()  # Event to stop speaking when needed
            speak_thread = threading.Thread(target=speak_sync, args=(result, stop_event))  # Start a new thread for speaking
            speak_thread.start()

            interrupt_event.clear()  # Clear the interrupt event so that we can continue listening

            # Start an asynchronous task to listen while speaking
            await listen_while_speaking_async(stop_event)

            # Stop the speaking thread once the listening is done
            stop_event.set()
            speak_thread.join()  # Wait for the speaking thread to complete

            # Add a short delay before starting the next cycle
            await asyncio.sleep(1)


# Run the main function asynchronously if the script is executed
if __name__ == "__main__":
    asyncio.run(main())
