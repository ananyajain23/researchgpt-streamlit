print("App starting...")

######################### Project Title: ResearchGPT – Research Paper RAG Chatbot with AgenticAI  #########################

#Description:

# ResearchGPT is a Streamlit-based AI chatbot that allows users to upload research papers in PDF format and interactively query them 
# using natural language. It combines OCR, table parsing, and document chunking with advanced retrieval-augmented generation 
# (RAG) to answer complex research paper related questions.

#Key components include:

# 1. PDF Parsing Agent: Extracts multi-column text and tables using pdfplumber.

# 2. OCR Agent: Extracts embedded text from medical images (e.g., charts or scans) using Tesseract OCR.

# 3. Vectorstore Agent: Embeds and indexes the extracted content using OpenAI embeddings + ChromaDB.

# 4. QA Agent: Answers user queries using GPT-4o-mini, grounded in the indexed data.

# 5. User Interface: Built in Streamlit for intuitive document upload, indexing, and chat.

#The system/code supports:

# 1. Text and table extraction from scanned or structured PDFs

# 2. Image-based OCR from charts, graphs, and figures

# 3. Context-aware question answering grounded in real content (no hallucination)

# 4. Smart rendering of answers including tables and relevant images

# 5. Streamlit for UI:
# ✅ Users can upload multiple research paper PDFs for automated text, table, and image (OCR) extraction.
# ✅ Dynamic progress indicators for each agent: PDF parsing, indexing, and OCR scanning.
# ✅ Interactive chat interface with real-time QA from indexed content using LangChain agents (includes preserving chat history so 
# assistant can provide context-aware, grounded answers)
# ✅ Tracks and displays assistant response time for transparency and performance feedback.
# ✅ Sidebar includes a chat history viewer with a slider to control how many recent messages to show.
# ✅ Clean, responsive layout with conditional rendering based on app state.

#--------------------------------------------------------------------------------------------------------------------------------------

#Enhancements made to the QAAgent and chat interface to support conversational context:

#- Incorporated chat history into the QA chain inputs, enabling the model to consider prior user-assistant exchanges.
#- Updated the prompt template to explicitly include chat history along with retrieved documents, guiding the model to provide 
# context-aware, grounded answers.
#- Added graceful handling of empty or missing chat history by defaulting to 'No prior conversation.'
#- Implemented logging of user and assistant messages with timestamps for persistent chat history tracking.
#- Improved user experience by displaying past chat messages and maintaining conversation continuity across multiple questions.
#- Enabled conditional rendering of answers (tables, images) for richer response formats.

# Updated streamlit UI to have agents and uploading,indexing,OCR on sidebar with option of viewing chat history with a slider to choose 
# how many interactions user wants to see. Done this so that center of screen is just for conversation between user and assistant.

# Overall, these changes turn a simple question-answer system into a multi-turn, context-aware conversational agent that provides more 
# coherent and accurate answers based on both the uploaded document content and the ongoing dialogue.

#---------------------------------------------------------------------------------------------------------------------------------------

import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
#tempfile is used to:
# In chatbot applications — especially those that allow file uploads (like PDFs, images, etc.) — we often use the tempfile 
# module for temporary file storage.
# When users upload files (e.g., PDFs in your MedTalk app), they arrive as in-memory objects (like BytesIO).
# However, many tools (like pdfplumber, PyMuPDF, Tesseract, etc.) expect a file path, not an in-memory stream.
# So we temporarily save the uploaded file to disk using tempfile, then pass that file path to the processing tool.


import pytesseract
#a Python wrapper for Tesseract OCR (Optical Character Recognition) used for image text extraction

import pandas as pd
import io

from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# === [ENV VARS AND SETUP] ===


load_dotenv() # Load API keys and secrets from .env file
api_key = os.getenv("OPENAI_API_KEY")

#Below three lines initializes keys in Streamlit’s st.session_state dictionary with default values if they don’t exist yet.
#st.session_state is a special dict that preserves state across user interactions and reruns in Streamlit apps.
st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("vectorstore_built", False)
st.session_state.setdefault("image_ocr_done", False)


import os
import platform
import pytesseract

# Only set tesseract path if running on Windows
if platform.system() == "Windows":
    tesseract_dir = r"D:\GENAI COURSE LEARNBAY\Capstone Projects\ResearchGPT - A RAG-Based Chatbot for Research Papers\Tesseract-OCR"
    poppler_bin = r"D:\GENAI COURSE LEARNBAY\Capstone Projects\ResearchGPT - A RAG-Based Chatbot for Research Papers\poppler-24.08.0\Library\bin"
    os.environ["PATH"] += os.pathsep + poppler_bin + os.pathsep + tesseract_dir
    pytesseract.pytesseract.tesseract_cmd = os.path.join(tesseract_dir, "tesseract.exe")


# Create folder to store extracted images if it doesn't exist
os.makedirs("figures", exist_ok=True)
#Creates a directory (folder) named "figures" in the current working directory.
#If the folder "figures" already exists, it does nothing and does not raise an error because of exist_ok=True.


# === [AGENTS] ===

import pdfplumber
# pdfplumber helps parse PDFs, including complex layouts and tables
#it is one of the best open-source tools for extracting structured tables from PDFs.

#Why are we using classes? 
#You can create different agents with different splitters or column settings:

#agent1 = FastPDFProcessorAgent(splitter1, num_columns=2)
#agent2 = FastPDFProcessorAgent(splitter2, num_columns=1)
#Each one behaves differently but uses the same code. This is a big strength of using classes in Python.

# I first used UnstructuredPDFLoader however that was quite slow as it is accurate but slow
#| Issue                                  | Explanation                                                                                            |
#| -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
#| ❌ **Poor column handling**             | It reads PDFs linearly. Multi-column text might get jumbled.                                           |
#| ❌ **Less control over page layout**    | You can’t easily crop pages, divide into columns, or tune tolerances like with `pdfplumber`.           |
#| ❌ **Heuristic-based**                  | Uses ML and heuristics, which are **less predictable** and **harder to debug** when extraction fails.  |
#| ⚠️ **Table extraction is hit or miss** | It might recognize tables visually but won't always extract them cleanly into structured rows/columns. |
#| ❌ **No fine-tuning of bounding boxes** | You can’t manually define or extract specific regions (like you can with pdfplumber or PyMuPDF).       |

# I used PyMuPDFLoader as well, but pdfplumber proved more advanced — it was better at extracting text from multi-column PDFs 
# (needed for the research paper project as the research papers (and even magazines) have multicolumn text.)

class FastPDFProcessorAgent:

    """
    Agent to parse PDFs extracting multi-column text and tables.
    Uses pdfplumber to handle page layout, columns, and tables.
    Reads PDFs using pdfplumber.
    Splits the text into smaller chunks using a splitter (RecursiveCharacterTextSplitter from LangChain).
    Returns a list of both text chunks and tables in a structured way.
    """

    def __init__(self, splitter,num_columns=2):

        """
        This function is automatically called when you create a new object from FastPDFProcessorAgent.
        It is a constructor or setup step.
        :param splitter: LangChain text splitter to chunk extracted text
        :param num_columns: Number of columns expected in PDF pages. if not provided, then default to 2.
        Wrote 2 as default as the pdf's provided (along with research paper project) have 2 columns only per page.
        """
        self.splitter = splitter
        self.num_columns = num_columns


    def extract_tables(self, pdf_path):

        """
        Extracts tables from each PDF page using pdfplumber's table extraction.
        Converts tables to tab-separated text and stores as Document objects.
        Skips empty or invalid tables.
        """

        # Create an empty list where we’ll store all the extracted tables (as Document objects).
        table_docs = []

        with pdfplumber.open(pdf_path) as pdf: # Opens the PDF file using pdfplumber, with ensures the file gets properly closed after processing, pdf now represents the entire document.
            for i, page in enumerate(pdf.pages): # looping/going through each page and extracting tables, converting those tables to clean, tab seperated strings.
                tables = page.extract_tables() # pdfplumber tries to detect tables on the page using its internal logic.
                                               # Returns a list of 2D lists, where each table is a list of rows → and each row is a list of cells.

                for t in tables:

                    if t:         #skips tables that are empty or none

                        # converts each row of the table into a line of tab-separated text (\t). 
                        # Loop through each row in the table t. For each row: Loop through each cell
                        # If the cell is None, replace it with an empty string "" to avoid errors.
                        # Join all cells in the row with a tab (\t)
                        # Finally, join all the rows together with a newline (\n)
                        # Example of converted table: 
                        # If t = [['Name', 'Age', 'City'], ['John', '30', 'New York']]
                        # Output: 
                        # Name\tAge\tCity
                        # John\t30\tNew York

                        table_text = "\n".join(["\t".join([cell if cell is not None else "" for cell in row]) for row in t if row])
                        
                        #✅ Perfect to avoid the TypeError from NoneType.

                        if table_text.strip(): #Make sure we’re not adding empty tables (just in case).
                            table_docs.append(Document(
                                page_content=table_text.strip(),
                                metadata={
                                    "type": "table",
                                    "source": "pdfplumber",
                                    "page": i + 1 # saving page number too so that if user asks where to find the table in pdf, it can get page number.
                                                  # (i + 1 because page numbers start from 1, not 0)
                                }
                            ))

        return table_docs
        #Return a list of these extracted tables
    

    def extract_text_by_columns(self, pdf_path):

        """
        Extracts text from PDF pages by dividing each page into vertical columns.
        Extracts text column-wise (left to right), preserving reading order for multi-column layouts.
        Skips empty columns.
        """

        docs = []

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages): #i is the page number (starts at 0), page is a pdfplumber object representing that page
                page_width = page.width
                page_height = page.height
                column_width = page_width / self.num_columns


                # Extract text from each column bounding box
                page_text = "" # This variable will store the combined text from all columns of the current page.


                for col in range(self.num_columns): #loop through each column
                    left = col * column_width
                    right = left + column_width
                    #left and right calculates where this column starts and ends on the page
                    # For 2 columns:
                    # Column 0: left = 0, right = 500
                    # Column 1: left = 500, right = 1000


                    # Crop page to current column bounding box
                    # This tells pdfplumber: “Only look inside this rectangle” (i.e., just this column)
                    column_crop = page.within_bbox((left, 0, right, page_height)) 

                    # Actually pulls out the text just inside the cropped column area
                    # Extract text with tolerance for cleaner extraction
                    # x_tolerance and y_tolerance: Help merge nearby characters or words that are slightly offset — improving extraction quality
                    col_text = column_crop.extract_text(x_tolerance=3, y_tolerance=3)


                    if col_text:
                        page_text += col_text.strip() + "\n\n"
                    # If the column has text, strip extra spaces and add to the page_text
                    # Adds double newline \n\n to separate columns clearly

    
                if page_text.strip(): #If anything was extracted from the page, wrap it into a Document
                    docs.append(Document(
                        page_content=page_text.strip(),
                        metadata={"type": "text", "page": i + 1}
                    ))

        return docs
    
    # above function looks good: splitting pages into equal vertical slices, extracting text from each slice, combining left-to-right.
    # Correctly skip empty text slices.


    def process(self, pdf_path):

        """
        Orchestrates extraction of multi-column text and tables.
        Splits text documents into chunks for embedding.
        Returns combined list of text chunks and table documents.
        """

        # Extract multi-column text
        text_docs = self.extract_text_by_columns(pdf_path)
        split_text_docs = self.splitter.split_documents(text_docs)

        # Extract tables normally
        table_docs = self.extract_tables(pdf_path)

        return split_text_docs + table_docs
        # combine the extracted text and tables into a single list. 
        # Example of output of process(): 
        # [
        # Document(page_content="Text chunk 1", metadata={"type": "text", "page": 1}),
        # Document(page_content="Text chunk 2", metadata={"type": "text", "page": 1}),
        # Document(page_content="Name\tAge\tCity\nJohn\t30\tNY", metadata={"type": "table", "page": 3}),
        # ...
        # ]

# hence our table and texts have page numbers in their metadata.


class ImageOCRAgent:

    """
    This is an agent (i.e., a modular component) responsible for:

    1. Scanning .jpg images from a given directory (./figures)

    2. Running OCR (Optical Character Recognition) using Tesseract

    3. Creating LangChain Document objects with the extracted text

    4. Adding these documents to your vector store for retrieval and question answering later
    """

    def extract_text_from_images(self, image_dir, min_chars=30):
        if not os.path.exists(image_dir): return []
        docs = []
        for img_file in sorted(os.listdir(image_dir)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")): continue
            img_path = os.path.join(image_dir, img_file)
            txt = pytesseract.image_to_string(img_path).strip()
            if len(txt) < min_chars: continue
            # 🆕 Add prefix for clarity
            pg = int(re.search(r"page(\d+)_img\d+", img_file).group(1)) if re.search(r"page(\d+)_img\d+", img_file) else None
            full_txt = f"[Image Page {pg}] {txt}" if pg else txt
            docs.append(Document(page_content=full_txt,
                                 metadata={"type":"image", "image_name":img_file, "page": pg}))
        return docs
    

class VectorstoreAgent:

    """
    Agent managing the vectorstore index that holds embedded documents.
    Uses OpenAI embeddings and ChromaDB for persistent vector storage.
    Adds documents to the store in batches, persists the vectorstore to disk so it can be reused between sessions

    """

    def __init__(self, persist_dir="chroma_index_agenticai"): #(Constructor)

        """
        When you create an instance of VectorstoreAgent, this code runs automatically.
        It sets up:
        1. A local directory to store your persistent ChromaDB index
        2. An embedding model (from OpenAI)
        3. A Chroma vectorstore instance
        """

        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True) #ensures the directory exists (creates it if not).

        # Initialize OpenAI embeddings model for vectorization
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Initialize Chroma vectorstore with embedding function and persistence directory
        self.vectorstore = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )


    def build_index(self, documents):

        """
        1. Takes in a list of LangChain Document objects
        2. Converts their page_content into embeddings
        3. Adds them to Chroma in batches of 50
        4. Saves the vectorstore permanently
        5. Returns the vectorstore so it can be used later (for retrieval)
        """

        for i in range(0, len(documents), 50):
            self.vectorstore.add_documents(documents[i:i + 50])
        self.vectorstore.persist()
        return self.vectorstore


class QAAgent:

    """
    RAG Agent to handle user queries.
    Uses a retriever from vectorstore to get relevant documents,
    and feeds into a GPT-4o-mini chain to answer questions grounded in those documents (i.e., based only on the retrieved docs — not hallucinated)
    """

    def __init__(self, vectorstore):

        # Initialize retriever with top-k (15) document retrieval using cosine similarity 
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        self.chain = self._build_chain()

    def _build_chain(self):

        # Prompt template instructs the model to only answer based on provided context, avoiding hallucination
        prompt = PromptTemplate.from_template("""
You are an analyst tasked with understanding detailed information and trends
from text documents, data tables, and OCR-extracted text from charts and graphs.
Use this information to provide answers related to the user question.
Do not make up answers. Use the provided context documents and chat history to answer. 
If you don't know the answer, just say you don't know.

{context}
                                              
Chat history:
{chat_history}                                              

User question:
{question}
""")
        # The chain uses the retriever to get relevant docs, formats them, then passes to ChatOpenAI with zero temperature for 
        # deterministic answers
        return (
            {
                "context": lambda x: format_docs(self.retriever.get_relevant_documents(x["question"])),
                # Expect 'chat_history' key in input dict (string of recent convo)
                "chat_history": lambda x: x.get("chat_history", "No prior conversation."),
                "question": lambda x: x["question"]
            }   #this step expands one input question into two variables for the prompt: 
                # 1. context: a string containing the concatenated relevant documents
                # 2. question: the raw user question string
            | prompt
            | ChatOpenAI(model="gpt-4o-mini", temperature=0)
            | StrOutputParser()
        )

    def answer(self, question, chat_history=None):

        # Invokes the QA chain with question string and optional chat_history string.
        # If chat_history is not provided, defaults to "No prior conversation."

        if chat_history is None:
            chat_history = "No prior conversation."

        inputs = {
            "question": question,
            "chat_history": chat_history
        }

        return self.chain.invoke(inputs)


# === [HELPERS] ===

#Sometimes the answers from the AI might include tabular data — for example:

#- Summaries of extracted data
#- Statistical results
#- OCR data converted into tables
#- Structured info like rows & columns

#This tabular data can be hard to read if shown as plain text (with tabs and newlines), so you want to display it visually as a table.


#What problem does try_render_table solve?

#- It automatically detects whether the AI’s text response contains a table formatted as tab-separated values.

#- If it does, it renders the response as a nice table in Streamlit (instead of showing raw, messy text).

#- If the response is not tabular or parsing fails, it falls back gracefully to rendering plain text.

#This gives a better user experience because:

#- Users get a clean, readable table instead of raw tab-separated text.

#- You avoid errors/crashes trying to render non-table text as a table.

#- Your UI adapts dynamically depending on the type of content returned by the AI.



def try_render_table(response: str):

    """
    1. It tries to interpret a string response as a table formatted with tab-separated values (TSV).

    2. If the response looks like a valid TSV table, it renders it nicely as a table in Streamlit.

    3. If it fails (e.g., response isn't a table or parsing fails), it returns False so you know to fall back to showing plain text instead.
    """

    lines = [line for line in response.strip().split("\n") if line]

    if not lines or len(lines) < 2 or '\t' not in lines[0]:
        #If no lines (not lines), it's empty, so not a table → return False.
        # If fewer than 2 lines, there's no header + data rows → return False.
        # If the first line (usually the header) does not contain a tab character, it's unlikely to be a TSV table → return False.
        return False


    try:

        #Joins the filtered lines back together into one string, separated by newlines.
        # Wraps this string with io.StringIO, which turns a string into a file-like object.
        # Use StringIO to treat string as file-like object for pandas.read_csv as pandas.read_csv needs a file-like object or file-path.

        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t")

        #io.StringIO lets you treat a string as a file object.
        # This means you can pass text data (like your table content as a string) to pandas.read_csv() without needing an actual file on disk.
        # It’s great for converting strings into file-like streams that libraries expect.

        st.table(df)
        return True
    
    except Exception:
        return False


def format_docs(docs):

    """
    Formats a list of Document objects into a single string block for the prompt.
    Adds labels based on document type (text, table, image) and relevant metadata.
    """

    chunks = [] #Initialize an empty list chunks to hold formatted strings for each document.

    for doc in docs: #docs are a list of document objects
        doc_type = doc.metadata.get("type", "unknown") #If no "type" key exists, default to "unknown".

        label = f"[{doc_type.upper()}]" #Create a label by converting the document type to uppercase and surrounding it with square brackets.
                                        #Example: if doc_type is "text", label becomes "[TEXT]".

        # Add page number if available
        page_num = doc.metadata.get("page")
        if page_num is not None:
            label += f" (Page {page_num})"

        if doc_type == "image":
            label += f" (Image name: {doc.metadata.get('image_name', 'unknown')})"
            # If the document is an image, add extra info to the label:
            # Append the image filename from metadata if available. If no image name is found, use "unknown" as fallback.
            # So for an image document, label might become: [IMAGE] (Image name: chart1.jpg)

        chunks.append(f"{label}:\n{doc.page_content}")
        #Append a formatted string to chunks containing:
        #The label followed by a colon and newline
        #The document's textual content (page_content)

    return "\n\n".join(chunks)


@st.cache_data(show_spinner=False)

#This is a Streamlit caching decorator.
#✅ Purpose: Caches the output of this function so that if called again with the same input, it doesn’t rerun but returns the result instantly 
# from memory or disk.

#The leading underscore _splitter disables hashing on that argument, as you intended.
# The underscore _splitter is a trick to tell Streamlit "do not consider this argument when checking the cache" because many objects 
# (like tokenizers, splitters) are not hashable, trying to cache on them would cause an error.
# Streamlit ignores arguments that start with _ for cache hashing.

def process_pdf_cached(_splitter, pdf_path):

    """
    Cache-optimized function to process PDF via FastPDFProcessorAgent.
    Avoids re-processing unchanged PDFs for performance.
    """
    agent = FastPDFProcessorAgent(_splitter)
    return agent.process(pdf_path) #Returns a list of Document objects 


def save_uploaded_files(uploaded_files):

    """
    Saves uploaded Streamlit files to temporary disk files.
    Returns list of temp file paths for further processing.
    - **`uploaded_files`**: a list of file objects that a user has uploaded via `st.file_uploader(...)`.
    - Each object in this list is of type `UploadedFile` (Streamlit class), which behaves like a file but isn't stored on disk yet.
    """

    temp_paths = [] #Initializes an empty list to store the file paths of the saved temporary files.

    for uploaded_file in uploaded_files:    # Loops through each uploaded file to save them one by one.

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode="wb") as tfile:

            tfile.write(uploaded_file.getbuffer())
            temp_paths.append(tfile.name)
            
            #✅ Creates a temporary file on disk using Python’s tempfile module.

            #delete=False: ensures the file is not deleted when closed, so it can be accessed later.

            #suffix=".pdf": appends .pdf to the filename to retain the correct file extension.

            #mode="wb": opens the file in write-binary mode, required for writing PDF bytes.

    return temp_paths


#Extract all images embedded inside the pages of a PDF file and save them as separate image files (JPEGs) into a specified 
# output directory.

import fitz  # PyMuPDF
import os
from PIL import Image
import io

def extract_images_from_pdf(pdf_path, output_dir="figures"): #pdf_path is the path of pdf from which it's seeing images and output_dir is where it'll save the extracted images

    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path) #fitz is the alias for the PyMuPDF library, fitz.open(pdf_path) opens the PDF document for reading.
    saved_xrefs = set()  # prevent duplicate saves

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True) #Extracts all images from the page, full=True ensures it returns complete image metadata, not just inline references

        for img_index, img in enumerate(image_list, start=1): #Loops through each image found on the current page, img_index is used for naming files.
            
            xref = img[0] #xref: a unique reference number to the image object inside the PDF.
            if xref in saved_xrefs:
                continue  # skip duplicates
            saved_xrefs.add(xref)
            base_image = doc.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"]))
            filename = f"page{page_num + 1}_img{img_index}.{base_image['ext']}"
            image.convert("RGB").save(os.path.join(output_dir, filename), "JPEG")

    print(f"Extracted images saved to {output_dir}")



# === Helper Function: log_chat ===
# This function logs every message (both user and assistant) to a local text file.
# It helps in keeping a persistent chat history outside the Streamlit session for later review.
# Each entry is saved as a JSON object on a new line, and includes:
# - The role of the speaker: either "user" or "assistant"
# - The message content
# - A timestamp when the message was logged
# This file ("chat_history_log.txt") acts as a conversation archive and can be shared or analyzed later.

import json
from datetime import datetime

CHAT_LOG_FILE = "chat_history_log.txt"

def log_chat(role, message):
    """
    Save chat messages to a file with timestamp, role (user/assistant), content, and optional metadata.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "message": message,
    }

    with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


# critical to rebuild the in-memory chat history (st.session_state.chat_history) when the app restarts.
def load_chat_history():
    if not os.path.exists(CHAT_LOG_FILE):
        return []

    history = []
    with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                history.append({"role": entry["role"], "content": entry["message"]})
            except json.JSONDecodeError:
                continue  # skip broken lines

    return history



#summarize the last few user–assistant interactions into a clean, structured text format, which is especially useful for feeding 
# into a prompt for an LLM (like in RAG or conversational agents). below function is just to be fed as context in prompt for llm.

def format_chat_history(chat_history, max_turns=10): #max_turns=10: Limits the number of past interactions that go as context to model in prompt.

    # Take last max_turns * 2 messages (user + assistant)
    recent = chat_history[-(max_turns * 2):] if chat_history else [] #If chat_history is empty, use an empty list.

    pairs = []
    for i in range(0, len(recent), 2): #Loop through the recent messages in steps of 2 — because one interaction is usually a pair: user + assistant.
        
        user_msg = recent[i]["content"] if i < len(recent) else ""
        assistant_msg = recent[i + 1]["content"] if (i + 1) < len(recent) else ""

        #Safely extract the content of the user and assistant messages.
        # If the list ends on a user message without an assistant response, it still works safely.

        pairs.append(f"User: {user_msg}\nAssistant: {assistant_msg}")
    return "\n".join(pairs) if pairs else "No prior conversation." #If pairs is empty, return "No prior conversation."


# Feature	               In-Memory (UI)	            Persistent Log File
# Chat display	           Current session only	        All sessions
# Chat history toggle	   Current session only	        N/A
# On app restart	       Reloads if coded	            Always grows

# I want entire history is saved in my txt file of all sessions and for displaying on streamlit ui when user will use slider it shows 
# only of current session, code is handling that. 🎉🥳


# === [STREAMLIT UI] ===


import os
import time  # for response time tracking
import pytesseract
import streamlit as st
from langchain.schema import Document

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


#start_building_index() is a simple callback function used with a Streamlit button.
#It sets a flag (building_index) in st.session_state to trigger the next processing step.
#Streamlit doesn’t immediately re-render on state change unless triggered by UI events — this sets up the session for the next re-render.

def start_building_index():
    st.session_state.building_index = True

# Initialize text splitter for chunking extracted text for embedding
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)


# Instantiate agents with necessary parameters
pdf_agent = FastPDFProcessorAgent(splitter, num_columns=2)  # Change to 3 if needed
image_agent = ImageOCRAgent()
vector_agent = VectorstoreAgent()


# Set Streamlit page config and header
st.set_page_config(page_title="ResearchGPT - Chatbot", page_icon="📄")
st.title("📄 ResearchGPT – Research Paper RAG Chatbot with AgenticAI 🛠️")
st.markdown("🔍 Upload your research papers and chat 💬 with them. Extracts **text**, **tables**, and **image text via OCR**.")


# Sidebar: Upload PDFs + Chat History Toggle
with st.sidebar:
    st.header("📥 Upload & Process PDFs")
    uploaded_files = st.file_uploader("Upload Research Paper PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        pdf_files = save_uploaded_files(uploaded_files)
        st.success(f"✅ Uploaded {len(pdf_files)} PDF file(s)")
    else:
        pdf_files = []
        st.info("📎 Upload PDF files to begin.")

    if pdf_files:
        if "building_index" not in st.session_state:
            st.session_state.building_index = False

        if not st.session_state.building_index:
            st.button("📚 Proceed", on_click=start_building_index)
        else:
            st.button("📚 Proceed", disabled=True)

            status_placeholder = st.empty()

            with st.spinner("🚀 Running all agents..."):

                # ================== PDF AGENT =====================
                status_placeholder.info("📄 PDF Agent: Extracting text and tables...")
                all_chunks = []
                progress_pdf = st.progress(0, text="Processing PDFs...")
                for i, pdf in enumerate(pdf_files):
                    st.sidebar.write(f"📂 Processing File {i+1}/{len(pdf_files)}")
                    chunks = process_pdf_cached(splitter, pdf)
                    #st.sidebar.info(f"📄 {os.path.basename(pdf)} yielded {len(chunks)} chunks")
                    all_chunks.extend(chunks)
                    progress_pdf.progress((i + 1) / len(pdf_files), text=f"✅ Processed {i + 1}/{len(pdf_files)} PDF(s)")
                progress_pdf.empty()
                status_placeholder.success("✅ PDF Agent complete.")

                # ✅ Extract images for OCR
                status_placeholder.info("🖼️ Extracting images for OCR...")
                for pdf_path in pdf_files:
                    extract_images_from_pdf(pdf_path, output_dir="figures")

                # ================== OCR AGENT =====================
                status_placeholder.info("🧠 OCR Agent: Scanning images...")
                ocr_docs = []
                image_dir = "./figures"
                image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])
                total_images = len(image_files)
                progress_ocr = st.progress(0, text=f"🖼️ Scanning 0 of {total_images} images...")

                import re

                for i, img_file in enumerate(image_files):
                    img_path = os.path.join(image_dir, img_file)
                    match = re.search(r"page(\d+)_img\d+", img_file)
                    page_number = int(match.group(1)) if match else None
                    try:
                        text = pytesseract.image_to_string(img_path).strip()
                        if len(text) >= 30:
                            # ✅ Prefix OCR text with [Image Page X]
                            prefixed_text = f"[Image Page {page_number}] {text}" if page_number else text
                            doc = Document(
                                page_content=prefixed_text,
                                metadata={
                                    "type": "image",
                                    "image_name": img_file,
                                    "page": page_number,
                                    "source": "OCR",
                                    "agent": "ImageOCRAgent"
                                }
                            )
                            ocr_docs.append(doc)
                    except Exception as e:
                        st.warning(f"OCR failed on {img_file}: {e}")
                    progress_ocr.progress((i + 1) / len(image_files), text=f"🖼️ OCR {i + 1}/{total_images}")
                progress_ocr.empty()
                status_placeholder.success("✅ OCR complete.")

                # ================== VECTORSTORE =====================
                status_placeholder.info("📦 Vectorstore Agent: Indexing documents...")
                progress_indexing = st.progress(0, text="Indexing...")

                # ✅ Unified ingestion: texts + tables + OCR
                tables = [c for c in all_chunks if c.metadata.get("type") == "table"]
                texts = [c for c in all_chunks if c.metadata.get("type") != "table"]
                documents = texts + tables + ocr_docs

                vectorstore = vector_agent.vectorstore
                for i in range(0, len(documents), 50):
                    vectorstore.add_documents(documents[i:i + 50])
                    progress_indexing.progress(min((i + 50) / len(documents), 1.0),
                                               text=f"📦 Indexed {min(i + 50, len(documents))}/{len(documents)}")
                vectorstore.persist()
                progress_indexing.empty()
                status_placeholder.success("✅ Indexing complete.")

                # Final state update
                st.session_state.vectorstore = vectorstore
                st.session_state.vectorstore_built = True
                st.session_state.building_index = False

            st.balloons()


    # ==================== Chat History Toggle ==================== (within sidebar)
    # dynamically displays the chat history between the user and the assistant and allows the user to control how many previous messages 
    # to view using a slider.

    st.markdown("---")
    st.subheader("🧾 Chat History")
    show_full_chat = st.checkbox("Show chat history", value=False) 

    #Creates a checkbox. If checked, it will show the chat history.
    # Default state is unchecked (value=False).
    #The result (True or False) is stored in the variable show_full_chat.


    if show_full_chat and st.session_state.get("chat_history"): #only proceeds if the checkbox is checked and if session_state has a chat history (i.e, conversation has occured.)
    
    # Add a slider to control how many messages to show (interactions)
        num_msgs_to_show = st.slider(
        "Number of messages to show",
        min_value=1,
        max_value=min(50, len(st.session_state.chat_history)), #Limits the slider to a max of 50 messages or however many exist.
        value=min(10, len(st.session_state.chat_history)), #Sets a default slider value (e.g., 10 if there are at least 10 messages).
        step=1, #Slider increases/decreases one message at a time.
        )

        # Slice last n messages based on slider
        # Extracts only the last num_msgs_to_show messages from the total chat history.
        # [-num_msgs_to_show:] uses Python list slicing to get the most recent entries.
        recent_msgs = st.session_state.chat_history[-num_msgs_to_show:]


        for msg in recent_msgs:
            role = "🧑‍⚕️ You" if msg["role"] == "user" else "🤖 Assistant"
            # Show full message content (or truncate if you want, here showing full)
            st.markdown(f"**{role}:** {msg['content']}")


# ====================== Chat Section ============================
if st.session_state.get("vectorstore_built"):
    st.markdown("---")
    st.subheader("💬 Ask your questions regarding the Research Papers:")

    qa_agent = QAAgent(st.session_state.vectorstore)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("💡 Ask a question...")

    # Render previous messages
    # This block of code is responsible for rendering (displaying) previous chat messages in the main chat area of the Streamlit app — 
    # specifically, in the conversation view below the input box.

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant": #special handling if assistant as the message might include a table to render, an image reference or plain text. 
                
                if try_render_table(msg["content"]):
                    continue

                # Calls a custom function try_render_table() to check if content is a markdown table.
                # If the table is successfully rendered, it skips the rest of the assistant checks using continue.

                else:
                    st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        log_chat("user", user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("🤔 Assistant is thinking..."):
                start_time = time.time()  # Start timer

                chat_history_str = format_chat_history(st.session_state.chat_history, max_turns=10)
                answer = qa_agent.answer(user_input, chat_history=chat_history_str)

                response_time = time.time() - start_time  # End timer

            # Show assistant response
            if try_render_table(answer):
                pass
            else:
                placeholder.markdown(answer)

            # Show response time below assistant message
            st.caption(f"⏱️ Response time: {response_time:.2f} seconds")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        log_chat("assistant", answer)

# Footer
st.markdown("---")
st.markdown("🧠 Powered by OpenAI + LangChain + Streamlit + AgenticAI | Built by Ananya Jain 👩🏻‍💻")


# some questions asked regarding medical book (apart from those mentioned in Project Details png)

#{"timestamp": "2025-07-11T03:57:26.261789", "role": "user", "message": "my name is ananya"}
#{"timestamp": "2025-07-11T03:57:27.838578", "role": "assistant", "message": "Hello Ananya! How can I assist you today?"}
#{"timestamp": "2025-07-11T03:57:53.081885", "role": "user", "message": "\"What role does positional encoding play in the transformer architecture?\""}
#{"timestamp": "2025-07-11T03:57:58.604453", "role": "assistant", "message": "Positional encoding plays a crucial role in the transformer architecture by providing information about the position of each token in the input sequence. Since the transformer model does not have a built-in notion of sequence order (unlike RNNs, which process data sequentially), positional encodings are added to the input embeddings to give the model a sense of the order of the tokens. \n\nHere are the key points regarding positional encoding:\n\n1. **Order Information**: Positional encodings allow the model to understand the relative positions of tokens in the sequence, which is essential for tasks that depend on the order of words, such as translation and text generation.\n\n2. **Sinusoidal Functions**: The original transformer model uses sinusoidal functions to generate positional encodings. Each position is represented by a vector, where each dimension corresponds to a different frequency of sine and cosine functions. This allows the model to learn relationships between positions.\n\n3. **Addition to Input Embeddings**: The positional encodings are added to the input embeddings before they are fed into the encoder and decoder layers. This combination helps the model to incorporate both the content of the tokens and their positions.\n\n4. **Handling Variable Lengths**: Positional encodings can be computed for any length of input sequence, making them suitable for variable-length sequences, which is a common scenario in natural language processing tasks.\n\nOverall, positional encoding is essential for enabling the transformer to process sequences effectively while maintaining the necessary information about the order of tokens."}
#{"timestamp": "2025-07-11T03:58:04.010590", "role": "user", "message": "What's my name?"}
#{"timestamp": "2025-07-11T03:58:05.286150", "role": "assistant", "message": "Your name is Ananya."}
#so remembers previous conversations! 

#create a table with two columns: layer type and complexity per layer. also tell me the page number from the pdf ive uploaded from 
# where you got the table.

#what are the two types of RAG models? Explain them with differences between the two.

# create table for: Open-DomainQATestScores.For TQA, left column uses the standard test set for Open Domain QA, right column uses the TQA-Wiki test set.
# add WQ and CT in previous table and show me updated table
