import numpy as np
from pypdf import PdfReader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

import os
import easyocr
from pdf2image import convert_from_path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

load_dotenv()

reader_ocr = easyocr.Reader(['en'], gpu=False)


# POPPLER_PATH = r"C:\Users\chinm\Downloads\Poppler\poppler-26.02.0\Library\bin"



print("Using updated rag_core.py")

SUBJECT_PROMPTS = {
    "General": """
You are a helpful placement prep assistant.
Use ONLY the context below to answer.
If not found, say: "I don't know."

Context: {context}
Question: {question}
Answer:""",

    "DSA": """
You are a DSA interview coach.
Explain concepts clearly with time/space complexity where relevant.
Use ONLY the context below.
If not found, say: "I don't know."

Context: {context}
Question: {question}
Answer:""",

    "OOPS": """
You are an Object Oriented Programming expert.
Give clear definitions with real-world examples.
Use ONLY the context below.
If not found, say: "I don't know."

Context: {context}
Question: {question}
Answer:""",

    "DBMS": """
You are a Database Management expert.
Explain with examples, mention ACID properties or normalization where relevant.
Use ONLY the context below.
If not found, say: "I don't know."

Context: {context}
Question: {question}
Answer:""",

    "OS": """
You are an Operating Systems interview expert.
Be precise, use examples like process scheduling or memory management.
Use ONLY the context below.
If not found, say: "I don't know."

Context: {context}
Question: {question}
Answer:""",

    "CN": """
You are a Computer Networks expert.
Explain protocols, layers, and concepts clearly.
Use ONLY the context below.
If not found, say: "I don't know."

Context: {context}
Question: {question}
Answer:"""
}

def extract_text_with_ocr(filepath):

    os.makedirs("ocr_cache", exist_ok=True)

    filename = os.path.basename(filepath)

    cache_path = os.path.join(
        "ocr_cache",
        filename.replace(".pdf", ".txt")
    )

    # --------------------
    # Load cached OCR
    # --------------------

    if os.path.exists(cache_path):

        print(f"⚡ Loading OCR cache: {filename}")

        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    # --------------------
    # OCR starts
    # --------------------

    print(f"🔍 Running OCR on {filename}")

    pages = convert_from_path(
        filepath,
        dpi=150,
        # poppler_path=POPPLER_PATH
    )

    print(f"📄 Total pages: {len(pages)}")

    text = ""

    for i, page in enumerate(pages):

        if i % 5 == 0:
            print(f"Processing page {i+1}/{len(pages)}")

        result = reader_ocr.readtext(
            np.array(page),
            detail=0
        )

        text += " ".join(result)
        text += "\n"

    # Save OCR cache
    

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("✅ OCR Finished")

    return text


def load_and_chunk(file_paths,chunk_size=500,chunk_overlap=50):

    all_docs = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    for file in file_paths:
        filepath = file["path"] 
        filename = file["name"]

        reader = PdfReader(filepath)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()

            if extracted:
                 text += extracted

        # if text.strip() == "":
        #     print(f"Using OCR for {filename}...")
        #     text = extract_text_with_ocr(filepath)
        # if len(text.strip()) < 100:

        #     print("No readable text found.")

        #     text = extract_text_with_ocr(filepath)
        if len(text.strip()) < 100:
            print(f"Skipping OCR for {filename}")
            continue
        chunks = splitter.split_text(text)
        

        docs = [
            Document(
                page_content=chunk,
                metadata={"source": filename}
            )
            for chunk in chunks
        ]

        all_docs.extend(docs)
            

    return all_docs
   

#SBuild FAISS vector store
def build_vectorstore(docs, force_rebuild=False):

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("Building FAISS index...")

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore
# def build_vectorstore(docs, index_path="faiss_index", force_rebuild=False):
#     embeddings = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2"
#     )
#     if os.path.exists(index_path) and not force_rebuild:
#         print("Loading existing FAISS index...")
#         return FAISS.load_local(
#             index_path,
#             embeddings,
#             allow_dangerous_deserialization=True
#         )

#     print("Building new FAISS index...")

#     vectorstore = FAISS.from_documents(
#         docs,
#         embeddings
#     )

#     vectorstore.save_local(index_path)

#     return vectorstore
  
    



def get_chain(vectorstore, subject="General"):

    

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    prompt = PromptTemplate(
        template=SUBJECT_PROMPTS[subject],
        input_variables=["context", "question"]
    )



    chain = RetrievalQA.from_chain_type(
        llm=llm,
       retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10
            }
        ),
        
        chain_type_kwargs={
            "prompt": prompt
        },
        return_source_documents=True

    )

    return chain


def get_chain_with_memory(vectorstore, subject="General", memory=None):

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
         retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10
            }
        ),
        memory=memory,
        return_source_documents=True
    )

    return chain, memory


def retrieve_with_scores(vectorstore, query, k=3):
    docs_with_scores = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    results = []

    for doc, score in docs_with_scores:
        results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "score": round(float(score), 4)
        })

    return results



   
