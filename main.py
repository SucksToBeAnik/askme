import io
from fastapi import FastAPI, HTTPException, UploadFile, Form
from typing import Annotated

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from ai import answer_random_query
import fitz
import faiss

from fastapi import FastAPI, UploadFile, HTTPException, Form
from typing import Annotated
import io
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq.chat_models import ChatGroq

from settings import env_store


app = FastAPI(title="Q/A on PDF")

# Initialize embeddings
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Get the dimension of the embeddings
sample_text = "This is a sample text to get the embedding dimension."
sample_embedding = embedding_function.embed_query(sample_text)
embedding_dim = len(sample_embedding)

# Initialize FAISS index with the correct dimension
index = faiss.IndexFlatL2(embedding_dim)
vector_store = FAISS(
    embedding_function=embedding_function,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Initialize the language model for question answering
llm = ChatGroq(
    api_key=env_store.GROQ_API_KEY, model="gemma2-9b-it", stop_sequences=None
)
qa_chain = load_qa_chain(llm, chain_type="stuff")


@app.post("/file")
async def upload_and_process_pdf(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    contents = await file.read()
    pdf_stream = io.BytesIO(contents)

    doc = fitz.open(stream=pdf_stream, filetype="pdf")

    # Extract text from each page
    text = ""

    for page in doc:
        text += page.get_textbox("text")

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    # Embed the text chunks using the same embedding function
    vector_store.add_texts(texts=chunks)

    return {
        "message": f"Processed {len(chunks)} text chunks from {file.filename}",
    }


@app.post("/file/query")
async def query_pdf(query: Annotated[str, Form()]):
    query_embedding = embedding_function.embed_query(query)
    relevant_docs = vector_store.similarity_search_by_vector(
        query_embedding, k=3 
    )

    if not relevant_docs:
        return {
            "query": query,
            "answer": "No relevant answers were found for the query.",
        }

    prompt = PromptTemplate.from_template(
        """
You are an expert researcher who only provides factual and accurate answers based strictly on the provided context. Do not include information or assumptions not found in the context. If the context does not contain sufficient information to answer the question accurately, simply respond with "The provided context does not contain enough information to answer the question."

Context: {context}

Now, based on the context, answer the following question:
Question: {question}
"""
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"context": relevant_docs, "question": query})

    return {
        "query": query,
        "answer": answer,
        "relevent": relevant_docs
    }


@app.post("/query")
async def rout(query: Annotated[str, Form()]):

    answer = answer_random_query(query)
    return answer
