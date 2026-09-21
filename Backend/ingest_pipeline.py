# ============================================================
# HYBRID RAG - PDF INGESTION PIPELINE
# ============================================================
#
# PDF
#   |
#   v
# PyPDF
#   |
#   v
# Text Documents
#   |
#   v
# Chunking
#   |
#   +--------------------+
#   |                    |
#   v                    v
# FAISS                Neo4j Aura
# Vector Store          Knowledge Graph
#
# Supports:
#
# 1. Default PDF
#    python -m Backend.ingest_pipeline
#
# 2. Any uploaded PDF
#    python -m Backend.ingest_pipeline "path\file.pdf"
#
# ============================================================

import re
import sys
from pathlib import Path

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .config import PDF_PATH, VECTOR_PATH
from .llm import embeddings
from .neo4j_kg import Neo4jKG


# ============================================================
# DIRECTORIES
# ============================================================

DATA_DIR = PDF_PATH.parent

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VECTOR_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD PDF
# ============================================================

def load_pdf(pdf_path=None):
    """
    Load a PDF and convert each page into a LangChain Document.
    """

    if pdf_path is None:
        pdf_path = PDF_PATH

    pdf_path = Path(pdf_path)

    print()
    print("=" * 70)
    print("PDF INGESTION")
    print("=" * 70)

    print(f"PDF path: {pdf_path}")

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF file not found:\n{pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":

        raise ValueError(
            f"Invalid file type: {pdf_path.suffix}\n"
            "Only PDF files are supported."
        )

    # --------------------------------------------------------
    # Read PDF
    # --------------------------------------------------------

    reader = PdfReader(
        str(pdf_path)
    )

    print(
        f"Total pages: {len(reader.pages)}"
    )

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            text = page.extract_text()

        except Exception as exc:

            print(
                f"Warning: Could not extract "
                f"page {page_number}: {exc}"
            )

            continue

        if not text:
            continue

        text = text.strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_number
                }
            )
        )

    if not documents:

        raise ValueError(
            "No readable text was found in the PDF."
        )

    print(
        f"Loaded {len(documents)} pages."
    )

    return documents


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(documents):
    """
    Split PDF documents into smaller chunks.
    """

    print()
    print("Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    if not chunks:

        raise ValueError(
            "No chunks were created."
        )

    print(
        f"Created {len(chunks)} chunks."
    )

    return chunks


# ============================================================
# BUILD FAISS
# ============================================================

def build_faiss(chunks):
    """
    Create and save FAISS vector store.
    """

    print()
    print("Building FAISS vector store...")

    if not chunks:

        raise ValueError(
            "Cannot create FAISS from empty chunks."
        )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    VECTOR_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(
        str(VECTOR_PATH)
    )

    print(
        "FAISS vector store saved to:"
    )

    print(
        VECTOR_PATH
    )

    return vector_store


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(text):
    """
    Extract simple technical/domain entities from text.

    This is intentionally lightweight and does not require
    another NLP package.
    """

    entities = set()

    # --------------------------------------------------------
    # Common technology/domain terms
    # --------------------------------------------------------

    known_entities = [
        "Spotify",
        "React",
        "React.js",
        "Node.js",
        "Node",
        "JavaScript",
        "TypeScript",
        "Python",
        "FastAPI",
        "Flask",
        "Django",
        "HTML",
        "CSS",
        "REST API",
        "API",
        "Web API",
        "Frontend",
        "Backend",
        "Database",
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "Redis",
        "Docker",
        "AWS",
        "Azure",
        "Google Cloud",
        "Cloud",
        "Server",
        "Client",
        "Browser",
        "Web Application",
        "Web Application Architecture",
        "Authentication",
        "Authorization",
        "Load Balancer",
        "Microservices",
    ]

    lower_text = text.lower()

    for entity in known_entities:

        if entity.lower() in lower_text:

            entities.add(
                entity
            )

    # --------------------------------------------------------
    # Acronyms
    # --------------------------------------------------------

    acronyms = re.findall(
        r"\b[A-Z]{2,6}\b",
        text
    )

    for acronym in acronyms:

        if acronym not in {
            "THE",
            "AND",
            "FOR",
            "WITH",
            "FROM",
            "THIS",
            "THAT",
            "HTTP",
            "HTTPS"
        }:

            entities.add(
                acronym
            )

    # --------------------------------------------------------
    # Technology names containing .js
    # --------------------------------------------------------

    js_terms = re.findall(
        r"\b[A-Za-z]+\.js\b",
        text
    )

    for term in js_terms:

        entities.add(
            term
        )

    return sorted(
        entities
    )


# ============================================================
# BUILD NEO4J
# ============================================================

def build_neo4j(chunks):
    """
    Populate Neo4j Aura.

    IMPORTANT:
    Your Neo4jKG implementation defines:

        add_document(
            document_id,
            text
        )

    Therefore we use:

        kg.add_document(
            document_id=document_id,
            text=text
        )

    NOT:

        content=
        metadata=
    """

    print()
    print("Populating Neo4j Aura Graph...")

    if not chunks:

        raise ValueError(
            "Cannot populate Neo4j from empty chunks."
        )

    kg = Neo4jKG()

    try:

        # ----------------------------------------------------
        # Clear previous graph
        # ----------------------------------------------------
        #
        # This makes every new PDF upload create a fresh
        # knowledge graph instead of mixing old PDF data
        # with the new PDF.
        #
        # ----------------------------------------------------

        print(
            "Clearing previous Neo4j graph..."
        )

        kg.clear_database()

        # ----------------------------------------------------
        # Create indexes
        # ----------------------------------------------------

        print(
            "Creating Neo4j indexes..."
        )

        kg.create_indexes()

        # ----------------------------------------------------
        # Add chunks
        # ----------------------------------------------------

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            text = chunk.page_content.strip()

            if not text:
                continue

            document_id = (
                f"chunk_{index}"
            )

            # ------------------------------------------------
            # CORRECT Neo4j METHOD
            # ------------------------------------------------

            kg.add_document(
                document_id=document_id,
                text=text
            )

            print(
                f"  Document {index}/{len(chunks)} added."
            )

            # ------------------------------------------------
            # Extract entities
            # ------------------------------------------------

            entities = extract_entities(
                text
            )

            for entity in entities:

                kg.add_entity(
                    entity_name=entity,
                    entity_type="Concept",
                    document_id=document_id
                )

        print()
        print(
            "Neo4j Aura Graph populated successfully."
        )

    finally:

        kg.close()


# ============================================================
# COMPLETE PDF INGESTION
# ============================================================

def ingest_pdf(pdf_path=None):
    """
    Complete Hybrid RAG ingestion pipeline.
    """

    print()
    print("=" * 70)
    print("HYBRID RAG PDF INGESTION PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1 - Load PDF
    # --------------------------------------------------------

    documents = load_pdf(
        pdf_path
    )

    # --------------------------------------------------------
    # STEP 2 - Split PDF
    # --------------------------------------------------------

    chunks = split_documents(
        documents
    )

    # --------------------------------------------------------
    # STEP 3 - FAISS
    # --------------------------------------------------------

    build_faiss(
        chunks
    )

    # --------------------------------------------------------
    # STEP 4 - Neo4j
    # --------------------------------------------------------

    build_neo4j(
        chunks
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)

    print(
        f"PDF pages     : {len(documents)}"
    )

    print(
        f"Total chunks  : {len(chunks)}"
    )

    print(
        f"FAISS path    : {VECTOR_PATH}"
    )

    print(
        "Neo4j status  : Successfully populated"
    )

    print()

    return {
        "pdf": str(
            pdf_path if pdf_path else PDF_PATH
        ),
        "pages": len(documents),
        "chunks": len(chunks),
        "vector_store": str(
            VECTOR_PATH
        ),
        "neo4j": True
    }


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Command-line entry point.

    Without argument:
        python -m Backend.ingest_pipeline

    With PDF:
        python -m Backend.ingest_pipeline "Backend\data\file.pdf"
    """

    # --------------------------------------------------------
    # Streamlit / command-line uploaded PDF
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        pdf_path = Path(
            sys.argv[1]
        )

    else:

        pdf_path = PDF_PATH

    ingest_pdf(
        pdf_path
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()