import streamlit as st
import sys
from pathlib import Path
import subprocess


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# BACKEND
# ============================================================

from Backend.question_pipeline import ask


# ============================================================
# PATHS
# ============================================================

DATA_DIR = PROJECT_ROOT / "Backend" / "data"

# Your existing ingest_pipeline.py expects this filename.
# The user can upload ANY PDF filename; Streamlit will
# save it using this internal filename.
INGEST_PDF_PATH = DATA_DIR / "spotify_web_app_architecture.pdf"


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def display_value(value):
    """Safely display dictionaries, lists and normal values."""

    if value is None:
        st.info("No data available.")
        return

    if isinstance(value, (dict, list)):
        st.json(value)
    else:
        st.write(value)


def display_source(source):
    """Safely display source information."""

    if source is None:
        st.info("No source available.")
        return

    if isinstance(source, (dict, list)):
        st.json(source)
    else:
        st.write(str(source))


# ============================================================
# PDF PROCESSING
# ============================================================

def process_uploaded_pdf(uploaded_file):
    """
    Accept any uploaded PDF.

    Example:
        architecture.pdf
        company_report.pdf
        my_document.pdf
        research.pdf

    The uploaded file is internally saved as:
        Backend/data/spotify_web_app_architecture.pdf

    This allows the existing ingestion pipeline to remain
    unchanged.
    """

    try:

        # ----------------------------------------------------
        # Create data directory
        # ----------------------------------------------------

        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Validate extension
        # ----------------------------------------------------

        original_name = uploaded_file.name

        if not original_name.lower().endswith(".pdf"):

            st.error(
                "Only PDF files are supported."
            )

            return False

        # ----------------------------------------------------
        # Save uploaded PDF
        # ----------------------------------------------------

        with open(
            INGEST_PDF_PATH,
            "wb",
        ) as output_file:

            output_file.write(
                uploaded_file.getbuffer()
            )

        # ----------------------------------------------------
        # Run existing ingestion pipeline
        #
        # IMPORTANT:
        # Use -m because Backend.ingest_pipeline uses
        # relative imports such as:
        #
        # from .llm import embeddings, llm
        # ----------------------------------------------------

        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "Backend.ingest_pipeline",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )

        # ----------------------------------------------------
        # Ingestion failed
        # ----------------------------------------------------

        if process.returncode != 0:

            st.error(
                "PDF ingestion failed."
            )

            if process.stdout:

                st.markdown(
                    "**Ingestion output:**"
                )

                st.code(
                    process.stdout
                )

            if process.stderr:

                st.markdown(
                    "**Error:**"
                )

                st.code(
                    process.stderr
                )

            return False

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        st.session_state[
            "pdf_processed"
        ] = True

        st.session_state[
            "uploaded_pdf_name"
        ] = original_name

        return True

    except Exception as exc:

        st.error(
            f"PDF processing error: {exc}"
        )

        return False


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧠 Hybrid RAG — FAISS + Neo4j"
)

st.caption(
    "Vector RAG + Knowledge Graph + Guardrails + Evaluation"
)


# ============================================================
# PDF UPLOAD
# ============================================================

st.header(
    "📄 Upload Knowledge Base"
)

st.write(
    "Upload any PDF document to create the Hybrid RAG knowledge base."
)

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    accept_multiple_files=False,
    help="You can upload any PDF document.",
)


# ============================================================
# SHOW SELECTED FILE
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Selected PDF: {uploaded_file.name}"
    )

    file_size_mb = (
        uploaded_file.size / (1024 * 1024)
    )

    st.write(
        f"File size: {file_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # Process button
    # --------------------------------------------------------

    if st.button(
        "🚀 Process PDF",
        type="primary",
    ):

        with st.spinner(
            "Processing PDF → FAISS → Neo4j..."
        ):

            success = process_uploaded_pdf(
                uploaded_file
            )

        if success:

            st.success(
                f"'{uploaded_file.name}' processed successfully!"
            )

            st.info(
                "Your Hybrid RAG knowledge base has been updated."
            )

            # ------------------------------------------------
            # Show ingestion details
            # ------------------------------------------------

            # Run output isn't returned from helper currently,
            # so simply show success here.
            st.rerun()


# ============================================================
# CURRENT PDF STATUS
# ============================================================

if st.session_state.get(
    "pdf_processed",
    False,
):

    st.success(
        "✅ Knowledge base ready"
    )

    st.caption(
        "Current PDF: "
        + st.session_state.get(
            "uploaded_pdf_name",
            "Uploaded PDF",
        )
    )


st.divider()


# ============================================================
# QUESTION / ANSWER
# ============================================================

st.header(
    "💬 Ask Your Knowledge Base"
)

question = st.text_input(
    "Ask a question about the uploaded PDF:"
)


# ============================================================
# ASK
# ============================================================

if st.button(
    "🔎 Ask",
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching FAISS + Neo4j..."
        ):

            try:

                result = ask(
                    question.strip()
                )

            except Exception as exc:

                st.error(
                    "Hybrid RAG failed."
                )

                st.exception(exc)

                st.stop()

        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader(
            "📝 Answer"
        )

        answer = result.get(
            "answer",
            "",
        )

        if answer:

            st.write(
                answer
            )

        else:

            st.warning(
                "No answer was generated."
            )


        st.divider()


        # ====================================================
        # VECTOR RETRIEVAL
        # ====================================================

        st.subheader(
            "🔎 Vector Retrieval"
        )

        vector_results = result.get(
            "vector_results",
            [],
        )

        if not vector_results:

            st.info(
                "No vector results found."
            )

        else:

            for index, item in enumerate(
                vector_results,
                start=1,
            ):

                with st.expander(
                    f"Vector Result {index}"
                ):

                    if isinstance(
                        item,
                        dict,
                    ):

                        # Content
                        content = item.get(
                            "content",
                            "",
                        )

                        if content:

                            st.markdown(
                                "**Content**"
                            )

                            st.write(
                                content
                            )

                        # Source
                        if "source" in item:

                            st.markdown(
                                "**Source**"
                            )

                            display_source(
                                item.get(
                                    "source"
                                )
                            )

                        # Rank
                        if "rank" in item:

                            st.markdown(
                                "**Rank**"
                            )

                            st.write(
                                item.get(
                                    "rank"
                                )
                            )

                        # Metadata
                        if "metadata" in item:

                            metadata = item.get(
                                "metadata"
                            )

                            if metadata:

                                st.markdown(
                                    "**Metadata**"
                                )

                                display_value(
                                    metadata
                                )

                    else:

                        display_value(
                            item
                        )


        st.divider()


        # ====================================================
        # KNOWLEDGE GRAPH
        # ====================================================

        st.subheader(
            "🕸️ Knowledge Graph Retrieval"
        )

        graph_results = result.get(
            "graph_results",
            [],
        )

        if not graph_results:

            st.info(
                "No Knowledge Graph results found."
            )

        else:

            for index, item in enumerate(
                graph_results,
                start=1,
            ):

                if isinstance(
                    item,
                    dict,
                ):

                    title = (
                        item.get("entity")
                        or item.get("name")
                        or item.get("id")
                        or f"Graph Entity {index}"
                    )

                else:

                    title = (
                        f"Graph Entity {index}"
                    )

                with st.expander(
                    str(title)
                ):

                    display_value(
                        item
                    )


        st.divider()


        # ====================================================
        # VALIDATED SOURCES
        # ====================================================

        st.subheader(
            "✅ Validated Sources"
        )

        sources = result.get(
            "sources",
            [],
        )

        if not sources:

            st.info(
                "No validated sources available."
            )

        else:

            for index, source in enumerate(
                sources,
                start=1,
            ):

                st.markdown(
                    f"**Source {index}**"
                )

                display_source(
                    source
                )