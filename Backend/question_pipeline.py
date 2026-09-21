import json
import re
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS

from .config import VECTOR_PATH, TOP_K_VECTOR, TOP_K_GRAPH
from .llm import embeddings, llm
from .models import RAGAnswer
from .guardrails_config import validate_answer
from .neo4j_kg import Neo4jKG


# ============================================================
# LOAD FAISS
# ============================================================

def load_vector_store():
    if not VECTOR_PATH.exists():
        raise FileNotFoundError(
            f"FAISS vector store not found at: {VECTOR_PATH}\n"
            "Run ingest_pipeline.py first."
        )

    return FAISS.load_local(
        str(VECTOR_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ============================================================
# FAISS RETRIEVAL
# ============================================================

def retrieve_vector_context(
    question: str,
    top_k: int = TOP_K_VECTOR,
) -> List[Dict[str, Any]]:

    vector_store = load_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=top_k,
    )

    results = []

    for index, document in enumerate(documents, start=1):
        results.append(
            {
                "source": "FAISS",
                "rank": index,
                "content": document.page_content,
                "metadata": document.metadata,
            }
        )

    return results


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(question: str) -> List[str]:
    cleaned = re.sub(
        r"[^\w\s.-]",
        " ",
        question,
    )

    words = cleaned.split()

    stop_words = {
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "does",
        "do",
        "used",
        "use",
        "application",
        "applications",
        "architecture",
        "components",
        "communicate",
        "communication",
    }

    entities = []

    for word in words:

        normalized = word.strip().lower()

        if not normalized:
            continue

        if normalized in stop_words:
            continue

        if len(normalized) < 2:
            continue

        entities.append(word.strip())

    unique_entities = []
    seen = set()

    for entity in entities:

        key = entity.lower()

        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)

    return unique_entities


# ============================================================
# NEO4J RETRIEVAL
# ============================================================

def retrieve_graph_context(
    question: str,
    top_k: int = TOP_K_GRAPH,
) -> List[Dict[str, Any]]:

    results = []

    try:

        kg = Neo4jKG()

        entities = extract_entities(question)

        # ----------------------------------------------------
        # IMPORTANT:
        # Neo4j query uses:
        #
        #     name IN $names
        #
        # Therefore $names MUST be a list.
        # ----------------------------------------------------

        for entity in entities[:top_k]:

            try:

                graph_result = kg.search_entities(
                    [entity]
                )

            except Exception as exc:

                print(
                    f"Neo4j entity retrieval warning "
                    f"for '{entity}': {exc}"
                )

                continue

            if not graph_result:
                continue

            if isinstance(graph_result, list):

                for item in graph_result:

                    if isinstance(item, dict):

                        results.append(
                            {
                                "source": "Neo4j",
                                "entity": entity,
                                **item,
                            }
                        )

                    else:

                        results.append(
                            {
                                "source": "Neo4j",
                                "entity": entity,
                                "content": str(item),
                            }
                        )

            elif isinstance(graph_result, dict):

                results.append(
                    {
                        "source": "Neo4j",
                        "entity": entity,
                        **graph_result,
                    }
                )

            else:

                results.append(
                    {
                        "source": "Neo4j",
                        "entity": entity,
                        "content": str(graph_result),
                    }
                )

            if len(results) >= top_k:
                break

        return results[:top_k]

    except Exception as exc:

        print(
            f"Neo4j retrieval warning: {exc}"
        )

        return []


# ============================================================
# FORMAT FAISS CONTEXT
# ============================================================

def format_vector_context(
    vector_results: List[Dict[str, Any]],
) -> str:

    if not vector_results:
        return (
            "No relevant FAISS documents "
            "were retrieved."
        )

    sections = []

    for item in vector_results:

        rank = item.get(
            "rank",
            "",
        )

        content = item.get(
            "content",
            "",
        )

        metadata = item.get(
            "metadata",
            {},
        )

        page = metadata.get(
            "page"
        )

        source = metadata.get(
            "source"
        )

        metadata_text = []

        if source:
            metadata_text.append(
                f"source={source}"
            )

        if page is not None:
            metadata_text.append(
                f"page={page}"
            )

        suffix = ""

        if metadata_text:
            suffix = (
                f" ({', '.join(metadata_text)})"
            )

        sections.append(
            f"[FAISS Document {rank}{suffix}]\n"
            f"{content}"
        )

    return "\n\n".join(sections)


# ============================================================
# FORMAT NEO4J CONTEXT
# ============================================================

def format_graph_context(
    graph_results: List[Dict[str, Any]],
) -> str:

    if not graph_results:
        return (
            "No relevant Neo4j graph information "
            "was retrieved."
        )

    sections = []

    for index, item in enumerate(
        graph_results,
        start=1,
    ):

        content = (
            item.get("content")
            or item.get("text")
            or item.get("description")
            or item.get("value")
        )

        if content is None:

            serializable = {}

            for key, value in item.items():

                if key not in {
                    "source",
                    "rank",
                }:
                    serializable[key] = value

            try:

                content = json.dumps(
                    serializable,
                    ensure_ascii=False,
                    default=str,
                )

            except Exception:

                content = str(
                    serializable
                )

        entity = item.get(
            "entity"
        )

        prefix = ""

        if entity:
            prefix = (
                f"Entity: {entity}\n"
            )

        sections.append(
            f"[Neo4j Graph {index}]\n"
            f"{prefix}"
            f"{content}"
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# HYBRID CONTEXT
# ============================================================

def build_hybrid_context(
    vector_results: List[Dict[str, Any]],
    graph_results: List[Dict[str, Any]],
) -> str:

    vector_context = format_vector_context(
        vector_results
    )

    graph_context = format_graph_context(
        graph_results
    )

    return f"""
==============================
FAISS VECTOR CONTEXT
==============================

{vector_context}


==============================
NEO4J GRAPH CONTEXT
==============================

{graph_context}
""".strip()


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question: str,
    context: str,
) -> RAGAnswer:

    prompt = f"""
You are a grounded Hybrid RAG assistant.

Answer the user's question using ONLY the
information contained in the supplied FAISS
and Neo4j context.

Do not invent facts.

If the retrieved context does not contain
enough information, say that the information
is not available in the retrieved context.

USER QUESTION:
{question}

RETRIEVED HYBRID CONTEXT:
{context}
"""

    structured_llm = llm.with_structured_output(
        RAGAnswer
    )

    result = structured_llm.invoke(
        prompt
    )

    if isinstance(
        result,
        RAGAnswer,
    ):
        return result

    if isinstance(
        result,
        dict,
    ):
        return RAGAnswer(
            **result
        )

    raise TypeError(
        "Unexpected LLM output type: "
        f"{type(result)}"
    )


# ============================================================
# GUARDRAILS
# ============================================================

def validate_generated_answer(
    rag_answer: RAGAnswer,
) -> RAGAnswer:

    answer_text = rag_answer.answer

    if answer_text is None:
        raise ValueError(
            "Generated answer is None."
        )

    if not isinstance(
        answer_text,
        str,
    ):
        answer_text = str(
            answer_text
        )

    answer_text = answer_text.strip()

    if not answer_text:
        raise ValueError(
            "Generated answer is empty."
        )

    rag_answer.answer = answer_text

    try:

        # IMPORTANT:
        # Pass only the string.
        validate_answer(
            answer_text
        )

    except Exception as exc:

        print(
            f"Guardrails validation warning: {exc}"
        )

    return rag_answer


# ============================================================
# MAIN HYBRID RAG
# ============================================================

def ask(
    question: str,
) -> Dict[str, Any]:

    if not isinstance(
        question,
        str,
    ):
        raise ValueError(
            "Question must be a string."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    # --------------------------------------------------------
    # 1. FAISS
    # --------------------------------------------------------

    try:

        vector_results = (
            retrieve_vector_context(
                question,
                TOP_K_VECTOR,
            )
        )

    except Exception as exc:

        print(
            f"FAISS retrieval warning: {exc}"
        )

        vector_results = []

    # --------------------------------------------------------
    # 2. NEO4J
    # --------------------------------------------------------

    graph_results = (
        retrieve_graph_context(
            question,
            TOP_K_GRAPH,
        )
    )

    # --------------------------------------------------------
    # 3. HYBRID CONTEXT
    # --------------------------------------------------------

    hybrid_context = (
        build_hybrid_context(
            vector_results,
            graph_results,
        )
    )

    # --------------------------------------------------------
    # 4. LLM
    # --------------------------------------------------------

    rag_answer = generate_answer(
        question,
        hybrid_context,
    )

    # --------------------------------------------------------
    # 5. GUARDRAILS
    # --------------------------------------------------------

    rag_answer = (
        validate_generated_answer(
            rag_answer
        )
    )

    # --------------------------------------------------------
    # 6. SOURCES
    # --------------------------------------------------------

    sources = []

    for item in vector_results:

        sources.append(
            {
                "type": "FAISS",
                "rank": item.get(
                    "rank"
                ),
                "content": item.get(
                    "content"
                ),
                "metadata": item.get(
                    "metadata",
                    {},
                ),
            }
        )

    for item in graph_results:

        content = (
            item.get("content")
            or item.get("text")
            or item.get("description")
            or item.get("value")
            or str(item)
        )

        sources.append(
            {
                "type": "Neo4j",
                "rank": item.get(
                    "rank"
                ),
                "entity": item.get(
                    "entity"
                ),
                "content": content,
            }
        )

    # --------------------------------------------------------
    # IMPORTANT FOR RAGAS
    # --------------------------------------------------------
    #
    # Return the raw vector_results and graph_results.
    #
    # evals.py uses these to build:
    #
    #     retrieved_contexts
    #
    # --------------------------------------------------------

    return {
        "question": question,

        # Plain string for frontend/evals.
        "answer": rag_answer.answer,

        # Structured answer if needed.
        "rag_answer": rag_answer.model_dump(),

        # Raw retrieval results.
        "vector_results": vector_results,
        "graph_results": graph_results,

        # Combined source list.
        "sources": sources,

        # Optional debugging.
        "hybrid_context": hybrid_context,
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "HYBRID RAG QUESTION PIPELINE"
    )

    print(
        "=" * 70
    )

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:
        raise SystemExit(
            "Question cannot be empty."
        )

    try:

        result = ask(
            question
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "FINAL ANSWER"
        )

        print(
            "=" * 70
        )

        print(
            result["answer"]
        )

        print(
            "\nVector contexts:",
            len(
                result[
                    "vector_results"
                ]
            ),
        )

        print(
            "Graph contexts:",
            len(
                result[
                    "graph_results"
                ]
            ),
        )

    except Exception as exc:

        print(
            "\n" + "=" * 70
        )

        print(
            "HYBRID RAG ERROR"
        )

        print(
            "=" * 70
        )

        print(
            str(exc)
        )