import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from ragas import EvaluationDataset
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
)

from Backend.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    EMBEDDING_MODEL,
    EVAL_DIR,
)

from Backend.question_pipeline import ask


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_QUESTIONS_PATH = EVAL_DIR / "test_questions.json"
RESULTS_PATH = EVAL_DIR / "results.json"


# ============================================================
# CONFIGURATION
# ============================================================

# Number of contexts sent to Ragas for each question.
MAX_CONTEXTS_FOR_EVAL = 10

# Maximum answer/context characters used during evaluation.
#
# This prevents excessively large prompts from consuming
# unnecessary tokens while preserving enough information
# for the evaluation metrics.
MAX_CONTEXT_CHARS = 12000
MAX_RESPONSE_CHARS = 16000

# Ragas evaluator output token limit.
#
# Faithfulness can require a longer structured evaluation
# response, so we explicitly provide a larger token budget.
RAGAS_MAX_TOKENS = 4096


# ============================================================
# VALIDATE ENVIRONMENT
# ============================================================

def validate_configuration() -> None:
    """
    Validate required environment configuration.
    """

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing.\n"
            "Please check your .env file."
        )

    if not OPENAI_MODEL:
        raise ValueError(
            "OPENAI_MODEL is missing."
        )

    if not EMBEDDING_MODEL:
        raise ValueError(
            "EMBEDDING_MODEL is missing."
        )


# ============================================================
# LOAD TEST QUESTIONS
# ============================================================

def load_questions() -> List[Dict[str, Any]]:
    """
    Load evaluation questions from:

        Backend/evals/test_questions.json
    """

    if not TEST_QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation questions file not found:\n"
            f"{TEST_QUESTIONS_PATH}"
        )

    with open(
        TEST_QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    # --------------------------------------------------------
    # Support either:
    #
    # [
    #   {"question": "..."},
    #   {"question": "..."}
    # ]
    #
    # or:
    #
    # {
    #   "questions": [
    #       {"question": "..."}
    #   ]
    # }
    # --------------------------------------------------------

    if isinstance(data, dict):
        if "questions" in data:
            data = data["questions"]
        else:
            raise ValueError(
                "test_questions.json contains a JSON object "
                "but no 'questions' field was found."
            )

    if not isinstance(data, list):
        raise ValueError(
            "test_questions.json must contain a JSON list."
        )

    normalized_questions = []

    for item in data:

        # -----------------------------------------------
        # String question
        # -----------------------------------------------
        if isinstance(item, str):
            normalized_questions.append(
                {
                    "question": item
                }
            )
            continue

        # -----------------------------------------------
        # Dictionary question
        # -----------------------------------------------
        if isinstance(item, dict):

            question = (
                item.get("question")
                or item.get("user_input")
                or item.get("query")
            )

            if not question:
                raise ValueError(
                    "An evaluation item does not contain "
                    "'question', 'user_input', or 'query'."
                )

            normalized_item = {
                "question": str(question)
            }

            # Optional reference answer
            if item.get("reference") is not None:
                normalized_item["reference"] = str(
                    item["reference"]
                )

            normalized_questions.append(
                normalized_item
            )

            continue

        raise ValueError(
            f"Unsupported evaluation item type: "
            f"{type(item).__name__}"
        )

    return normalized_questions


# ============================================================
# BUILD CONTEXTS
# ============================================================

def clean_context_text(text: Any) -> str:
    """
    Convert context content into clean text.
    """

    if text is None:
        return ""

    if isinstance(text, str):
        return text.strip()

    return str(text).strip()


def build_contexts(result: Dict[str, Any]) -> List[str]:
    """
    Convert Hybrid RAG retrieval results into the format
    expected by modern Ragas Collections metrics.

    Ragas expects:

        retrieved_contexts=[
            "context 1",
            "context 2",
            ...
        ]
    """

    contexts: List[str] = []

    # ========================================================
    # VECTOR RESULTS
    # ========================================================

    vector_results = result.get(
        "vector_results",
        []
    )

    if isinstance(vector_results, list):

        for item in vector_results:

            if isinstance(item, dict):

                content = (
                    item.get("content")
                    or item.get("text")
                    or item.get("page_content")
                )

                content = clean_context_text(content)

                if content:
                    contexts.append(content)

            elif isinstance(item, str):

                content = clean_context_text(item)

                if content:
                    contexts.append(content)

    # ========================================================
    # GRAPH RESULTS
    # ========================================================

    graph_results = result.get(
        "graph_results",
        []
    )

    if isinstance(graph_results, list):

        for item in graph_results:

            if isinstance(item, dict):

                possible_fields = [
                    "content",
                    "text",
                    "description",
                    "value",
                    "name",
                ]

                graph_text_parts = []

                for field in possible_fields:

                    value = item.get(field)

                    if value is not None:

                        value = clean_context_text(value)

                        if value:
                            graph_text_parts.append(
                                f"{field}: {value}"
                            )

                if graph_text_parts:

                    contexts.append(
                        "Neo4j Graph Context:\n"
                        + "\n".join(graph_text_parts)
                    )

            elif isinstance(item, str):

                content = clean_context_text(item)

                if content:
                    contexts.append(
                        "Neo4j Graph Context:\n"
                        + content
                    )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_contexts = []

    seen = set()

    for context in contexts:

        normalized = context.strip()

        if not normalized:
            continue

        key = normalized.lower()

        if key not in seen:

            seen.add(key)

            unique_contexts.append(
                normalized
            )

    # ========================================================
    # LIMIT CONTEXTS
    # ========================================================

    unique_contexts = unique_contexts[
        :MAX_CONTEXTS_FOR_EVAL
    ]

    return unique_contexts


# ============================================================
# BUILD RAGAS DATASET
# ============================================================

def build_dataset(
    questions: List[Dict[str, Any]]
) -> EvaluationDataset:
    """
    Run Hybrid RAG for every evaluation question and create
    a modern Ragas EvaluationDataset.
    """

    samples = []

    print()
    print("=" * 70)
    print("RUNNING HYBRID RAG EVALUATION")
    print("=" * 70)

    total = len(questions)

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question = item["question"]

        print()
        print(
            f"[{index}/{total}] {question}"
        )

        try:

            result = ask(question)

        except Exception as exc:

            print(
                f"  Hybrid RAG ERROR: {exc}"
            )

            # Keep evaluation running for other questions.
            continue

        answer = result.get(
            "answer",
            ""
        )

        if answer is None:
            answer = ""

        answer = str(answer).strip()

        contexts = build_contexts(
            result
        )

        print(
            f"  Answer length: {len(answer)}"
        )

        print(
            f"  Retrieved contexts: "
            f"{len(contexts)}"
        )

        # ----------------------------------------------------
        # Limit extremely large evaluation input.
        # ----------------------------------------------------

        evaluation_answer = answer[
            :MAX_RESPONSE_CHARS
        ]

        evaluation_contexts = []

        total_chars = 0

        for context in contexts:

            remaining = (
                MAX_CONTEXT_CHARS
                - total_chars
            )

            if remaining <= 0:
                break

            trimmed_context = context[
                :remaining
            ]

            evaluation_contexts.append(
                trimmed_context
            )

            total_chars += len(
                trimmed_context
            )

        sample = {
            "user_input": question,
            "response": evaluation_answer,
            "retrieved_contexts": evaluation_contexts,
        }

        # ----------------------------------------------------
        # Optional reference answer.
        # ----------------------------------------------------

        reference = item.get(
            "reference"
        )

        if reference:
            sample["reference"] = reference

        samples.append(sample)

    if not samples:
        raise RuntimeError(
            "No successful Hybrid RAG evaluation samples "
            "were created."
        )

    dataset = EvaluationDataset.from_list(
        samples
    )

    print()
    print(
        f"Created dataset with "
        f"{len(samples)} samples."
    )

    return dataset


# ============================================================
# CREATE RAGAS EVALUATORS
# ============================================================

def create_evaluator():
    """
    Create modern Ragas evaluator objects.

    Ragas 0.4.x Collections metrics require modern
    LLM/embedding interfaces.
    """

    # ========================================================
    # OPENAI ASYNC CLIENT FOR RAGAS
    # ========================================================

    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY
    )

    # ========================================================
    # MODERN RAGAS LLM
    # ========================================================

    evaluator = llm_factory(
        OPENAI_MODEL,
        client=client,
        temperature=0,
        max_tokens=RAGAS_MAX_TOKENS,
    )

    # ========================================================
    # MODERN RAGAS EMBEDDINGS
    # ========================================================

    embedding_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY
    )

    ragas_embeddings = embedding_factory(
        "openai",
        model=EMBEDDING_MODEL,
        client=embedding_client,
        interface="modern",
    )

    print()
    print(
        "Ragas evaluator initialized."
    )

    return (
        evaluator,
        ragas_embeddings,
    )


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

async def evaluate_question(
    question: str,
    response: str,
    contexts: List[str],
    evaluator,
    ragas_embeddings,
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate:

    1. Faithfulness
    2. Answer Relevancy
    3. Context Precision

    using modern Ragas Collections metrics.
    """

    result = {
        "question": question,
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
    }

    # ========================================================
    # FAITHFULNESS
    # ========================================================

    try:

        faithfulness_metric = Faithfulness(
            llm=evaluator
        )

        faithfulness_result = (
            await faithfulness_metric.ascore(
                user_input=question,
                response=response,
                retrieved_contexts=contexts,
            )
        )

        result[
            "faithfulness"
        ] = float(
            faithfulness_result.value
        )

    except Exception as exc:

        print(
            f"  Faithfulness ERROR: {exc}"
        )

    # ========================================================
    # ANSWER RELEVANCY
    # ========================================================

    try:

        answer_relevancy_metric = (
            AnswerRelevancy(
                llm=evaluator,
                embeddings=ragas_embeddings,
            )
        )

        # IMPORTANT:
        #
        # AnswerRelevancy.ascore() does NOT accept
        # retrieved_contexts.
        #
        # Only user_input and response are supplied.

        answer_relevancy_result = (
            await answer_relevancy_metric.ascore(
                user_input=question,
                response=response,
            )
        )

        result[
            "answer_relevancy"
        ] = float(
            answer_relevancy_result.value
        )

    except Exception as exc:

        print(
            f"  Answer Relevancy ERROR: {exc}"
        )

    # ========================================================
    # CONTEXT PRECISION
    # ========================================================

    try:

        context_precision_metric = (
            ContextPrecisionWithoutReference(
                llm=evaluator
            )
        )

        context_precision_result = (
            await context_precision_metric.ascore(
                user_input=question,
                response=response,
                retrieved_contexts=contexts,
            )
        )

        result[
            "context_precision"
        ] = float(
            context_precision_result.value
        )

    except Exception as exc:

        print(
            f"  Context Precision ERROR: {exc}"
        )

    return result


# ============================================================
# RUN RAGAS EVALUATION
# ============================================================

async def run_ragas_evaluation(
    dataset: EvaluationDataset,
    evaluator,
    ragas_embeddings,
) -> List[Dict[str, Any]]:
    """
    Run Ragas metrics for every SingleTurnSample.

    IMPORTANT:
    EvaluationDataset yields SingleTurnSample objects.

    Therefore use:

        sample.user_input
        sample.response
        sample.retrieved_contexts

    NOT:

        sample["user_input"]
        sample["response"]
        sample["retrieved_contexts"]
    """

    print()
    print("=" * 70)
    print("RUNNING RAGAS METRICS")
    print("=" * 70)

    results = []

    for index, sample in enumerate(
        dataset,
        start=1,
    ):

        # ====================================================
        # SINGLE TURN SAMPLE
        # ====================================================

        question = sample.user_input

        response = sample.response

        contexts = (
            sample.retrieved_contexts
            or []
        )

        reference = getattr(
            sample,
            "reference",
            None,
        )

        print()
        print(
            f"[{index}] {question}"
        )

        print(
            f"  Retrieved contexts: "
            f"{len(contexts)}"
        )

        if not contexts:

            print(
                "  WARNING: No retrieved "
                "contexts available."
            )

            results.append(
                {
                    "question": question,
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                }
            )

            continue

        # ====================================================
        # RUN METRICS
        # ====================================================

        result = await evaluate_question(
            question=question,
            response=response,
            contexts=contexts,
            evaluator=evaluator,
            ragas_embeddings=ragas_embeddings,
            reference=reference,
        )

        results.append(result)

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        faithfulness = result[
            "faithfulness"
        ]

        answer_relevancy = result[
            "answer_relevancy"
        ]

        context_precision = result[
            "context_precision"
        ]

        print(
            "  Faithfulness: "
            + (
                f"{faithfulness:.4f}"
                if faithfulness is not None
                else "None"
            )
        )

        print(
            "  Answer Relevancy: "
            + (
                f"{answer_relevancy:.4f}"
                if answer_relevancy is not None
                else "None"
            )
        )

        print(
            "  Context Precision: "
            + (
                f"{context_precision:.4f}"
                if context_precision is not None
                else "None"
            )
        )

    return results


# ============================================================
# CALCULATE AVERAGES
# ============================================================

def calculate_averages(
    results: List[Dict[str, Any]]
) -> Dict[str, Optional[float]]:
    """
    Calculate averages only from successful metric values.

    Failed metrics represented by None are excluded.
    """

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
    ]

    averages = {}

    for metric_name in metric_names:

        values = []

        for result in results:

            value = result.get(
                metric_name
            )

            if value is None:
                continue

            try:

                values.append(
                    float(value)
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        if values:

            averages[
                metric_name
            ] = sum(values) / len(values)

        else:

            averages[
                metric_name
            ] = None

    return averages


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    questions: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    averages: Dict[str, Optional[float]],
) -> None:
    """
    Save complete evaluation results to:

        Backend/evals/results.json
    """

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "project": "Hybrid RAG",
        "evaluation_framework": "Ragas",
        "metrics": [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
        ],
        "total_questions": len(
            questions
        ),
        "successful_questions": len(
            results
        ),
        "results": results,
        "averages": averages,
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

def print_final_results(
    results: List[Dict[str, Any]],
    averages: Dict[str, Optional[float]],
) -> None:
    """
    Print evaluation results in a readable format.
    """

    print()
    print("=" * 70)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 70)

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()

        print(
            f"Question {index}: "
            f"{result['question']}"
        )

        faithfulness = result[
            "faithfulness"
        ]

        answer_relevancy = result[
            "answer_relevancy"
        ]

        context_precision = result[
            "context_precision"
        ]

        print(
            "Faithfulness: "
            + (
                f"{faithfulness:.4f}"
                if faithfulness is not None
                else "None"
            )
        )

        print(
            "Answer Relevancy: "
            + (
                f"{answer_relevancy:.4f}"
                if answer_relevancy is not None
                else "None"
            )
        )

        print(
            "Context Precision: "
            + (
                f"{context_precision:.4f}"
                if context_precision is not None
                else "None"
            )
        )

    # ========================================================
    # AVERAGES
    # ========================================================

    print()
    print("=" * 70)
    print("AVERAGE METRICS")
    print("=" * 70)

    for metric_name in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
    ]:

        value = averages.get(
            metric_name
        )

        valid_count = sum(
            1
            for result in results
            if result.get(metric_name)
            is not None
        )

        total_count = len(results)

        if value is None:

            print(
                f"{metric_name}: "
                f"No values"
            )

        else:

            print(
                f"{metric_name}: "
                f"{value:.4f} "
                f"({valid_count}/{total_count} valid)"
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Main evaluation pipeline.
    """

    print("=" * 70)
    print("HYBRID RAG + RAGAS EVALUATION")
    print("=" * 70)

    # ========================================================
    # CONFIGURATION
    # ========================================================

    validate_configuration()

    # ========================================================
    # LOAD QUESTIONS
    # ========================================================

    questions = load_questions()

    print()
    print(
        f"Loaded {len(questions)} "
        f"evaluation questions."
    )

    # ========================================================
    # BUILD HYBRID RAG DATASET
    # ========================================================

    dataset = build_dataset(
        questions
    )

    # ========================================================
    # CREATE RAGAS EVALUATORS
    # ========================================================

    (
        evaluator,
        ragas_embeddings,
    ) = create_evaluator()

    # ========================================================
    # RUN RAGAS
    # ========================================================

    results = asyncio.run(
        run_ragas_evaluation(
            dataset=dataset,
            evaluator=evaluator,
            ragas_embeddings=ragas_embeddings,
        )
    )

    # ========================================================
    # CALCULATE AVERAGES
    # ========================================================

    averages = calculate_averages(
        results
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print_final_results(
        results,
        averages,
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_results(
        questions=questions,
        results=results,
        averages=averages,
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULTS_PATH
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()