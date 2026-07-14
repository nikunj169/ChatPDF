"""RAGAS evaluation module for measuring retrieval and generation quality."""

from typing import List, Dict, Any

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset


def run_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
) -> Dict[str, float]:
    """Run RAGAS evaluation and return metric scores.

    Args:
        questions: List of user questions.
        answers: List of generated answers.
        contexts: List of lists of retrieved context strings per question.
        ground_truths: List of reference/ground-truth answers.

    Returns:
        Dictionary with metric names and their scores.
    """
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    scores = {
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "context_precision": result["context_precision"],
    }
    return scores


def run_evaluation_from_conversation(
    conversation,
    eval_pairs: List[Dict[str, str]],
) -> Dict[str, float]:
    """Evaluate a conversation chain on a set of question-answer pairs.

    Args:
        conversation: The ConversationalRetrievalChain to evaluate.
        eval_pairs: List of dicts with 'question' and 'ground_truth' keys.

    Returns:
        Dictionary with RAGAS metric scores.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for pair in eval_pairs:
        q = pair["question"]
        gt = pair["ground_truth"]

        response = conversation({"question": q})
        answer = response["answer"]
        source_docs = response.get("source_documents", [])
        context_texts = [doc.page_content for doc in source_docs]

        questions.append(q)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(gt)

    return run_evaluation(questions, answers, contexts, ground_truths)
