from dotenv import load_dotenv
import streamlit as st

from htmlTemplates import css, bot_template, user_template, citation_template
from ingestion import get_pdf_pages, get_chunks
from retrieval import get_vectorstore, get_conversation_chain, ask_question
from evaluation import run_evaluation
from config import CHUNKING_STRATEGIES, RETRIEVER_K


def handle_question(question):
    answer, sources = ask_question(st.session_state.conversation, question)
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "bot", "content": answer, "sources": sources})

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.write(user_template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)
            if msg.get("sources"):
                items = ""
                for s in msg["sources"]:
                    items += f'<div class="citation-item">📄 {s["source"]} (p.{s["page"]}): <em>{s["snippet"]}...</em></div>'
                st.write(citation_template.replace("{{CITATIONS}}", items), unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)

        st.subheader("Chunking Strategy")
        strategy_key = st.selectbox(
            "Select chunking strategy:",
            options=list(CHUNKING_STRATEGIES.keys()),
            format_func=lambda k: CHUNKING_STRATEGIES[k]["name"],
        )
        strategy = CHUNKING_STRATEGIES[strategy_key]
        st.caption(strategy["description"])

        chunk_params = {}
        if strategy_key == "fixed_size":
            chunk_params["chunk_size"] = st.slider("Chunk size", 200, 2000, strategy["default_chunk_size"])
            chunk_params["chunk_overlap"] = st.slider("Chunk overlap", 0, 500, strategy["default_chunk_overlap"])
            chunk_params["separator"] = st.text_input("Separator", strategy["default_separator"])
        elif strategy_key == "semantic":
            chunk_params["buffer_size"] = st.slider("Buffer size", 1, 10, strategy["default_buffer_size"])
            chunk_params["breakpoint_percentile"] = st.slider("Breakpoint percentile", 50, 99, strategy["default_breakpoint_percentile"])

        st.subheader("Retrieval")
        retriever_k = st.slider("Number of retrieved chunks (k)", 1, 10, RETRIEVER_K)

        if st.button("Process"):
            if not docs:
                st.error("Please upload at least one PDF.")
                return
            with st.spinner("Processing documents..."):
                pages = get_pdf_pages(docs)
                if not pages:
                    st.error("No text could be extracted from the uploaded PDFs.")
                    return
                documents = get_chunks(pages, strategy=strategy_key, **chunk_params)
                st.session_state.documents = documents
                vectorstore = get_vectorstore(documents)
                st.session_state.conversation = get_conversation_chain(vectorstore, k=retriever_k)
                st.session_state.chat_history = []
                st.session_state.eval_ready = True
            st.success(f"Processed {len(pages)} pages into {len(documents)} chunks using {strategy['name']}.")

        if st.session_state.get("eval_ready"):
            st.subheader("RAGAS Evaluation")
            st.caption("Add question-answer pairs to evaluate retrieval and generation quality.")
            eval_q = st.text_area("Question", key="eval_q")
            eval_gt = st.text_area("Ground truth answer", key="eval_gt")
            if "eval_pairs" not in st.session_state:
                st.session_state.eval_pairs = []

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add pair"):
                    if eval_q.strip() and eval_gt.strip():
                        st.session_state.eval_pairs.append({"question": eval_q.strip(), "ground_truth": eval_gt.strip()})
                        st.success(f"Added. Total pairs: {len(st.session_state.eval_pairs)}")
                    else:
                        st.warning("Both fields required.")
            with col2:
                if st.button("Clear pairs"):
                    st.session_state.eval_pairs = []
                    st.info("Cleared.")

            if st.session_state.get("eval_pairs"):
                st.write(f"**{len(st.session_state.eval_pairs)} evaluation pair(s)**")
                if st.button("Run RAGAS Evaluation"):
                    with st.spinner("Running evaluation..."):
                        scores = run_evaluation_from_pairs(
                            st.session_state.conversation,
                            st.session_state.eval_pairs,
                        )
                    st.session_state.eval_scores = scores

            if st.session_state.get("eval_scores"):
                scores = st.session_state.eval_scores
                st.subheader("Results")
                for metric, score in scores.items():
                    st.metric(label=metric.replace("_", " ").title(), value=f"{score:.3f}")


def run_evaluation_from_pairs(conversation, eval_pairs):
    questions, answers, contexts, ground_truths = [], [], [], []
    for pair in eval_pairs:
        answer, sources = ask_question(conversation, pair["question"])
        questions.append(pair["question"])
        answers.append(answer)
        contexts.append([s["snippet"] for s in sources])
        ground_truths.append(pair["ground_truth"])
    return run_evaluation(questions, answers, contexts, ground_truths)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "eval_ready" not in st.session_state:
        st.session_state.eval_ready = False
    if "eval_pairs" not in st.session_state:
        st.session_state.eval_pairs = []
    if "eval_scores" not in st.session_state:
        st.session_state.eval_scores = None

    st.header("Chat with multiple PDFs :books:")
    question = st.text_input("Ask a question from your documents:")
    if question and st.session_state.conversation:
        handle_question(question)
    elif question and not st.session_state.conversation:
        st.warning("Please upload and process documents first.")

    render_sidebar()


if __name__ == "__main__":
    main()
