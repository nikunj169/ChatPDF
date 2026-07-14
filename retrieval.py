"""Retrieval and conversation chain with citation support."""

from typing import List, Tuple, Dict, Any

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

from config import EMBEDDING_MODEL, EMBEDDING_DEVICE, LLM_TEMPERATURE, RETRIEVER_K


CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
)


def get_vectorstore(documents: List[Document]):
    """Create a FAISS vector store from documents with metadata."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )
    vectorstore = faiss.FAISS.from_documents(documents=documents, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, k: int = RETRIEVER_K):
    """Create a conversational retrieval chain that returns source documents for citations."""
    llm = ChatOpenAI(temperature=LLM_TEMPERATURE)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory,
        return_source_documents=True,
    )
    return conversation_chain


def ask_question(conversation, question: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Ask a question and return the answer with citation metadata."""
    response = conversation({"question": question})
    answer = response["answer"]
    sources = []
    seen = set()
    for doc in response.get("source_documents", []):
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "snippet": doc.page_content[:200],
            })
    return answer, sources
