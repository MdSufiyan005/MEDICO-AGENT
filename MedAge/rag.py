"""
RAG initialization and QA chain (text-only).
"""
import uuid
import logging
from pathlib import Path
from typing import List, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.llms import LLM
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from unstructured.partition.pdf import partition_pdf
from langchain_core.runnables import RunnablePassthrough

from groq import Groq
from MedAge.config import OUTPUT_PATH

logger = logging.getLogger(__name__)

# Basic PDF partitioning (text-only)
def load_pdf_text_elements(pdf_path: str):
    raw_pdf_elements = partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=False,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=None,
    )
    text_elements = []
    for e in raw_pdf_elements:
        if 'CompositeElement' in repr(e):
            text_elements.append(e.text)
    return text_elements

# Wrapper LLM for Groq (keeps same interface)
class GroqLLM(LLM):
    model: str
    client: Any

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self):
        return "groq"

# Build summary chain (Runnable chain)
def build_summary_chain(groq_api_key: str, model_name: str = "llama-3.1-8b-instant"):
    client = Groq(api_key=groq_api_key)
    groq_llm = GroqLLM(model=model_name, client=client)
    summary_prompt_template = """
Summarize the following {element_type}:
{element}
"""
    summary_template = PromptTemplate.from_template(summary_prompt_template)
    summary_chain = (
        RunnablePassthrough.assign(element_type=lambda _: "text")
        | summary_template
        | groq_llm
        | StrOutputParser()
    )
    return summary_chain

# Create vectorstore from text elements
def build_vectorstore_from_texts(text_elements: List[str], embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    documents = []
    for e in text_elements:
        i = str(uuid.uuid4())
        doc = Document(
            page_content=e,  # store original content
            metadata={
                "id": i,
                "type": "text",
                "original_content": e
            }
        )
        documents.append(doc)
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_documents(documents=documents, embedding=embedding_model)
    # save locally
    Path("faiss_index").mkdir(exist_ok=True)
    vectorstore.save_local("faiss_index")
    return vectorstore
