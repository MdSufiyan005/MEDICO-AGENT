import logging
from typing import List, Any
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from MedAge.vision import (
    _load_vision_model,
    _load_multiple_images,
    _prepare_model_inputs_multiple,
    _generate_answer,
    _decode_answer
)
from MedAge.config import GROQ_API_KEY, DEVICE

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field
# Keep track of search results globally (same behaviour as original)
search_results_store: dict = {}

class SearchInput(BaseModel):
    query: str = Field(description="The medical query to search for. Do NOT include image references.")

@tool(args_schema=SearchInput)
def search_medical_info(query: str) -> List[str]:
    """
    Search for grounded medical information using TavilySearch.
    Use this for ALL queries regarding recent news, latest research, specific articles, or external information.
    """
    logger.info("🔍 TOOL CALLED: search_medical_info")
    logger.info(f"📝 Query: {query}")
    search = TavilySearch(max_results=3, search_depth="advanced")
    try:
        search_results = search.invoke({"query": query})
    except Exception as e:
        logger.exception("TavilySearch invocation failed")
        return [f"Search failed: {e}"]
    results: List[str] = []
    if isinstance(search_results, dict) and 'result' in search_results:
        for i in search_results['result']:
            if isinstance(i, dict) and 'content' in i:
                results.append(i['content'])
    elif isinstance(search_results, list):
        for item in search_results:
            if isinstance(item, dict) and 'content' in item:
                results.append(item['content'])
            elif isinstance(item, str):
                results.append(item)
    else:
        results.append(str(search_results))
    logger.info(f"✅ Tool executed successfully. Found {len(results)} results.")
    return results

@tool
def great_brain_specialist(query: str, images: List[str] = []) -> str:
    """
    Fine-tuned vision model tool for medical image analysis.
    Accepts a query and optional list of image file paths.
    DO NOT use this tool for recent news, latest research articles, or external web searches. It has no internet access.
    """
    logger.info("🧠 TOOL CALLED: great_brain_specialist")
    logger.info(f"📝 Query: {query}")
    logger.info(f"🖼️ Images: {len(images) if images else 0}")
    try:
        model, tokenizer = _load_vision_model()
    except Exception as e:
        logger.exception("Failed to load vision model")
        return f"Error loading vision model: {e}"

    # Filter out invalid image paths (e.g. "None", "null", empty strings)
    if images:
        images = [
            img for img in images 
            if img and str(img).lower() not in ['none', 'null', '']
        ]

    if not images:
        # text-only flow
        try:
            messages = [{"role": "user", "content": query}]
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = tokenizer(text=[input_text], return_tensors="pt", padding=True)
            for k, v in inputs.items():
                import torch
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(DEVICE)
            logger.info("🔄 Generating text-only analysis...")
            output = _generate_answer(model, inputs)
            answer = _decode_answer(output, tokenizer, input_text)
            logger.info(f"✅ Analysis generated successfully ({len(answer)} chars)")
            return answer
        except Exception as e:
            logger.exception("Error in text-only processing")
            return f"Error in text-only processing: {str(e)}"

    # image + text flow
    try:
        pil_images = _load_multiple_images(images)
        logger.info("🔄 Preparing model inputs...")
        inputs, input_text = _prepare_model_inputs_multiple(query, pil_images, tokenizer)
        logger.info("🔄 Generating analysis...")
        output = _generate_answer(model, inputs)
        answer = _decode_answer(output, tokenizer, input_text)
        logger.info(f"✅ Analysis generated successfully ({len(answer)} chars)")
        return answer
    except FileNotFoundError as e:
        logger.error(str(e))
        return f"Error: {str(e)}"
    except ValueError as e:
        logger.error(str(e))
        return f"Error: {str(e)}"
    except Exception as e:
        logger.exception("Error in great_brain_specialist")
        return f"Error in great_brain_specialist: {str(e)}"

# lung_disease_rag requires a vectorstore and qa_chain object.
# The app will set these attributes at runtime (see app.agent where rag is initialized)
@tool
def lung_disease_rag(query: str) -> str:
    """
    Retrieve and answer questions based on the local RAG vectorstore (text-only docs).
    The vectorstore and qa_chain are expected to be attached to this module at runtime.
    """
    logger.info("📚 TOOL CALLED: lung_disease_rag")
    logger.info(f"📝 Query: {query}")
    try:
        from app import rag as rag_module  # local import to avoid circular at module import time
        vectorstore = getattr(rag_module, "vectorstore", None)
        qa_chain = getattr(rag_module, "qa_chain", None)
        if vectorstore is None or qa_chain is None:
            return "Error: RAG is not initialized."
        relevant_docs = vectorstore.similarity_search(query)
        context = ""
        for d in relevant_docs:
            if d.metadata.get('type') == 'text':
                context += "[text]" + d.metadata.get('original_content', "")
        result = qa_chain.invoke({"context": context, "question": query})
        logger.info(f"✅ RAG query executed successfully ({len(result)} chars)")
        return result
    except Exception as e:
        logger.exception("Error in lung_disease_rag")
        return f"Error in RAG processing: {str(e)}"
