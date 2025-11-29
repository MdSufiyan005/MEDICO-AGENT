import logging
import uuid
from typing import List, Optional, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from MedAge.schemas import QueryRequest, AnalysisResponse
from MedAge.utils import save_upload_file
from MedAge.agent import graph
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Medical Image Analysis API",
    description="Chat endpoint for text and/or image-based medical queries",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory store for search results (keeps parity with original)
from MedAge.tools import search_results_store

def _run_graph_and_collect(input_content: str, session_id: str):
    """
    Runs the graph for the given input_content (HumanMessage) and returns
    the final assessment and metadata.
    """
    config = {"configurable": {"thread_id": session_id}}
    input_data = {"messages": [HumanMessage(content=input_content)]}
    session_search_results: List[str] = []
    final_state = None
    for event in graph.stream(input_data, config=config):
        node_name = list(event.keys())[0]
        logger.info(f" → Node '{node_name}' completed")
        final_state = event[node_name]
        # collect tool messages
        if isinstance(final_state, dict) and "messages" in final_state:
            for msg in final_state["messages"]:
                if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
                    session_search_results.extend(msg.content)
    # Normalize messages (same logic)
    if final_state:
        if isinstance(final_state, dict):
            final_messages = final_state.get("messages", [])
        elif isinstance(final_state, list):
            final_messages = final_state
        elif hasattr(final_state, 'content'):
            final_messages = [final_state]
        else:
            final_messages = []
        if not isinstance(final_messages, list):
            final_messages = [final_messages] if final_messages else []
        assistant_responses = [
            msg for msg in final_messages
            if hasattr(msg, 'content') and msg.content and not isinstance(msg, ToolMessage)
        ]
        if assistant_responses:
            final_response = assistant_responses[-1]
            assessment = final_response.content
        else:
            assessment = "No response generated"
        total_messages = len(final_messages)
        tool_calls_made = len([msg for msg in final_messages if isinstance(msg, ToolMessage)])
        search_results_store[session_id] = session_search_results
        return {
            "assessment": assessment,
            "total_messages": total_messages,
            "tool_calls_made": tool_calls_made,
            "search_results": session_search_results
        }
    raise RuntimeError("No result obtained from the processing graph")

@app.post("/chat", response_model=AnalysisResponse, summary="Chat with text/images (JSON)")
async def chat_json(request: QueryRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        image_paths = request.image_paths or []
        input_content = f"Question: {request.question}\nImages: {image_paths if image_paths else 'None'}"
        logger.info(f"📝 Chat JSON request (session={session_id}) question_len={len(request.question)} images={len(image_paths)}")
        result = _run_graph_and_collect(input_content, session_id)
        return AnalysisResponse(
            success=True,
            session_id=session_id,
            question=request.question,
            assessment=result["assessment"],
            total_messages=result["total_messages"],
            tool_calls_made=result["tool_calls_made"],
            search_results=result["search_results"],
            error=None
        )
    except Exception as e:
        logger.exception("Chat JSON endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/upload", response_model=AnalysisResponse, summary="Chat with file uploads (Swagger-friendly)")
async def chat_upload(
    question: str = Form(..., description="Your medical question"),
    session_id: Optional[str] = Form(None, description="Session ID for conversation history"),
    files: Optional[List[Any]] = File(None, description="Upload medical images (optional)")
):
    try:
        session_id = session_id or str(uuid.uuid4())
        image_paths = []
        valid_files: List[UploadFile] = []
        if files:
            for idx, f in enumerate(files):
                if isinstance(f, UploadFile):
                    if not getattr(f, "filename", None):
                        continue
                    valid_files.append(f)
                else:
                    logger.info(f"Ignoring non-file placeholder in files[{idx}]: {type(f)}")
        if valid_files:
            for file in valid_files:
                try:
                    saved_path = await save_upload_file(file)
                    image_paths.append(saved_path)
                except HTTPException:
                    raise
                except Exception as e:
                    logger.exception(f"Failed to save upload {getattr(file,'filename',None)}: {e}")
        input_content = f"Question: {question}\nImages: {image_paths if image_paths else 'None'}"
        logger.info(f"📝 Chat Upload request (session={session_id}) question_len={len(question)} images={len(image_paths)}")
        result = _run_graph_and_collect(input_content, session_id)
        return AnalysisResponse(
            success=True,
            session_id=session_id,
            question=question,
            assessment=result["assessment"],
            total_messages=result["total_messages"],
            tool_calls_made=result["tool_calls_made"],
            search_results=result["search_results"],
            error=None
        )
    except Exception as e:
        logger.exception("Chat Upload endpoint failed")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health")
async def health_check():
    from MedAge.config import DEVICE, _model_cache
    return {
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": _model_cache["model"] is not None
    }

@app.post("/delete_uploads")
async def delete_all_uploads():
    from MedAge.config import UPLOAD_DIR
    try:
        files = list(UPLOAD_DIR.glob("*"))
        count = 0
        for f in files:
            try:
                f.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")
        return {"deleted_files": count}
    except Exception as e:
        logger.exception("Failed to delete uploads")
        raise HTTPException(status_code=500, detail=str(e))
