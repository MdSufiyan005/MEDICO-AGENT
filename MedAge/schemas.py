# from typing import Optional, List
# from pydantic import BaseModel

# class QueryRequest(BaseModel):
#     question: str
#     image_paths: Optional[List[str]] = None
#     session_id: Optional[str] = None

# class AnalysisResponse(BaseModel):
#     success: bool
#     session_id: str
#     question: str
#     assessment: str
#     total_messages: int
#     tool_calls_made: int
#     search_results: List[str]
#     error: Optional[str] = None
