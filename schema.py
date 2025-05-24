from pydantic import BaseModel ,Field 
from typing import List, Optional, Any , Dict
import os
# --- Pydantic Models for API ---
class ProcessQueryRequest(BaseModel):
    user_question: str

class QueryAnalysisData(BaseModel):
    relevant: str
    query: str
    relevant_tables: List[str]
    query_types: List[str]

class SimilarExample(BaseModel): 
    nl: str
    id: Optional[str] = None 
    sql: Optional[str] = None
    tables: Optional[List[str]] = None
    type: Optional[str] = None
    

class ProcessQueryResponse(BaseModel):
    original_question: str
    analysis: Optional[QueryAnalysisData] = None
    similar_examples: List[SimilarExample] = []
    assembled_prompt_snippet: Optional[str] = None
    generated_sql: Optional[str] = None
    query_result: Optional[Any] = None
    nl_response: Optional[str] = None
    error_message: Optional[str] = None

class NLSQLInputExample(BaseModel): 
    nl: str = Field(..., description="Natural language question.")
    sql: Optional[str] = Field(None, description="The corresponding SQL query.")
    tables: Optional[List[str]] = Field(None, description="List of tables involved.")
    type: Optional[str] = Field(None, description="Type of query (e.g., selection, join).")
    id: Optional[str] = Field(None, description="Optional identifier for the example.")    

class QdrantPoint(BaseModel): 
    id: Any 
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[Any] = None 

class GetAllPointsResponse(BaseModel):
    points: List[QdrantPoint]
    next_offset: Optional[Any] = None 
    count: int

class DeletePointResponse(BaseModel):
    message: str
    point_id_deleted: Any
    details: Optional[str] = None

# Temporary directory for uploaded files
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
