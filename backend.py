from fastapi import FastAPI, HTTPException , UploadFile, File , Query ,Path 
from pydantic import BaseModel ,Field 
from typing import List, Optional, Any , Dict

import uuid 
import os
import json
import uvicorn
import shutil 

# Import config variables and objects from config.py
from config import (
    QDRANT_HOST,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    embedding_model,
    llm,
    sql_generation_llm,
    natural_language_llm,
    qdrant_client_instance,
    vector_store,
    DB_CONNECTION_STRING,
    db,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME2
)

# Import logic functions and constants from backend_logic.py
from backend_logic import (
    DB_SCHEMA_EXAMPLE,
    DB_SCHEMA_EXAMPLE_DESCRIPTION,
    TEXT_TO_SQL_INSTRUCTION,
    validate_rewrite_identify_tables_and_types_logic,
    retrieve_similar_examples_logic,
    assemble_text_to_sql_prompt_logic,
    generate_sql_from_prompt_logic,
    execute_sql_query_logic,
    generate_natural_language_response_logic,
    add_json_examples_to_vector_store_logic,
    add_single_example_to_vector_store_logic,
    get_all_qdrant_points_logic,
    delete_qdrant_point_logic
)

# --- Pydantic Models for API ---
from schema import (
    ProcessQueryRequest,
    QueryAnalysisData,
    SimilarExample,
    ProcessQueryResponse,
    NLSQLInputExample,
    QdrantPoint,
    GetAllPointsResponse,
    DeletePointResponse,
    TEMP_UPLOAD_DIR
)

# --- FastAPI App ---
app = FastAPI()

@app.get("/")
async def root():
    return "Please add /docs to the URL to access the API documentation."

@app.get("/health")
async def health_check():
    # Corrected database status check
    db_status = "connected" if db is not None else "not connected"
    vector_store_status = "available" if vector_store is not None else "not available"
    return {
        "status": "ok",
        "database_status": db_status,
        "vector_store_status": vector_store_status,
        "qdrant_collection": QDRANT_COLLECTION_NAME if vector_store else None
    }

@app.post("/process-query", response_model=ProcessQueryResponse)
async def process_query_endpoint(request: ProcessQueryRequest):
    user_question = request.user_question
    response_data = ProcessQueryResponse(original_question=user_question)

    try:
        llm_output_json_str_1 = validate_rewrite_identify_tables_and_types_logic(
            user_query=user_question,
            db_schema=DB_SCHEMA_EXAMPLE,
            llm_instance=llm
        )
        
        extracted_json_str = llm_output_json_str_1
        first_brace = llm_output_json_str_1.find('{')
        last_brace = llm_output_json_str_1.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            extracted_json_str = llm_output_json_str_1[first_brace : last_brace + 1]
        
        try:
            analysis_dict = json.loads(extracted_json_str)
            response_data.analysis = QueryAnalysisData(**analysis_dict)
        except json.JSONDecodeError as e:
            response_data.error_message = f"Failed to parse query analysis from LLM: {e}. Raw output: {llm_output_json_str_1[:200]}..."
            return response_data
        
        rewritten_query = response_data.analysis.query

        if response_data.analysis.relevant in ['yes', 'maybe']:
            if rewritten_query and rewritten_query.strip() and vector_store:
                similar_examples_raw = retrieve_similar_examples_logic(
                    query_text=rewritten_query,
                    vector_store_instance=vector_store,
                    k=3
                )
                response_data.similar_examples = [SimilarExample(**ex) for ex in similar_examples_raw]

            final_text_to_sql_prompt = assemble_text_to_sql_prompt_logic(
                instruction=TEXT_TO_SQL_INSTRUCTION,
                rewritten_query=rewritten_query,
                few_shot_examples=[ex.dict() for ex in response_data.similar_examples],
                relevant_table_names=response_data.analysis.relevant_tables,
                full_db_schema=DB_SCHEMA_EXAMPLE_DESCRIPTION
            )
            response_data.assembled_prompt_snippet = final_text_to_sql_prompt[:1000] + ("..." if len(final_text_to_sql_prompt) > 1000 else "")

            generated_sql = generate_sql_from_prompt_logic(
                assembled_prompt=final_text_to_sql_prompt,
                sql_llm_instance=sql_generation_llm
            )
            response_data.generated_sql = generated_sql

            if generated_sql and db: 
                query_result = execute_sql_query_logic(
                    sql_query=generated_sql,
                    db_instance=db 
                )
                response_data.query_result = str(query_result)

                if isinstance(query_result, str) and ("Error executing SQL" in query_result or query_result == "No SQL query to execute."):
                    response_data.nl_response = "Could not generate a final answer due to an issue with the SQL query or its execution."
                elif query_result is not None:
                    nl_response = generate_natural_language_response_logic(
                        user_question=user_question,
                        sql_result=str(query_result),
                        nl_llm_instance=natural_language_llm
                    )
                    response_data.nl_response = nl_response
                else:
                     response_data.nl_response = "The query executed but returned no data to form an answer."

            elif not db:
                response_data.error_message = (response_data.error_message or "") + " SQL execution skipped: DB not available."
            elif not generated_sql:
                response_data.nl_response = "No SQL query was generated, so no data could be fetched."
        else:
            response_data.nl_response = "The question was determined to be not relevant to the database schema or could not be processed for SQL generation."

    except HTTPException as e:
        response_data.error_message = e.detail
    except Exception as e:
        response_data.error_message = f"An unexpected error occurred: {str(e)}"
    
    return response_data

@app.post("/add-examples")
async def add_examples_endpoint(file: UploadFile = File(...)):
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store is not available.")
    
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JSON file.")

    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        num_added = add_json_examples_to_vector_store_logic(temp_file_path, vector_store)
        
        return {"message": f"Successfully processed file '{file.filename}'. Added {num_added} documents to the vector store."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e: 
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        await file.close()

@app.post("/add-single-example", response_model=Dict[str, Any]) 
async def add_single_example_endpoint(example: NLSQLInputExample):
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store is not available for adding examples.")
    try:
        added_info = add_single_example_to_vector_store_logic(example.model_dump(), vector_store)
        return {
            "message": "Successfully added single example to the vector store.",
            "added_document_info": added_info
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
@app.get("/get-all-examples", response_model=GetAllPointsResponse)
async def get_all_examples_endpoint(
    limit: int = Query(10, ge=1, le=1000, description="Number of points to retrieve."),
    offset: Optional[str] = Query(None, description="Offset for pagination (UUID string or int from previous response's next_offset)."),
    with_payload: bool = Query(True, description="Whether to include the payload."),
    with_vectors: bool = Query(False, description="Whether to include the vectors (can be large).")
):
    if not qdrant_client_instance:
        raise HTTPException(status_code=503, detail="Qdrant client is not available.")
    if not QDRANT_COLLECTION_NAME or QDRANT_COLLECTION_NAME == "your_default_collection_name":
        raise HTTPException(status_code=400, detail="Qdrant collection name is not configured properly.")
    
    processed_offset = None
    if offset:
        try:
            uuid.UUID(offset)
            processed_offset = offset
        except ValueError:
            try:
                processed_offset = int(offset)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid offset format. Must be a valid UUID or integer string.")
    
    try:
        points_data, next_page_offset = get_all_qdrant_points_logic(
            qdrant_client_instance=qdrant_client_instance,
            collection_name=QDRANT_COLLECTION_NAME,
            limit=limit,
            offset=processed_offset,
            with_payload=with_payload,
            with_vectors=with_vectors
        )
        return GetAllPointsResponse(
            points=[QdrantPoint(**p) for p in points_data],
            next_offset=str(next_page_offset) if next_page_offset is not None else None,
            count=len(points_data)
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching Qdrant points: {str(e)}")

@app.delete("/delete-example/{point_id_str}", response_model=DeletePointResponse)
async def delete_example_endpoint(
    point_id_str: str = Path(..., description="The ID of the Qdrant point to delete (integer or UUID string).")
):
    if not qdrant_client_instance:
        raise HTTPException(status_code=503, detail="Qdrant client is not available.")
    if not QDRANT_COLLECTION_NAME or QDRANT_COLLECTION_NAME == "your_default_collection_name":
        raise HTTPException(status_code=400, detail="Qdrant collection name is not configured properly.")

    try:
        result = delete_qdrant_point_logic(
            qdrant_client_instance=qdrant_client_instance,
            collection_name=QDRANT_COLLECTION_NAME,
            point_id=point_id_str 
        )
        return DeletePointResponse(
            message=f"Attempted to delete point with ID '{point_id_str}'.",
            point_id_deleted=result["point_id_deleted"],
            details=result.get("details")
        )
    except ValueError as e: 
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e: 
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while deleting Qdrant point: {str(e)}")


if __name__ == "__main__":
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME]):
        print("CRITICAL: Azure OpenAI environment variables are not set. The application will not function correctly.")
    uvicorn.run(app, host="127.0.0.1", port=8000)