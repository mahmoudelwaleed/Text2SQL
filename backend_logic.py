from fastapi import HTTPException 
from typing import List, Any, Optional , Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import LLMChain
import json
from uuid import uuid4, UUID
from qdrant_client import models


DB_SCHEMA_EXAMPLE = """
Album(AlbumId, Title, ArtistId)
Artist(ArtistId, Name)
Customer(CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
Employee(EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
Genre(GenreId, Name)
Invoice(InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
InvoiceLine(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
MediaType(MediaTypeId, Name)
Playlist(PlaylistId, Name)
PlaylistTrack(PlaylistId, TrackId)
Track(TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)
"""

DB_SCHEMA_EXAMPLE_DESCRIPTION = """
Album(AlbumId, Title, ArtistId)
# Stores information about music albums.
Artist(ArtistId, Name)
# Contains data about music artists.
Customer(CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
# Holds customer information.
Employee(EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
# Stores employee details.
Genre(GenreId, Name)
# Represents music genres.
Invoice(InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
# Contains invoice information.
InvoiceLine(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
# Represents individual items within an invoice.
MediaType(MediaTypeId, Name)
# Stores media types for tracks.
Playlist(PlaylistId, Name)
# Contains user-created playlists.
PlaylistTrack(PlaylistId, TrackId)
# Associates tracks with playlists.
Track(TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)
# Stores detailed information about each music track.
"""

RELEVANCE_REWRITE_TABLES_TYPES_PROMPT_TEMPLATE  = """You are an AI assistant. Your task is to analyze a user question based on a database schema, determine if it's answerable, rewrite it for clarity if applicable, identify the relevant tables, and classify query types.

### Database Schema:
{schema}

### User Question:
{query}

### Instructions:

1. **Relevance** – Can the question be answered using the schema?
   - "yes": Clearly answerable using the schema.
   - "no": Not answerable using the schema.
   - "maybe": Possibly answerable or needs more information.


2. **Rewrite the Question**  
If relevant ("yes" or "maybe"), rewrite the question to:
- Be clearer.
- Use appropriate table and column names from the schema.
If not relevant ("no"), return the original question.

3. **Identify Relevant Tables**:
   - If relevance is "yes" or "maybe", list only the table names needed to answer the question.
   - If relevance is "no", return an empty list.

4.  **Identify Query Types**:
    - If relevance is "yes" or "maybe", analyze the rewritten question and list the types of operations likely needed to answer it. Multiple types can apply.
    - Allowed types: `selection`, `filter`, `aggregation`, `order`, `subquery`, `limit`, `join`, `other`.
    - If relevance is "no", provide an empty list for query_types.

### Output Format:
Respond with a single, valid JSON object. Do not include any other text before or after the JSON object.
Example of the JSON object structure:
{{
  "relevant": "yes",
  "query": "Rewritten user question using schema terms.",
  "relevant_tables": ["Table1", "Table2"],
  "query_types": ["filter", "join"]
}}

### Now Process:
Database Schema:
{schema}

User Question:
{query}

Output JSON:
"""

TEXT_TO_SQL_INSTRUCTION = """You are an expert SQL generator for MYSQL.
Given a database schema and a user question, generate a syntactically correct SQL query that answers the question.
Pay close attention to the exact table and column names provided in the schema.
Use the provided examples, if any, as a guide for query structure and style.
Output ONLY the SQL query. Do not add any explanation or preamble do not add ```sql , just the sql syntax.
"""

SQL_RESULT_TO_NL_PROMPT_TEMPLATE = """You are an AI assistant.
Given an original user question and the result of a SQL query executed to answer that question,
provide a concise, natural language response to the user.
Do not mention the SQL query or the database. Just answer the question based on the provided data.

Original User Question:
{user_question}

SQL Query Result:
{sql_result}

Natural Language Answer:
"""

def validate_rewrite_identify_tables_and_types_logic(user_query: str, db_schema: str, llm_instance) -> str:
    prompt = PromptTemplate(template=RELEVANCE_REWRITE_TABLES_TYPES_PROMPT_TEMPLATE, input_variables=["query", "schema"])
    chain = LLMChain(llm=llm_instance, prompt=prompt)
    response = chain.invoke({"query": user_query, "schema": db_schema})
    return response['text']

def retrieve_similar_examples_logic(query_text: str, vector_store_instance, k: int = 3) -> list:
    if not query_text or not query_text.strip() or vector_store_instance is None:
        return []
    try:
        similar_docs = vector_store_instance.similarity_search(query_text, k=k)
        return [{"nl": doc.page_content, **doc.metadata} for doc in similar_docs]
    except Exception:
        return []

def format_dynamic_schema_logic(relevant_table_names: list, full_db_schema: str) -> str:
    if not relevant_table_names:
        return "No specific table schema provided. Please infer from the question."
    schema_lines = full_db_schema.strip().split('\n')
    dynamic_schema_parts = []
    for table_name in relevant_table_names:
        for line_idx, line in enumerate(schema_lines):
            if line.strip().startswith(table_name + "("):
                dynamic_schema_parts.append(line.strip())
                for desc_line_idx in range(line_idx + 1, len(schema_lines)):
                    desc_line = schema_lines[desc_line_idx].strip()
                    if desc_line.startswith("#"):
                        dynamic_schema_parts.append(desc_line)
                    elif "(" in desc_line and ")" in desc_line and not desc_line.startswith("#"):
                        break
                    elif not desc_line.startswith("#") and desc_line:
                        break
                break
    return "\n".join(dynamic_schema_parts) if dynamic_schema_parts else "Selected table schemas not found or empty."

def format_few_shot_examples_logic(few_shot_examples: list) -> str:
    if not few_shot_examples:
        return ""
    formatted_examples_str = "### Examples (NL to SQL):\n"
    for ex in few_shot_examples:
        nl = ex.get('nl', 'No NL provided')
        sql = ex.get('sql', 'No SQL provided')
        formatted_examples_str += f"-- User Question: {nl}\nSQL: {sql}\n\n"
    return formatted_examples_str.strip()

def assemble_text_to_sql_prompt_logic(instruction: str, rewritten_query: str, few_shot_examples: list, relevant_table_names: list, full_db_schema: str) -> str:
    dynamic_schema_str = format_dynamic_schema_logic(relevant_table_names, full_db_schema)
    few_shots_str = format_few_shot_examples_logic(few_shot_examples)
    prompt_parts = [instruction, "\n### Database Schema:", "Only use the following tables and their columns.", dynamic_schema_str]
    if few_shots_str:
        prompt_parts.append("\n" + few_shots_str)
    prompt_parts.extend(["\n### Task:", "Convert the following user question to a SQL query.", f"User Question: {rewritten_query}", "SQL Query:"])
    return "\n".join(prompt_parts)

def generate_sql_from_prompt_logic(assembled_prompt: str, sql_llm_instance) -> str:
    prompt_template = PromptTemplate.from_template("{final_prompt}")
    sql_generation_chain = LLMChain(llm=sql_llm_instance, prompt=prompt_template)
    response = sql_generation_chain.invoke({"final_prompt": assembled_prompt})
    return response.get('text', '').strip()

def execute_sql_query_logic(sql_query: str, db_instance):
    if not db_instance:
        raise HTTPException(status_code=500, detail="Database connection not available in logic.")
    if not sql_query or not sql_query.strip():
        return "No SQL query to execute."
    try:
        return db_instance.run(sql_query)
    except Exception as e:
        return f"Error executing SQL: {str(e)}"

def generate_natural_language_response_logic(user_question: str, sql_result: str, nl_llm_instance) -> str:
    prompt = PromptTemplate(template=SQL_RESULT_TO_NL_PROMPT_TEMPLATE, input_variables=["user_question", "sql_result"])
    chain = LLMChain(llm=nl_llm_instance, prompt=prompt)
    response = chain.invoke({"user_question": user_question, "sql_result": str(sql_result)})
    return response.get('text', "Could not generate a natural language response.").strip()

def add_json_examples_to_vector_store_logic(json_examples_filepath: str, vector_store_instance):
    if not vector_store_instance:
        raise ValueError("Vector store instance is not available.")
    try:
        with open(json_examples_filepath, 'r', encoding='utf-8') as f:
            examples_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {json_examples_filepath} was not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from {json_examples_filepath}. Ensure it's a valid JSON array of objects.")

    if not isinstance(examples_data, list):
        raise ValueError("JSON content must be an array (list) of example objects.")

    documents_to_add = []
    ids_to_add = []

    for i, example in enumerate(examples_data):
        if not isinstance(example, dict):
            print(f"Warning: Item at index {i} in the JSON array is not a valid object (dictionary). Skipping.")
            continue

        nl_content = example.get("nl")
        if not nl_content or not isinstance(nl_content, str) or not nl_content.strip():
            print(f"Skipping example at index {i} due to missing or invalid 'nl' field: {example.get('id', 'N/A')}")
            continue

        user_provided_id = example.get("id") # Get the ID from JSON
        qdrant_point_id = None

        if user_provided_id is not None:
            # Try to treat as an integer first
            if isinstance(user_provided_id, int) and user_provided_id > 0: # Qdrant expects unsigned integers
                qdrant_point_id = user_provided_id
            elif isinstance(user_provided_id, str):
                try:
                    parsed_int_id = int(user_provided_id)
                    if parsed_int_id > 0:
                        qdrant_point_id = parsed_int_id
                    else:
                        # If it's a non-positive integer string, treat as potential UUID or generate new
                        pass # Fall through to UUID check or generation
                except ValueError:
                    # Not an integer string, try as UUID string
                    try:
                        UUID(user_provided_id) # Validate if it's a UUID format
                        qdrant_point_id = user_provided_id # Use as UUID string
                    except ValueError:
                        print(f"Warning: User-provided ID '{user_provided_id}' for example at index {i} is not a valid positive integer or UUID. Generating a new ID.")
                        # Fall through to generate new UUID
            else:
                # User provided ID is not int or string (e.g. float, bool) - invalid for Qdrant ID
                 print(f"Warning: User-provided ID '{user_provided_id}' (type: {type(user_provided_id)}) for example at index {i} is not a valid type (int or string). Generating a new ID.")
                 # Fall through to generate new UUID

        if qdrant_point_id is None: # If no valid ID found from user input or parsing failed
            qdrant_point_id = str(uuid4())
            print(f"Generated new UUID '{qdrant_point_id}' for example at index {i} (original user ID: {user_provided_id}).")


        ids_to_add.append(qdrant_point_id) # This list will contain either ints or UUID strings

        metadata = {
            "user_id_from_file": str(user_provided_id) if user_provided_id is not None else None, # Store original user-provided ID
            "sql": example.get("sql"),
            "tables": example.get("tables"),
            "type": example.get("type"),
            "qdrant_point_id_ref": str(qdrant_point_id) # Store reference to the actual Qdrant ID used
        }
        filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
        documents_to_add.append(
            Document(page_content=nl_content, metadata=filtered_metadata)
        )

    if documents_to_add:
        if len(documents_to_add) != len(ids_to_add):
            raise RuntimeError("Mismatch between number of documents and IDs generated for Qdrant.")
        try:
            print(f"Attempting to add {len(documents_to_add)} documents with IDs: {ids_to_add}") # Debug print
            vector_store_instance.add_documents(documents_to_add, ids=ids_to_add)
            return len(documents_to_add)
        except Exception as e:
            print(f"Error during Qdrant add_documents. IDs passed: {ids_to_add}. Documents: {len(documents_to_add)}")
            print(f"Qdrant client error details: {e}")
            raise RuntimeError(f"An error occurred while adding documents to Qdrant: {e}")
    return 0

def add_single_example_to_vector_store_logic(example_data: Dict[str, Any], vector_store_instance):
    if not vector_store_instance:
        raise ValueError("Vector store instance is not available.")
    nl_content = example_data.get("nl")
    if not nl_content or not nl_content.strip():
        raise ValueError("The 'nl' field is missing or empty in the provided example.")

    qdrant_point_id = str(uuid4()) 
    
    metadata = {
        "user_id": example_data.get("id"), 
        "sql": example_data.get("sql"),
        "tables": example_data.get("tables"),
        "type": example_data.get("type"),
        "qdrant_point_id_ref": qdrant_point_id
    }
    filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
    
    document_to_add = Document(page_content=nl_content, metadata=filtered_metadata)
    vector_store_instance.add_documents([document_to_add], ids=[qdrant_point_id])
    
    return {"qdrant_point_id": qdrant_point_id, "nl_content": nl_content}


def get_all_qdrant_points_logic(
    qdrant_client_instance,
    collection_name: str,
    limit: int = 10,
    offset: Optional[Any] = None,
    with_payload: bool = True,
    with_vectors: bool = False
) -> List[Dict[str, Any]]:
    if not qdrant_client_instance:
        raise ValueError("Qdrant client instance is not available.")
    try:
        points, next_offset = qdrant_client_instance.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors
        )
        result_points = []
        for point in points:
            point_dict = {"id": point.id, "payload": point.payload}
            if with_vectors and point.vector:
                point_dict["vector"] = point.vector
            result_points.append(point_dict)
        return result_points, next_offset
    except Exception as e:
        raise RuntimeError(f"Error retrieving points from Qdrant collection '{collection_name}': {e}")

def delete_qdrant_point_logic(
    qdrant_client_instance,
    collection_name: str,
    point_id: Any
) -> Dict[str, Any]:
    if not qdrant_client_instance:
        raise ValueError("Qdrant client instance is not available.")
    
    try:
        processed_point_id = None
        if isinstance(point_id, str):
            try:
                processed_point_id = int(point_id)
            except ValueError:
                processed_point_id = point_id 
        elif isinstance(point_id, int):
            processed_point_id = point_id
        else:
            raise ValueError(f"Point ID '{point_id}' must be an integer or a string (UUID).")
        operation_info = qdrant_client_instance.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=[processed_point_id]),
            wait=True 
        )
        status = getattr(operation_info, 'status', None)

        if status == models.UpdateStatus.COMPLETED or status == models.UpdateStatus.OK or status is None and operation_info is not None:
            return {"status": "success", "point_id_deleted": point_id, "details": str(operation_info)}
        else:
            raise RuntimeError(f"Failed to delete point {point_id}. Status: {status}, Info: {operation_info}")
            
    except Exception as e:
        raise RuntimeError(f"Error deleting point {point_id} from Qdrant collection '{collection_name}': {e}")