import streamlit as st
import requests
import json
import pandas as pd

# --- Configuration ---
FASTAPI_BASE_URL = "http://localhost:8000"
ADMIN_PASSWORD = "admin"

# --- Helper Functions to Interact with FastAPI ---

def process_user_query(user_question: str):
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/process-query", json={"user_question": user_question})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding JSON response from backend.")
        return {"error_message": "Invalid JSON response."}


def add_single_example(example_data: dict):
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/add-single-example", json=example_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error adding example: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding JSON response from backend.")
        return {"message": "Failed to add example due to invalid JSON response."}


def add_examples_from_file(uploaded_file): # Renamed function for clarity internally
    if uploaded_file is not None:
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            # CORRECTED ENDPOINT URL HERE
            response = requests.post(f"{FASTAPI_BASE_URL}/add-examples", files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error uploading file: {e}")
            return None
        except json.JSONDecodeError:
            st.error("Error decoding JSON response from backend.")
            return {"message": "Failed to upload due to invalid JSON response."}
    return None

def get_all_examples(limit: int = 10, offset: str = None, with_payload: bool = True, with_vectors: bool = False):
    params = {"limit": limit, "with_payload": with_payload, "with_vectors": with_vectors}
    if offset:
        params["offset"] = offset
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/get-all-examples", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching examples: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding JSON response from backend.")
        return {"points": [], "count": 0, "next_offset": None}


def delete_example_by_id(point_id: str):
    try:
        response = requests.delete(f"{FASTAPI_BASE_URL}/delete-example/{point_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error deleting example {point_id}: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding JSON response from backend.")
        return {"message": f"Failed to delete example {point_id} due to invalid JSON response."}

def get_health_check():
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/health")
        response.raise_for_status()
        return response.json()
    except:
        return {"status": "error", "detail": "Backend not reachable"}


# --- Streamlit UI ---

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "mode" not in st.session_state:
    st.session_state.mode = "User"
if "current_nl_query" not in st.session_state:
    st.session_state.current_nl_query = ""
if "current_sql_query" not in st.session_state:
    st.session_state.current_sql_query = ""
if "current_tables" not in st.session_state:
    st.session_state.current_tables = []
if "current_query_type" not in st.session_state:
    st.session_state.current_query_type = ""


st.set_page_config(layout="wide", page_title="NL to SQL Query Engine")
st.title("Natural Language to SQL Query Engine")

if not st.session_state.logged_in:
    st.sidebar.header("Login")
    password = st.sidebar.text_input("Enter Admin Password:", type="password", key="admin_login_password")
    if st.sidebar.button("Login as Admin"):
        if password == ADMIN_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.mode = "Admin"
            st.rerun()
        else:
            st.sidebar.error("Incorrect password.")
    if st.sidebar.button("Continue as User"):
        st.session_state.logged_in = True
        st.session_state.mode = "User"
        st.rerun()
else:
    st.sidebar.header(f"Mode: {st.session_state.mode}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.mode = "User"
        st.session_state.current_nl_query = ""
        st.session_state.current_sql_query = ""
        st.session_state.current_tables = []
        st.session_state.current_query_type = ""
        st.rerun()

    if st.session_state.logged_in:
        if st.session_state.mode == "User":
            st.header("Ask Your Question")
            user_question = st.text_area("Enter your natural language query here:", height=100, key="user_nl_query_user_mode")
            if st.button("Get Answer", key="user_get_answer"):
                if user_question:
                    with st.spinner("Processing..."):
                        result = process_user_query(user_question)
                    if result:
                        if result.get("error_message"):
                            st.error(f"Error: {result['error_message']}")
                        elif result.get("nl_response"):
                            st.success("Your Answer:")
                            st.markdown(result["nl_response"])
                        else:
                            st.warning("No answer received or unfamiliar response.")
                else:
                    st.warning("Please enter a question.")

        elif st.session_state.mode == "Admin":
            st.header("Admin Dashboard")
            admin_action = st.selectbox(
                "Choose an action:",
                ["Test NL to SQL", "Add Single Example", "Add Examples from File", "View All Examples", "Delete Example", "Backend Health"],
                key="admin_action_selectbox"
            )

            if admin_action == "Test NL to SQL":
                st.subheader("Test NL to SQL")
                nl_query_admin = st.text_area("Natural Language Query:", height=100, key="admin_nl_query")
                if st.button("Process", key="admin_process_nl"):
                    if nl_query_admin:
                        with st.spinner("Processing..."):
                            result = process_user_query(nl_query_admin)
                        if result:
                            st.session_state.current_nl_query = nl_query_admin
                            st.json(result)

                            if result.get("analysis") and result.get("analysis").get("relevant") in ["yes", "maybe"]:
                                st.session_state.current_sql_query = result.get("generated_sql", "")
                                st.session_state.current_tables = result.get("analysis", {}).get("relevant_tables", [])
                                st.session_state.current_query_type = ", ".join(result.get("analysis", {}).get("query_types", []))
                            else:
                                st.session_state.current_sql_query = ""
                                st.session_state.current_tables = []
                                st.session_state.current_query_type = ""
                        else:
                            st.error("No response from backend.")
                    else:
                        st.warning("Please enter a query.")

                if st.session_state.current_nl_query and st.session_state.current_sql_query:
                    st.markdown("---")
                    st.markdown("#### Add this result as an example?")
                    with st.expander("Edit and Add Example"):
                        edited_nl = st.text_area("NL Query:", value=st.session_state.current_nl_query, key="edit_nl_for_example")
                        edited_sql = st.text_area("SQL Query:", value=st.session_state.current_sql_query, key="edit_sql_for_example")
                        edited_tables_str = st.text_input("Relevant Tables (Comma-separated):", value=", ".join(st.session_state.current_tables), key="edit_tables_for_example")
                        edited_type = st.text_input("Query Type(s) (Comma-separated):", value=st.session_state.current_query_type, key="edit_type_for_example")
                        example_id = st.text_input("Example ID (Optional):", key="edit_id_for_example")

                        if st.button("Add Example to Vector Store", key="admin_add_edited_example"):
                            if edited_nl and edited_sql:
                                tables_list = [t.strip() for t in edited_tables_str.split(',') if t.strip()]
                                example_data = {
                                    "nl": edited_nl, "sql": edited_sql, "tables": tables_list,
                                    "type": edited_type, "id": example_id if example_id else None,
                                }
                                add_response = add_single_example(example_data)
                                if add_response:
                                    st.success(f"Example added: {add_response.get('message', '')}")
                                    st.json(add_response.get("added_document_info", {}))
                                    st.session_state.current_nl_query = ""
                                    st.session_state.current_sql_query = ""
                                else:
                                    st.error("Failed to add example.")
                            else:
                                st.warning("NL and SQL are required to add an example.")


            elif admin_action == "Add Single Example":
                st.subheader("Add Single Example")
                with st.form("add_single_example_form"):
                    nl = st.text_area("Natural Language Query:", key="single_nl")
                    sql = st.text_area("SQL Query:", key="single_sql")
                    tables_str = st.text_input("Relevant Tables (Comma-separated):", key="single_tables")
                    q_type = st.text_input("Query Type:", key="single_type")
                    ex_id = st.text_input("Example ID (Optional):", key="single_id")
                    submitted = st.form_submit_button("Add Example")

                    if submitted:
                        if nl and sql:
                            tables_list = [t.strip() for t in tables_str.split(',') if t.strip()]
                            example_data = {
                                "nl": nl, "sql": sql, "tables": tables_list,
                                "type": q_type, "id": ex_id if ex_id else None
                            }
                            response = add_single_example(example_data)
                            if response:
                                st.success(f"Response: {response.get('message', '')}")
                                st.json(response.get("added_document_info", {}))
                        else:
                            st.warning("NL and SQL are required.")

            elif admin_action == "Add Examples from File": # This UI text is fine
                st.subheader("Add Examples from File")
                uploaded_file = st.file_uploader("Choose a JSON file", type=["json"], key="admin_file_uploader")
                if uploaded_file is not None:
                    if st.button("Upload and Add File", key="admin_upload_json"):
                        # The function call add_examples_from_file now uses the corrected endpoint
                        response = add_examples_from_file(uploaded_file)
                        if response:
                            st.success(response.get("message", "File processed."))
                        else:
                            st.error("Failed to upload file.")

            elif admin_action == "View All Examples":
                st.subheader("View All Examples")
                limit_view = st.number_input("Items per page:", min_value=1, max_value=100, value=10, key="view_limit")
                offset_view = st.text_input("Offset (for next page):", key="view_offset_input")

                if st.button("Fetch Examples", key="admin_fetch_examples"):
                    with st.spinner("Fetching..."):
                        data = get_all_examples(limit=limit_view, offset=offset_view if offset_view else None)
                    if data and "points" in data:
                        st.success(f"{data.get('count', 0)} examples found. Next offset: {data.get('next_offset', 'None')}")
                        if data["points"]:
                            try:
                                df_data = []
                                for p in data["points"]:
                                    row = {"qdrant_id": p.get("id")}
                                    if p.get("payload"):
                                        row.update(p["payload"])
                                    df_data.append(row)
                                df = pd.DataFrame(df_data)
                                st.dataframe(df)
                            except Exception as e:
                                st.write(f"Error displaying payloads as table ({e}), showing raw JSON:")
                                st.json(data["points"])
                        else:
                            st.info("No examples found.")
                    else:
                        st.error("Failed to fetch examples.")


            elif admin_action == "Delete Example":
                st.subheader("Delete Example")
                point_id_to_delete = st.text_input("Qdrant Point ID to delete:", key="admin_delete_point_id")
                if st.button("Delete Example", key="admin_confirm_delete"):
                    if point_id_to_delete:
                        response = delete_example_by_id(point_id_to_delete)
                        if response:
                            st.success(response.get("message", "Deletion attempt processed."))
                            st.json(response)
                    else:
                        st.warning("Please enter a point ID.")

            elif admin_action == "Backend Health":
                st.subheader("Backend Health")
                if st.button("Check Health", key="admin_check_health"):
                    health_status = get_health_check()
                    st.json(health_status)

st.markdown("---")
st.markdown("NL-SQL App | Built with FastAPI & Streamlit")