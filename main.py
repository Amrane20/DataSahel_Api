import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, List
import pandas as pd
from utils import save_uploaded_file
from utils import save_uploaded_file, generate_session_id
import json
import glob
from enum import Enum
from fastapi.responses import FileResponse 
from fastapi.middleware.cors import CORSMiddleware

import boto3
from botocore.exceptions import NoCredentialsError

app = FastAPI()

# =================================================================
origins = [
    # "https://your-datasahel-ui-url.vercel.app",
    # "http://localhost:5173",
    # "http://localhost:3000", 
    # "http://localhost",    
    "http://localhost:5173",
    "https://data-sahel-pv81uo9ot-amranes-projects.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
)
# =================================================================

class KeyConfig(BaseModel):
    session_id: str
    main_file_key: str
    reference_files_keys: Dict[str, str] # A dictionary for {filename: key_column}


# Enums to represent the user's choices from dropdowns
class PrimaryCondition(str, Enum):
    IS_EMPTY = "IS_EMPTY"
    IS_NOT_EMPTY = "IS_NOT_EMPTY"

class ComparisonOperator(str, Enum):
    DIFFERENT = "DIFFERENT"
    MATCH = "MATCH"

class Action(str, Enum):
    KEEP = "KEEP"
    REPLACE = "REPLACE"

# A Pydantic model for a single rule
class ComparisonRule(BaseModel):
    main_column: str
    reference_column: str
    primary_condition: PrimaryCondition
    comparison_operator: Optional[ComparisonOperator] = None # Only used if primary_condition is IS_NOT_EMPTY
    action: Action

# The main model for the request body
class RuleSetConfig(BaseModel):
    session_id: str
    rules: List[ComparisonRule]


# This model defines a SINGLE new column configuration
class NewColumnInfo(BaseModel):
    source_columns: List[str]
    new_column_name: str
    prefix: Optional[str] = ""
    suffix: Optional[str] = ""

# This is the main model for the request, which now contains a LIST of new columns
class CreateColumnsConfig(BaseModel):
    session_id: str
    main_file_key: str
    reference_file_key: str
    new_columns: List[NewColumnInfo] # This is now a list


@app.get("/")
def read_root():
    return {"message": "DataSahel Says Hello"}


@app.get("/start-session")
def start_session():
    session_id = generate_session_id()
    
    # Create session folders
    os.makedirs(f"main_files/{session_id}", exist_ok=True)
    os.makedirs(f"reference_files/{session_id}", exist_ok=True)
    
    return {"session_id": session_id, "message": "Session started"}


# old one
# @app.post("/upload-main-file")
# async def upload_main_file(session_id: str = Form(...), file: UploadFile = File(...)):
#     folder_path = f"main_files/{session_id}"

#     if not os.path.exists(folder_path):
#         raise HTTPException(status_code=400, detail="Invalid session ID.")

#     # Save the uploaded file temporarily
#     temp_path = await save_uploaded_file(file, folder_path)
#     final_path = temp_path
#     final_filename = file.filename

#     # If the file is an Excel file, convert it to CSV
#     if temp_path.endswith((".xlsx", ".xls")):
#         try:
#             df = pd.read_excel(temp_path)
#             # Create a new path for the CSV file with the same base name
#             final_filename = Path(temp_path).stem + ".csv"
#             final_path = os.path.join(folder_path, final_filename)
#             df.to_csv(final_path, index=False)
#             # Remove the original Excel file
#             os.remove(temp_path)
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error converting Excel to CSV: {e}")

#     # Now, we read the CSV for the preview
#     try:
#         df_preview = pd.read_csv(final_path, nrows=2)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error reading file preview: {e}")

#     # Clean up the preview for JSON response
#     clean_preview_df = df_preview.astype(object).where(pd.notna(df_preview), None)
#     preview_dict = clean_preview_df.to_dict(orient="records")
#     file_size_kb = round(os.path.getsize(final_path) / 1024, 2)

#     return {
#         "filename": final_filename, # Return the new .csv filename
#         "size_kb": file_size_kb,
#         "preview": preview_dict,
#         "session_id": session_id
#     }



@app.post("/upload-main-file")
async def upload_main_file(session_id: str = Form(...), file: UploadFile = File(...)):
    # The key is the full path in the S3 bucket
    file_key = f"{session_id}/main_files/{file.filename}"
    
    try:
        # Upload the file directly to S3
        s3_client.upload_fileobj(file.file, S3_BUCKET_NAME, file_key)
        
        # For simplicity, we'll stop converting to CSV for now
        # and assume the user uploads a CSV. We can add this back later.
        
        # To get a preview, we need to read the file from S3
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        # Read only the first few lines for the preview
        file_content = response['Body'].read().decode('utf-8').splitlines()
        
        # This is a simplified preview logic for CSV
        header = file_content[0].split(',')
        first_row = file_content[1].split(',')
        second_row = file_content[2].split(',')
        
        preview_dict = [
            dict(zip(header, first_row)),
            dict(zip(header, second_row))
        ]

        # Get file size from S3 metadata
        head_object = s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        file_size_kb = round(head_object['ContentLength'] / 1024, 2)

    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not available.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred with S3: {e}")

    return {
        "filename": file.filename,
        "size_kb": file_size_kb,
        "preview": preview_dict,
        "session_id": session_id
    }


@app.get("/debug-env")
async def debug_env():
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    
    return {
        "aws_access_key_id_is_set": bool(access_key),
        "aws_secret_access_key_is_set": bool(secret_key),
        "s3_bucket_name_is_set": bool(bucket_name)
    }

@app.delete("/delete-main-file/{session_id}/{filename}")
async def delete_main_file(session_id: str, filename: str):
    file_path = os.path.join("main_files", session_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        os.remove(file_path)
        return {"message": "File deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")
    


@app.delete("/delete-ref-file/{session_id}/{filename}")
async def delete_ref_file(session_id: str, filename: str):
    meta_path = os.path.join("reference_files", session_id, "meta.json")
    file_path = os.path.join("reference_files", session_id, filename)

    # --- 1. Update metadata count ---
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r+') as f:
                meta_data = json.load(f)
                # Ensure count is greater than 0 before decrementing
                if meta_data.get("uploaded_count", 0) > 0:
                    meta_data["uploaded_count"] -= 1
                
                # Go back to the beginning of the file to overwrite it
                f.seek(0)
                json.dump(meta_data, f, indent=4)
                f.truncate() # Remove old content if the new content is shorter
        except (IOError, json.JSONDecodeError) as e:
            # Handle cases where meta.json might be corrupted or unreadable
            raise HTTPException(status_code=500, detail=f"Error processing metadata: {e}")

    # --- 2. Delete the actual file ---
    if not os.path.exists(file_path):
        # Even if the file is not found, the metadata might have been corrected,
        # so we don't raise an error unless metadata also failed.
        # We can just return a success message.
        return {"message": "File not found, metadata may have been corrected."}
    
    try:
        os.remove(file_path)
        return {"message": "File and metadata updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")




@app.post("/set-reference-count")
async def set_reference_count(session_id: str = Form(...), reference_count: int = Form(...)):
    folder_path = f"reference_files/{session_id}"
    
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail="Invalid session ID.")
    
    meta_path = os.path.join(folder_path, "meta.json")
    
    meta_data = {
        "reference_count": reference_count,
        "uploaded_count": 0
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta_data, f)
    
    return {
        "message": f"Expected {reference_count} reference files.",
        "session_id": session_id
    }


@app.post("/upload-reference-file")
async def upload_reference_file(session_id: str = Form(...), file: UploadFile = File(...)):
    folder_path = f"reference_files/{session_id}"

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail="Invalid session ID.")

    meta_path = os.path.join(folder_path, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=400, detail="Missing metadata. Set reference count first.")

    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    if meta_data["uploaded_count"] >= meta_data["reference_count"]:
        raise HTTPException(status_code=400, detail="Upload limit reached.")

    # Save, convert if necessary, and clean up
    temp_path = await save_uploaded_file(file, folder_path)
    final_path = temp_path
    final_filename = file.filename

    if temp_path.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(temp_path)
            final_filename = Path(temp_path).stem + ".csv"
            final_path = os.path.join(folder_path, final_filename)
            df.to_csv(final_path, index=False)
            os.remove(temp_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting Excel to CSV: {e}")
            
    # Update count and save metadata
    meta_data["uploaded_count"] += 1
    with open(meta_path, "w") as f:
        json.dump(meta_data, f)
    
    # Prepare and return response
    try:
        df_preview = pd.read_csv(final_path, nrows=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file preview: {e}")

    clean_preview_df = df_preview.astype(object).where(pd.notna(df_preview), None)
    preview_dict = clean_preview_df.to_dict(orient="records")
    file_size_kb = round(os.path.getsize(final_path) / 1024, 2)

    return {
        "filename": final_filename,
        "size_kb": file_size_kb,
        "preview": preview_dict,
        "uploaded_reference_count": meta_data["uploaded_count"],
        "total_reference_expected": meta_data["reference_count"]
    }




@app.get("/columns/{session_id}")
async def get_columns(session_id: str):
    """
    Gets the column names from the uploaded main and reference files,
    and also returns the name of the main file.
    """
    main_folder = f"main_files/{session_id}"
    ref_folder = f"reference_files/{session_id}"

    if not os.path.exists(main_folder) or not os.path.exists(ref_folder):
        raise HTTPException(status_code=404, detail="Session not found.")

    # Updated response structure to include the main file's name
    response = {
        "main_file_name": None,
        "main_file_columns": [],
        "reference_files_columns": {}
    }

    # --- Get Main File Name and Columns ---
    # At this stage, the file has been converted to CSV, so we only look for .csv
    main_files = glob.glob(f"{main_folder}/*.csv")
    if main_files:
        main_file_path = main_files[0]
        
        # --- NEW: Get the filename from the path ---
        main_filename = os.path.basename(main_file_path)
        response["main_file_name"] = main_filename
        
        try:
            df_main_cols = pd.read_csv(main_file_path, nrows=0).columns.tolist()
            response["main_file_columns"] = df_main_cols
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading main file: {e}")

    # --- Get Reference Files Columns (No changes here) ---
    ref_files = glob.glob(f"{ref_folder}/*.csv")
    for file_path in ref_files:
        filename = os.path.basename(file_path)
        try:
            df_ref_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
            response["reference_files_columns"][filename] = df_ref_cols
        except Exception as e:
            response["reference_files_columns"][filename] = f"Error reading file: {e}"
            
    return response




@app.post("/configure/keys")
async def configure_keys(config: KeyConfig):

    # Saves the user's choice of key columns for the session.

    meta_path = f"reference_files/{config.session_id}/meta.json"

    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Session metadata not found. Have you set the reference count?")

    # Read existing metadata
    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    # Add the new key configuration
    meta_data["key_config"] = {
        "main_file_key": config.main_file_key,
        "reference_files_keys": config.reference_files_keys
    }

    # Write the updated metadata back to the file
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=4)

    return {"message": "Key columns have been configured successfully.", "config": meta_data["key_config"]}




@app.post("/configure/rules")
async def configure_rules(config: RuleSetConfig):

    meta_path = f"reference_files/{config.session_id}/meta.json"

    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Session metadata not found.")

    # Read existing metadata
    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    # Add the new rules configuration
    meta_data["rules_config"] = config.dict(exclude={"session_id"})

    # Write the updated metadata back to the file
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=4)

    return {"message": "Comparison rules have been configured successfully.", "config": meta_data["rules_config"]}




def execute_comparison_logic(session_id: str):
    print(f"[{session_id}] Starting comparison process...")
    meta_path = f"reference_files/{session_id}/meta.json"
    main_folder = f"main_files/{session_id}"
    ref_folder = f"reference_files/{session_id}"
    results_folder = f"results/{session_id}"
    os.makedirs(results_folder, exist_ok=True)

    # --- 1. Load Configuration ---
    try:
        with open(meta_path, 'r') as f:
            config = json.load(f)
        key_config = config["key_config"]
        rules_config = config["rules_config"]["rules"]
    except (IOError, KeyError) as e:
        print(f"[{session_id}] Error: Configuration is missing or invalid: {e}. Aborting.")
        return

    # --- 2. Load Reference File and Create Indexed Lookup ---
    try:
        ref_keys = key_config["reference_files_keys"]
        ref_filename = list(ref_keys.keys())[0] # Assumes one reference file
        ref_key_column = ref_keys[ref_filename]
        ref_file_path = os.path.join(ref_folder, ref_filename)
        ref_df = pd.read_csv(ref_file_path)
        ref_df.set_index(ref_key_column, inplace=True, drop=False)
        print(f"[{session_id}] Reference file indexed successfully.")
    except Exception as e:
        print(f"[{session_id}] Error loading reference file: {e}. Aborting.")
        return

    # --- 3. Process Main File and Apply Rules ---
    main_key_column = key_config["main_file_key"]
    main_file_path = glob.glob(f"{main_folder}/*.csv")[0]
    
    changes_summary = []
    processed_chunks = []

    print(f"[{session_id}] Starting to process main file...")
    # The file is always CSV, so we can reliably use chunksize
    chunk_iterator = pd.read_csv(main_file_path, chunksize=5000)

    for chunk in chunk_iterator:
        original_chunk = chunk.copy()

        def apply_rules_to_row(row):
            main_key_value = row[main_key_column]
            try:
                ref_row = ref_df.loc[main_key_value]
            except KeyError:
                return row

            for rule in rules_config:
                main_col = rule["main_column"]
                ref_col = rule["reference_column"]
                if rule["primary_condition"] == "IS_EMPTY" and pd.isna(row[main_col]):
                    if rule["action"] == "REPLACE" and pd.notna(ref_row[ref_col]):
                        row[main_col] = ref_row[ref_col]
                elif rule["primary_condition"] == "IS_NOT_EMPTY" and pd.notna(row[main_col]):
                    op = rule["comparison_operator"]
                    if (op == "DIFFERENT" and row[main_col] != ref_row[ref_col]) or \
                       (op == "MATCH" and row[main_col] == ref_row[ref_col]):
                        if rule["action"] == "REPLACE":
                            row[main_col] = ref_row[ref_col]
            return row

        modified_chunk = chunk.apply(apply_rules_to_row, axis=1)
        
        # Track Changes for Summary Report (Wide Format)
        diff_mask = original_chunk.ne(modified_chunk) & (original_chunk.notna() | modified_chunk.notna())
        changed_rows_indices = diff_mask.any(axis=1)

        if changed_rows_indices.any():
            original_changed = original_chunk[changed_rows_indices]
            modified_changed = modified_chunk[changed_rows_indices]
            for index in original_changed.index:
                key_value = original_changed.loc[index, main_key_column]
                change_record = {'ID': key_value}
                changed_cols = diff_mask.loc[index][diff_mask.loc[index]].index.tolist()
                for col in changed_cols:
                    change_record[f"{col}_old_value"] = original_changed.loc[index, col]
                    change_record[f"{col}_new_value"] = modified_changed.loc[index, col]
                changes_summary.append(change_record)

        processed_chunks.append(modified_chunk)

    # --- 4. Save Results ---
    final_df = pd.concat(processed_chunks, ignore_index=True)
    final_output_path = os.path.join(results_folder, "final_output.xlsx")
    final_df.to_excel(final_output_path, index=False)
    
    summary_df = pd.DataFrame(changes_summary)
    summary_output_path = os.path.join(results_folder, "changes_summary.xlsx")
    summary_df.to_excel(summary_output_path, index=False)
    
    print(f"[{session_id}] Comparison finished. Results saved.")
    
    # --- 5. Update Status ---
    config["status"] = "completed"
    config["results"] = {
        "full_output": "final_output.xlsx",
        "summary_report": "changes_summary.xlsx"
    }
    with open(meta_path, 'w') as f:
        json.dump(config, f, indent=4)



@app.post("/execute-comparison/{session_id}")
async def execute_comparison(session_id: str, background_tasks: BackgroundTasks):
    """
    Triggers the background task to run the comparison logic.
    """
    meta_path = f"reference_files/{session_id}/meta.json"
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Session not found or not fully configured.")

    # Add the main logic function to run in the background
    background_tasks.add_task(execute_comparison_logic, session_id)

    # Return an immediate response to the user
    return {"message": "Comparison process has been started. You will be notified upon completion."}



@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """
    Checks the status of the comparison process for a given session.
    """
    meta_path = f"reference_files/{session_id}/meta.json"
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Session not found.")

    with open(meta_path, 'r') as f:
        config = json.load(f)

    status = config.get("status", "processing")
    results = config.get("results", None)

    return {"status": status, "results": results}


def create_columns_logic(session_id: str, config_dict: dict):
    """This background task now processes a LIST of new columns."""
    print(f"[{session_id}] Starting 'Create Columns' background task...")
    results_folder = f"results/{session_id}"
    meta_path = f"reference_files/{session_id}/meta.json"
    os.makedirs(results_folder, exist_ok=True)

    try:
        config = CreateColumnsConfig(**config_dict)
        main_file_path = glob.glob(f"main_files/{session_id}/*.csv")[0]
        ref_file_path = glob.glob(f"reference_files/{session_id}/*.csv")[0]
        
        main_df = pd.read_csv(main_file_path)
        ref_df = pd.read_csv(ref_file_path)

        merged_df = pd.merge(
            main_df, ref_df,
            left_on=config.main_file_key, right_on=config.reference_file_key,
            how='left', suffixes=('', '_ref')
        )

        # Loop through each new column configuration from the list
        for column_info in config.new_columns:
            def create_new_value(row):
                if pd.isna(row[column_info.source_columns[0]]):
                    return ""
                
                combined_values = [str(row[col]) for col in column_info.source_columns if col in row and pd.notna(row[col])]
                final_string = " ".join(combined_values)
                return column_info.prefix + final_string + column_info.suffix

            merged_df[column_info.new_column_name] = merged_df.apply(create_new_value, axis=1)

        # Create a list of all the new column names that were added
        new_column_names = [col.new_column_name for col in config.new_columns]
        
        # Clean up the final DataFrame
        final_df = merged_df[main_df.columns.tolist() + new_column_names]
        
        output_filename = "output_with_new_columns.xlsx"
        output_path = os.path.join(results_folder, output_filename)
        final_df.to_excel(output_path, index=False)
        print(f"[{session_id}] Output file saved successfully.")

        # Update meta.json with the final status
        if os.path.exists(meta_path):
            with open(meta_path, 'r+') as f:
                meta_data = json.load(f)
                meta_data["status"] = "completed"
                meta_data["results"] = {"output_file": output_filename}
                f.seek(0)
                json.dump(meta_data, f, indent=4)
                f.truncate()
        
    except Exception as e:
        print(f"[{session_id}] FATAL ERROR in background task: {e}")
        # Update meta.json with a failed status
        if os.path.exists(meta_path):
            with open(meta_path, 'r+') as f:
                meta_data = json.load(f)
                meta_data["status"] = "failed"
                meta_data["error"] = str(e)
                f.seek(0)
                json.dump(meta_data, f, indent=4)
                f.truncate()


@app.post("/services/create-column")
async def create_columns_with_string(config: CreateColumnsConfig, background_tasks: BackgroundTasks):
    """This endpoint now accepts a list of new columns to create."""
    session_id = config.session_id
    meta_path = f"reference_files/{session_id}/meta.json"

    if os.path.exists(meta_path):
        with open(meta_path, 'r+') as f:
            meta_data = json.load(f)
            meta_data["status"] = "processing"
            meta_data["results"] = None
            f.seek(0)
            json.dump(meta_data, f, indent=4)
            f.truncate()

    background_tasks.add_task(create_columns_logic, session_id, config.dict())
    
    return {"message": "The 'Create Columns' process has been started."}




@app.get("/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """
    Serves the result files for download.
    """
    file_path = os.path.join("results", session_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
        
    return FileResponse(path=file_path, media_type='application/octet-stream', filename=filename)

















