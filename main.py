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



app = FastAPI()


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




@app.get("/start-session")
def start_session():
    session_id = generate_session_id()
    
    # Create session folders
    os.makedirs(f"main_files/{session_id}", exist_ok=True)
    os.makedirs(f"reference_files/{session_id}", exist_ok=True)
    
    return {"session_id": session_id, "message": "Session started"}




@app.post("/upload-main-file")
async def upload_main_file(session_id: str = Form(...), file: UploadFile = File(...)):
    folder_path = f"main_files/{session_id}"

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail="Invalid session ID.")

    # Save the uploaded file
    saved_path = await save_uploaded_file(file, folder_path)

    # Check file type and read it into a DataFrame
    try:
        if file.filename.endswith(".csv"):
            # df = pd.read_csv(saved_path)
            df = pd.read_csv(saved_path, nrows=2) 
        else:
            # df = pd.read_excel(saved_path)
            df = pd.read_excel(saved_path, nrows=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    # Get file size in KB
    file_size_kb = round(os.path.getsize(saved_path) / 1024, 2)
    preview_df = df.head(2)
    clean_preview_df = preview_df.astype(object).where(pd.notna(preview_df), None)
    preview_dict = clean_preview_df.to_dict(orient="records")

    return {
        "filename": file.filename,
        "size_kb": file_size_kb,
        "preview": preview_dict, # Use the cleaned-up version
        "session_id": session_id
    }


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

    # Load metadata
    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    uploaded = meta_data["uploaded_count"]
    total = meta_data["reference_count"]

    # ✅ Raise error if user tries to upload more files than expected
    if uploaded >= total:
        raise HTTPException(
            status_code=400,
            detail=f"Upload limit reached: you declared {total} reference files and already uploaded {uploaded}."
        )

    # Save the uploaded file
    saved_path = await save_uploaded_file(file, folder_path)

    # Read preview
    try:
        df = pd.read_csv(saved_path) if file.filename.endswith(".csv") else pd.read_excel(saved_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    # Update count
    meta_data["uploaded_count"] += 1
    with open(meta_path, "w") as f:
        json.dump(meta_data, f)

    # Prepare response
    file_size_kb = round(os.path.getsize(saved_path) / 1024, 2)
    preview_df = df.head(2)
    clean_preview_df = preview_df.astype(object).where(pd.notna(preview_df), None)
    preview_dict = clean_preview_df.to_dict(orient="records")

    return {
        "filename": file.filename,
        "size_kb": file_size_kb,
        "preview": preview_dict,
        "uploaded_reference_count": meta_data["uploaded_count"],
        "total_reference_expected": meta_data["reference_count"]
    }







@app.get("/columns/{session_id}")
async def get_columns(session_id: str):

    # Gets the column names from the uploaded main and reference files.

    main_folder = f"main_files/{session_id}"
    ref_folder = f"reference_files/{session_id}"

    if not os.path.exists(main_folder) or not os.path.exists(ref_folder):
        raise HTTPException(status_code=404, detail="Session not found.")

    response = {
        "main_file_columns": [],
        "reference_files_columns": {}
    }

    # --- Get Main File Columns ---
    # Find the first csv or xlsx file in the main folder
    main_files = glob.glob(f"{main_folder}/*.csv") + glob.glob(f"{main_folder}/*.xlsx")
    if main_files:
        main_file_path = main_files[0]
        try:
            # Efficiently read only the header row to get columns
            df_main_cols = pd.read_csv(main_file_path, nrows=0).columns.tolist() if main_file_path.endswith(".csv") else pd.read_excel(main_file_path, nrows=0).columns.tolist()
            response["main_file_columns"] = df_main_cols
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading main file: {e}")

    # --- Get Reference Files Columns ---
    ref_files = glob.glob(f"{ref_folder}/*.csv") + glob.glob(f"{ref_folder}/*.xlsx")
    for file_path in ref_files:
        filename = os.path.basename(file_path)
        try:
            # Read only the header for each reference file
            df_ref_cols = pd.read_csv(file_path, nrows=0).columns.tolist() if file_path.endswith(".csv") else pd.read_excel(file_path, nrows=0).columns.tolist()
            response["reference_files_columns"][filename] = df_ref_cols
        except Exception as e:
            # If one file fails, we can note it and continue
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
    results_folder = f"results/{session_id}" # Define a folder for results
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
        # This example handles one reference file; can be extended for more
        ref_filename = list(ref_keys.keys())[0]
        ref_key_column = ref_keys[ref_filename]
        ref_file_path = os.path.join(ref_folder, ref_filename)
        
        # Load the entire reference file
        ref_df = pd.read_excel(ref_file_path) if ref_file_path.endswith(".xlsx") else pd.read_csv(ref_file_path)
        
        # Set the key column as the index for super-fast lookups
        ref_df.set_index(ref_key_column, inplace=True)
        print(f"[{session_id}] Reference file indexed successfully.")
    except Exception as e:
        print(f"[{session_id}] Error loading reference file: {e}. Aborting.")
        return

    # --- 3. Process Main File and Apply Rules ---
    main_key_column = key_config["main_file_key"]
    main_files = glob.glob(f"{main_folder}/*.csv") + glob.glob(f"{main_folder}/*.xlsx")
    if not main_files:
        print(f"[{session_id}] Error: Main file not found. Aborting.")
        return
    main_file_path = main_files[0]
    
    # List to store records of changed rows for the summary report
    changes_summary = []
    # List to store the processed (and modified) chunks
    processed_chunks = []

    print(f"[{session_id}] Starting to process main file in chunks...")
    chunk_iterator = pd.read_excel(main_file_path, chunksize=5000) if main_file_path.endswith(".xlsx") else pd.read_csv(main_file_path, chunksize=5000)

    for chunk in chunk_iterator:
        original_chunk = chunk.copy() # Keep a copy to compare for changes

        def apply_rules_to_row(row):
            main_key_value = row[main_key_column]
            
            try:
                # Use the index to instantly find the reference row
                ref_row = ref_df.loc[main_key_value]
            except KeyError:
                # No matching key in the reference file, so can't apply rules
                return row

            # Now, apply all configured rules to this row
            for rule in rules_config:
                main_col = rule["main_column"]
                ref_col = rule["reference_column"]
                main_val = row[main_col]
                ref_val = ref_row[ref_col]
                
                # Condition: Main value is empty/null
                if rule["primary_condition"] == "IS_EMPTY" and pd.isna(main_val):
                    if rule["action"] == "REPLACE" and pd.notna(ref_val):
                        row[main_col] = ref_val # Modify the row

                # Condition: Main value is NOT empty
                elif rule["primary_condition"] == "IS_NOT_EMPTY" and pd.notna(main_val):
                    op = rule["comparison_operator"]
                    if (op == "DIFFERENT" and main_val != ref_val) or \
                       (op == "MATCH" and main_val == ref_val):
                        if rule["action"] == "REPLACE":
                            row[main_col] = ref_val # Modify the row
            return row

        # Apply the logic to every row in the chunk
        modified_chunk = chunk.apply(apply_rules_to_row, axis=1)
        
        # --- Track Changes for Summary Report ---
        # Compare the modified chunk with the original one
        diff = original_chunk != modified_chunk
        changed_indices = diff.any(axis=1)
        if changed_indices.any():
            changed_original = original_chunk[changed_indices]
            changed_modified = modified_chunk[changed_indices]
            # You can format this summary however you like
            for index in changed_original.index:
                changes_summary.append({
                    "key": changed_original.loc[index, main_key_column],
                    "original": changed_original.loc[index].to_dict(),
                    "modified": changed_modified.loc[index].to_dict()
                })

        processed_chunks.append(modified_chunk)

    # --- 4. Save Results ---
    # Combine all processed chunks into a final DataFrame
    final_df = pd.concat(processed_chunks, ignore_index=True)
    
    # Save the full, modified file
    final_output_path = os.path.join(results_folder, "final_output.csv")
    final_df.to_csv(final_output_path, index=False)
    
    # Save the summary of changes
    summary_df = pd.DataFrame(changes_summary)
    summary_output_path = os.path.join(results_folder, "changes_summary.csv")
    summary_df.to_csv(summary_output_path, index=False)
    
    print(f"[{session_id}] Comparison finished. Results saved.")
    
    # --- 5. Update Status ---
    config["status"] = "completed"
    config["results"] = {
        "full_output": final_output_path,
        "summary_report": summary_output_path
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






@app.get("/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """
    Serves the result files for download.
    """
    file_path = os.path.join("results", session_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
        
    return FileResponse(path=file_path, media_type='application/octet-stream', filename=filename)


















