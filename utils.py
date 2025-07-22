import os
from pathlib import Path
from fastapi import UploadFile
import random
import string
import shutil



async def save_uploaded_file(file, upload_dir: str) -> str:
    # Ensure the upload directory exists
    Path(upload_dir).mkdir(parents=True, exist_ok=True)

        # Make sure the file has a name
    if not file.filename:
        raise ValueError("Uploaded file has no filename")
    
    file_path = Path(upload_dir) / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"Full path: {file_path}")
    return str(file_path)



def generate_session_id(prefix="DTS", length=4) -> str:
    """Generate a short readable session ID like DTS-4KF8"""
    random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    return f"{prefix}-{random_part}"