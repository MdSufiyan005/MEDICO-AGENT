# import shutil
# import uuid
# import logging
# from pathlib import Path
# from fastapi import UploadFile, HTTPException

# from MedAge.config import UPLOAD_DIR

# logger = logging.getLogger(__name__)

# async def save_upload_file(file: UploadFile) -> str:
#     if not file.content_type.startswith('image/'):
#         raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
#     file_extension = Path(file.filename).suffix or ".png"
#     unique_filename = f"{uuid.uuid4()}{file_extension}"
#     file_path = UPLOAD_DIR / unique_filename
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     logger.info(f"✅ Saved uploaded file: {file_path}")
#     return str(file_path)
