from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from preprocess import extract_frames, save_upload_file_tmp
from model import model_instance

app = FastAPI(title="Video Action Recognition API")

# CORS Setup - Allow All for Demo/Dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Video Action Recognition API is execution"}

@app.post("/predict")
async def predict_action(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")
    
    # Save to temp file
    temp_video_path = save_upload_file_tmp(file)
    
    try:
        # Preprocess
        input_data = extract_frames(temp_video_path)
        
        # Inference
        result = model_instance.predict(input_data)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
