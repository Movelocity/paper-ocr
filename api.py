from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List
import cv2
import numpy as np
from PIL import Image
import asyncio
from functools import lru_cache

from pdftolatex.block_detector import segment
from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor

import os
os.environ["HTTPS_PROXY"] = "http://localhost:7890"
os.environ["HTTP_PROXY"] = "http://localhost:7890"
# Load the OCR model and processor once when the app starts
print("Loading Texify...")
model = load_model()
processor = load_processor()
os.environ["HTTPS_PROXY"] = ""
os.environ["HTTP_PROXY"] = ""

print("Loading RapidOCR...")
from rapidocr_onnxruntime import RapidOCR
rapidOcrEngine = RapidOCR()

app = FastAPI(
    title="Paper OCR API",
    description="API for performing OCR on academic paper images",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BoundingBox(BaseModel):
    x: float = Field(..., description="Fractional x-coordinate of the bounding box")
    y: float = Field(..., description="Fractional y-coordinate of the bounding box")
    w: float = Field(..., description="Fractional width of the bounding box")
    h: float = Field(..., description="Fractional height of the bounding box")
    text: str = Field(..., description="OCR text for this bounding box")
    idx: int = Field(..., description="The order of the box")

class OCRResponse(BaseModel):
    bounding_boxes: List[BoundingBox] = Field(..., description="List of bounding boxes with OCR results")

@lru_cache(maxsize=100)
def process_image(image_bytes: bytes, two_col: bool):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image format")
    
    bboxes = segment(img, two_col=two_col, preview=False)
    
    h, w = img.shape[:2]
    small_block_w = w/20
    results = []

    for idx, box in enumerate(bboxes):
        img_part = img[box.y:box.y+box.height, box.x:box.x+box.width, :]
        
        if box.width < small_block_w:
            result, _ = rapidOcrEngine(img_part)  # normal text
            try:
                ocr_result = result[0][1]
            except:
                ocr_result = ""
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img_part, cv2.COLOR_BGR2RGB))
            ocr_result = batch_inference([pil_img], model, processor)[0]  # latex
            if idx == 0:
                ocr_result = ocr_result.replace(r"$\pm$ 20% of the ", "")  # remove this error text in page eyebrow

        bbox = BoundingBox(
            x=box.x / w,
            y=box.y / h,
            w=box.width / w,
            h=box.height / h,
            text=ocr_result.strip(),
            idx=idx
        )
        results.append(bbox)

    
    return results

@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr(
    image: UploadFile = File(..., description="Image file to perform OCR on"),
    two_col: bool = Form(..., description="Whether the paper layout is two-column")
):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Only JPEG and PNG are supported.")
    
    contents = await image.read()
    
    try:
        results = await asyncio.to_thread(process_image, contents, two_col)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during processing")
    
    return OCRResponse(bounding_boxes=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)