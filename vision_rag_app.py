from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from typing import Optional, List, Dict
import io
from PIL import Image
import fitz  # PyMuPDF
import os
from enum import Enum
from byaldi import RAGMultiModalModel
from dotenv import load_dotenv
import base64
import torch
import shutil
import json
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for document storage
UPLOAD_DIR = "uploads"
PDF_DIR = os.path.join(UPLOAD_DIR, "pdfs")
IMAGE_DIR = os.path.join(UPLOAD_DIR, "images")
INDEX_DIR = os.path.join(UPLOAD_DIR, "index")
METADATA_FILE = os.path.join(UPLOAD_DIR, "metadata.json")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Model Enums
class RetrievalModel(str, Enum):
    MULTIMODAL = "multimodal"
    HYBRID = "hybrid"

class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize the multimodal RAG model
        self.multimodal_rag = RAGMultiModalModel.from_pretrained(
            "vidore/colpali",
            device=self.device
        )
        
        # Load existing metadata
        self.metadata = self.load_metadata()
        
        # Initialize index if exists
        if os.path.exists(INDEX_DIR):
            self.load_index()
        
    def load_metadata(self) -> Dict:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        return {"pdfs": {}, "images": {}}
    
    def save_metadata(self):
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f)
    
    def load_index(self):
        try:
            self.multimodal_rag.load_index(INDEX_DIR)
        except Exception as e:
            print(f"Error loading index: {e}")
    
    def save_index(self):
        self.multimodal_rag.save_index(INDEX_DIR)

    async def build_index_from_pdf(self, pdf_path: str, pdf_name: str):
        """Extract images and text from PDF and build/update the index."""
        doc = fitz.open(pdf_path)
        images_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text
            text = page.get_text()
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save image
                image_filename = f"{pdf_name}_page_{page_num + 1}_img_{img_index + 1}.png"
                image_path = os.path.join(IMAGE_DIR, image_filename)
                
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)
                
                # Convert image bytes to base64
                image_base64 = base64.b64encode(image_bytes).decode()
                
                # Store image data with associated text
                images_data.append({
                    "image": image_base64,
                    "text": text,
                    "page": page_num + 1,
                    "source": pdf_name
                })
        
        doc.close()
        
        # Update the index with new images and text
        try:
            self.multimodal_rag.add_documents(images_data)
            self.save_index()
            
            # Update metadata
            self.metadata["pdfs"][pdf_name] = {
                "pages": len(doc),
                "images": len(images_data)
            }
            self.save_metadata()
            
            return len(images_data)
        except Exception as e:
            raise Exception(f"Error building index: {e}")

    def get_retriever(self, model_type: RetrievalModel):
        if model_type == RetrievalModel.MULTIMODAL:
            return self.multimodal_rag.get_retriever()
        else:  # HYBRID
            return self.multimodal_rag.get_hybrid_retriever()

    async def process_query(self, image: Image.Image, question: str, retrieval_model: RetrievalModel):
        # Convert image to format expected by the model
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_data = base64.b64encode(buffered.getvalue()).decode()

        # Get the appropriate retriever
        retriever = self.get_retriever(retrieval_model)

        try:
            # Process the query using the multimodal RAG model
            response = self.multimodal_rag.generate(
                query=question,
                image=image_data,
                retriever=retriever
            )
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model processing error: {str(e)}")

# Initialize model manager
model_manager = ModelManager()

# Add this after app initialization
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save PDF file
        pdf_path = os.path.join(PDF_DIR, file.filename)
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Build/update index from PDF
        num_images = await model_manager.build_index_from_pdf(pdf_path, file.filename)
        
        return JSONResponse(content={
            "message": "PDF processed and indexed successfully",
            "pdf_name": file.filename,
            "extracted_images": num_images
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/index-status")
async def get_index_status():
    try:
        return JSONResponse(content=model_manager.metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_query(
    file: UploadFile = File(...),
    question: str = Form(...),
    retrieval_model: RetrievalModel = Form(...)
):
    try:
        # Process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process query using the model manager
        response = await model_manager.process_query(
            image=image,
            question=question,
            retrieval_model=retrieval_model
        )
        
        return JSONResponse(content={"response": response})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "retrieval_models": [model.value for model in RetrievalModel]
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)