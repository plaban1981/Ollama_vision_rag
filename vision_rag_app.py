from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from typing import Optional
import io
from PIL import Image
from enum import Enum
from byaldi import RAGMultiModalModel
from dotenv import load_dotenv
import base64
import torch

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure templates and static files
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Enums
class RetrievalModel(str, Enum):
    MULTIMODAL = "multimodal"
    HYBRID = "hybrid"

# Initialize RAG Model
class ModelManager:
    def __init__(self):
        # Initialize the multimodal RAG model
        self.multimodal_rag = RAGMultiModalModel.from_pretrained(
            "vidore/colpali",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Set up retriever options
        self.retrievers = {
            RetrievalModel.MULTIMODAL: self.multimodal_rag.get_retriever(),
            RetrievalModel.HYBRID: self.multimodal_rag.get_hybrid_retriever()
        }

    def get_retriever(self, model_type: RetrievalModel):
        return self.retrievers[model_type]

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

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "retrieval_models": [model.value for model in RetrievalModel]
        }
    )

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 