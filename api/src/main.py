from typing import Optional
from fastapi import FastAPI, HTTPException, status 
from pydantic import BaseModel, Field
import torch 
import torch.nn.functional as F 
from pathlib import Path 
import logging 
import time 
from contextlib import asynccontextmanager
from inference_utils import construct_alphabet_list, encode_text, get_alphabet_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

MODEL_DIR = Path("../../ml/packaged_models") 
TRACED_MODEL_NAME = "handwriting_model.traced.pt" 
METADATA_MODEL_NAME = "handwriting_model.pt" 

traced_model: Optional[torch.jit.ScriptModule] = None 
model_metadata: Optional[dict] = None 
device: Optional[torch.device] = None 
alphabet_map: Optional[dict[str, int]] = None 
ALPHABET_LIST: Optional[list[str]] = None 
max_text_len: Optional[int] = None 

class HandwritingRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=40, description="Text to generate handwriting for")
    max_length: int = Field(default=700, ge=50, le=1500, description="Maximum number of stroke points")
    bias: float = Field(default=0.75, ge=0.1, le=2.0, description="Sampling bias for generation")

class StrokePoint(BaseModel):
    x: float 
    y: float 
    pen_state: float # 0 for pen down, 1 for pen up 
    
class HandwritingResponse(BaseModel):
    input_text: str
    generated_strokes: list[StrokePoint]
    num_points: int
    message: str = "Successfully generated handwriting."
    generation_time_ms: float

class HealthResponse(BaseModel):
    status: str 
    model_loaded: bool 
    device: str 
    model_metadata_keys: Optional[list[str]] = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global traced_model, model_metadata, device, alphabet_map, max_text_len, ALPHABET_LIST
    logger.info("Attempting to load model resources during startup") 
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        traced_model_path = MODEL_DIR / TRACED_MODEL_NAME
        metadata_model_path = MODEL_DIR / METADATA_MODEL_NAME
        
        if  not traced_model_path.exists():
            logger.error(f"Traced model not found at {traced_model_path}")
            raise FileNotFoundError(f"Traced model not found at {traced_model_path}")
        if not metadata_model_path or not metadata_model_path.exists():
            logger.error(f"Metadata model file not found at {metadata_model_path}")
            raise FileNotFoundError(f"Metadata model file not found at {metadata_model_path}")

        # Load the traced model
        traced_model = torch.jit.load(traced_model_path, map_location=device)
        if traced_model:
            traced_model.eval()
            logger.info(f"Traced model loaded successfully from {traced_model_path}")

        # Load the metadata
        model_metadata = torch.load(metadata_model_path, map_location='cpu')
        if model_metadata:
            logger.info(f"Model metadata loaded successfully from {metadata_model_path}")
            logger.info(f"Model metadata keys: {list(model_metadata.keys())}")
            
            config_full = model_metadata['config_full']
            if not config_full or not isinstance(config_full, dict):
                raise ValueError(f"Key `config_full` not found or not a dict")
            
            dataset_config = config_full['dataset']

            if not dataset_config or not isinstance(dataset_config, dict):
                raise ValueError(f"Key `dataset` not found or not a dict in config_full")
            alphabet_str = dataset_config['alphabet_string']
            max_text_len = dataset_config['max_text_len']

            ALPHABET_LIST = construct_alphabet_list(alphabet_str)
            alphabet_map = get_alphabet_map(ALPHABET_LIST) 
            
            logger.info(f"Alphabet created. Size: {len(ALPHABET_LIST)}")
            logger.info("Model resources are loaded and ready")
        else:
            raise ValueError(f"Failed to load content frm metadata file")

    except Exception as e:
        logger.error(f"Error loading model resources: {e}", exc_info=True)
        traced_model = None
        model_metadata = None
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down API and cleaning up resources")
    traced_model = None 
    model_metadata = None 

app = FastAPI(
    title="Scriptify API", 
    description="API to generate handwriting from text using a PyTorch model.", 
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the Scriptify Handwriting Generation API!"}

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    global traced_model, model_metadata, device, alphabet_map, max_text_len, ALPHABET_LIST 
    
    is_healthy = all([traced_model, model_metadata, device, alphabet_map, max_text_len, ALPHABET_LIST])
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=bool(traced_model),
        device=str(device) if device else "unknown",
        model_metadata_keys=list(model_metadata.keys()) if model_metadata else None,
    )

def text_to_tensor(text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert text to tensor format expected by the model"""
    global alphabet_map, max_text_len  
    if alphabet_map is None:
        raise ValueError("Alphabet map not initialized during api startup")
    if max_text_len is None:
        raise ValueError("`max_text_len` is not initialized during api startup")
    padded_encoded_np, true_length = encode_text(
        text=text, 
        char_to_index_map=alphabet_map, 
        max_length=max_text_len 
    )

    char_seq = torch.tensor([padded_encoded_np],dtype=torch.long, device=device)
    char_len = torch.tensor([true_length],dtype=torch.long, device=device)

    return char_seq, char_len

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for Scriptify API...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir=".")