from typing import Optional
from fastapi import FastAPI, HTTPException, status 
from pydantic import BaseModel, Field
import torch 
import torch.nn.functional as F 
from pathlib import Path 
import logging 
import time 
from contextlib import asynccontextmanager
from inference_utils import construct_alphabet_list, convert_offsets_to_absolute_coords, encode_text, get_alphabet_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

MODEL_DIR = Path("../../ml/packaged_models") 
SCRIPTED_MODEL_NAME = "handwriting_model.scripted.pt" 
METADATA_MODEL_NAME = "handwriting_model.pt" 

scripted_model: Optional[torch.jit.ScriptModule] = None 
model_metadata: Optional[dict] = None 
device: Optional[torch.device] = None 
alphabet_map: Optional[dict[str, int]] = None 
ALPHABET_LIST: Optional[list[str]] = None 
ALPHABET_SIZE: Optional[int] = None 
max_text_len: Optional[int] = None 
output_mixture_components: Optional[int] = None # To store num_mixtures for GMM sampling
lstm_size: Optional[int] = None 
attention_mixture_components: Optional[int] = None 

# Patience for early stopping in generate_strokes
PATIENCE_PEN_UP_EOS = 15
MIN_MOVEMENT_THRESHOLD = 0.02 


class HandwritingRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=40, description="Text to generate handwriting for")
    max_length: int = Field(default=700, ge=50, le=1500, description="Maximum number of stroke points")
    bias: float = Field(default=0.75, ge=0.1, le=2.0, description="Sampling bias for generation")
class HandwritingResponse(BaseModel):
    sucess: bool = True
    input_text: str
    generation_time_ms: float
    num_points: int
    strokes: list[list[float]]
    message: str = "Successfully generated handwriting."

class HealthResponse(BaseModel):
    status: str 
    model_loaded: bool 
    device: str 
    model_metadata_keys: Optional[list[str]] = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global scripted_model, model_metadata, device, alphabet_map, max_text_len, ALPHABET_LIST, output_mixture_components, lstm_size, attention_mixture_components, ALPHABET_SIZE
    logger.info("Attempting to load model resources during startup") 
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        scripted_model_path = MODEL_DIR / SCRIPTED_MODEL_NAME
        metadata_model_path = MODEL_DIR / METADATA_MODEL_NAME
        
        if  not scripted_model_path.exists():
            logger.error(f"Traced model not found at {scripted_model_path}")
            raise FileNotFoundError(f"Traced model not found at {scripted_model_path}")
        if not metadata_model_path or not metadata_model_path.exists():
            logger.error(f"Metadata model file not found at {metadata_model_path}")
            raise FileNotFoundError(f"Metadata model file not found at {metadata_model_path}")

        # Load the traced model
        scripted_model = torch.jit.load(scripted_model_path, map_location=device)
        if scripted_model:
            scripted_model.eval()
            logger.info(f"Traced model loaded successfully from {scripted_model_path}")

        # Load the metadata
        model_metadata = torch.load(metadata_model_path, map_location='cpu')
        if model_metadata:
            logger.info(f"Model metadata loaded successfully from {metadata_model_path}")
            logger.info(f"Model metadata keys: {list(model_metadata.keys())}")
            
            config_full = model_metadata['config_full']
            if not config_full or not isinstance(config_full, dict):
                raise ValueError(f"Key `config_full` not found or not a dict")
            
            dataset_config = config_full['dataset']
            model_params = config_full['model_params']

            if not dataset_config or not isinstance(dataset_config, dict):
                raise ValueError(f"Key `dataset` not found or not a dict in config_full")
            alphabet_str = dataset_config['alphabet_string']
            max_text_len = dataset_config['max_text_len']
            output_mixture_components = model_params['output_mixture_components']

            lstm_size = model_params['lstm_size']
            attention_mixture_components = model_params['attention_mixture_components']

            ALPHABET_LIST = construct_alphabet_list(alphabet_str)
            ALPHABET_SIZE = len(ALPHABET_LIST)
            alphabet_map = get_alphabet_map(ALPHABET_LIST) 
            
            logger.info(f"Alphabet created. Size: {len(ALPHABET_LIST)}")
            logger.info("Model resources are loaded and ready")
        else:
            raise ValueError(f"Failed to load content frm metadata file")

    except Exception as e:
        logger.error(f"Error loading model resources: {e}", exc_info=True)
        scripted_model = None
        model_metadata = None
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down API and cleaning up resources")
    scripted_model = None 
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
    global scripted_model, model_metadata, device, alphabet_map, max_text_len, ALPHABET_LIST 
    
    is_healthy = all([scripted_model, model_metadata, device, alphabet_map, max_text_len, ALPHABET_LIST])
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=bool(scripted_model),
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

    char_seq = torch.from_numpy(padded_encoded_np).to(device=device, dtype=torch.long)
    char_len = torch.tensor([true_length], device=device, dtype=torch.long)

    return char_seq, char_len

def generate_strokes(
    char_seq: torch.Tensor,
    char_lengths: torch.Tensor,
    max_gen_len: int,
    api_bias: float,
    current_device: torch.device
) -> list[list[float]]:
    """Generate strokes using the model's built-in sample method"""
    global scripted_model
    if scripted_model is None:
        raise ValueError("Scripted model not initialized.")
    
    with torch.no_grad():
        try:
            stroke_tensors = scripted_model.sample(
                char_seq, 
                char_lengths, 
                max_length=max_gen_len, 
                bias=api_bias
            )
            
            if len(stroke_tensors) == 1 and stroke_tensors[0].dim() == 2:
                all_strokes_tensor = stroke_tensors[0]
                stroke_offsets = all_strokes_tensor.cpu().numpy().tolist()
            else:
                stroke_offsets = []
                for stroke_tensor in stroke_tensors:
                    if stroke_tensor.dim() == 2:
                        stroke_data = stroke_tensor.squeeze(0).cpu().numpy().tolist()
                    else:
                        stroke_data = stroke_tensor.cpu().numpy().tolist()
                    
                    if len(stroke_data) == 3:
                        stroke_offsets.append(stroke_data)
            
            return stroke_offsets
            
        except Exception as e:
            logger.error(f"Error in model sampling: {e}", exc_info=True)
            return []
            
@app.post("/generate", response_model=HandwritingResponse, tags=["Generation"])
async def generate_handwriting_endpoint(request: HandwritingRequest):
    if not all([scripted_model, model_metadata, device, alphabet_map, max_text_len]):
        logger.error("API not fully initialized. Check /health endpoint.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or required resources not loaded."
        )
    
    assert device is not None, "Device is None inside generate_handwriting"
    start_time = time.time()
    
    try:
        char_seq_tensor, char_lengths_tensor = text_to_tensor(request.text, device)

        relative_stroke_offsets = generate_strokes(
            char_seq_tensor, char_lengths_tensor, request.max_length, request.bias, device
        )

        if not relative_stroke_offsets:
            return HandwritingResponse(
                success=False,
                input_text=request.text,
                strokes=[],
                num_points=0,
                generation_time_ms=(time.time() - start_time) * 1000,
                message="No strokes generated."
            )

        absolute_stroke_coords = convert_offsets_to_absolute_coords(relative_stroke_offsets)
        generation_time_ms = (time.time() - start_time) * 1000

        return HandwritingResponse(
            input_text=request.text,
            strokes=absolute_stroke_coords,
            num_points=len(absolute_stroke_coords),
            generation_time_ms=generation_time_ms
        )
    except ValueError as ve:
        logger.error(f"ValueError during generation for '{request.text}': {ve}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error for '{request.text}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for Scriptify API...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir=".")