import time 
import torch 
import wandb 
import psutil 
import pynvml 
import argparse  
import logging 
import threading
from pathlib import Path 
from typing import Optional
from src.utils.text_utils import construct_alphabet_list, encode_text, get_alphabet_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

SCRIPTED_MODEL_NAME = "model.scripted.pt" 
METADATA_MODEL_NAME = "model.pt" 

scripted_model: Optional[torch.jit.ScriptModule] = None 
model_metadata: Optional[dict] = None 
device: Optional[torch.device] = None 
alphabet_map: Optional[dict[str, int]] = None 
ALPHABET_LIST: Optional[list[str]] = None 
ALPHABET_SIZE: Optional[int] = None 
max_text_len: Optional[int] = None 

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

    char_seq = torch.from_numpy(padded_encoded_np).unsqueeze(0).to(device=device, dtype=torch.long)
    char_len = torch.tensor([true_length], device=device, dtype=torch.long)

    return char_seq, char_len


def init_model(model_dir: Path):

    global scripted_model, model_metadata, alphabet_map, max_text_len, ALPHABET_LIST, ALPHABET_SIZE
    logger.info("Attempting to load model resources during startup") 
    try:
        
        scripted_model_path = model_dir / SCRIPTED_MODEL_NAME
        metadata_model_path = model_dir / METADATA_MODEL_NAME
        
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
     
def sample(  char_seq: torch.Tensor,
    char_lengths: torch.Tensor,
    max_gen_len: int,
    bias: float,): 
    global scripted_model
    if scripted_model is None:
        raise ValueError("Scripted model not initialized.")
    
    with torch.inference_mode():
        try:
            stroke_tensors = scripted_model.sample(
                char_seq, 
                char_lengths, 
                max_length=max_gen_len, 
                bias=bias
            )
            return stroke_tensors
        
        except Exception as e:
            logger.error(f"Error in model sampling: {e}", exc_info=True)
            return []
     
def gpu_stats():
    if not torch.cuda.is_available():
        return 0.0, 0.0 

    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu 
    return (float) (mem.used) / 1e6, util # gpu memory in MB, gpu used in % 

def peak_sampler(stop_event: threading.Event, peak: dict):
    
    while not stop_event.is_set():
        peak["cpu"] = max(peak['cpu'],psutil.cpu_percent(None))
        peak["gpu"] = max(peak['gpu'], gpu_stats()[1])
        time.sleep(0.01) # 10ms delay 


def benchmark(text: str, run_name: str, runs: int, chars: torch.Tensor, chars_len: torch.Tensor, 
              max_stroke_len: int = 1000, bias: float = 2):
    latencies = []
    cpu_peaks = []
    gpu_peaks = []
    gpu_mem_peaks = []

    for run in range(runs):
        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()
        peak = {"cpu": 0, "gpu": 0}
        stop_event = threading.Event()
        sampler = threading.Thread(target=peak_sampler, args=(stop_event, peak))
        sampler.start() 

        time_start = time.perf_counter() 
        sample(chars,char_lengths=chars_len, max_gen_len=max_stroke_len, bias=bias)
        
        time_end = time.perf_counter() 
        
        curr_latency = (time_end - time_start) * 1e3 # ms 
        
        stop_event.set() 
        sampler.join() 
        
        latencies.append(curr_latency)
        cpu_peaks.append(peak['cpu'])
        gpu_peaks.append(peak['gpu'])
        if torch.cuda.is_available():
            gpu_mem_peaks.append(torch.cuda.max_memory_allocated(device) / 1e6) # MB
        else:
            gpu_mem_peaks.append(0)

    latencies_tensor = torch.tensor(latencies) 
    summary = {
        "model_run_used": run_name,
        "text_used:": text,  
        "p50_ms": float(latencies_tensor.median()), 
        "p95_ms": float(torch.quantile(latencies_tensor, 0.95)), 
        "cpu_peak_%": float(max(cpu_peaks)), 
        "gpu_peak_%": float(max(gpu_peaks)), 
        "gpu_mem_peak_MB": float(max(gpu_mem_peaks))
    }

    # wandb log 
    wandb.log(summary) 

def main():
    global device 

    parser = argparse.ArgumentParser(description="Benchmark inference")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the run directory contianing packaged model metadata (`model.pt`) and the scripted model (`model.scripted.pt`). By default, each run's packages are stored separetly inside packaged_models")
    parser.add_argument("--text", type=str, default="Hello World", help="Text to generate to text the inference with.")
    parser.add_argument("--num_runs", type=int, default=100, help="The number of runs to per setting")
    parser.add_argument("--max_stroke_len", type=int, default=1200, help="Maximum length of the generated stroke sequence.")
    parser.add_argument("--bias", type=float, default=2, help="Sampling bias (temperature). Lower values make it more deterministic.")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available.")
    
    args = parser.parse_args()
    torch.manual_seed(seed=42)
    wandb.init(project="scriptify",reinit=True)
    use_cuda: bool = not args.no_cuda and torch.cuda.is_available() 

    if use_cuda:
        device = torch.device("cuda")
        pynvml.nvmlInit()
        print("Using GPU for prediction.")
    else:
        device = torch.device("cpu")
        print("Using CPU for prediction.")

    model_path =  Path(args.model_path)
    if model_path is None or not model_path.exists():
        raise ValueError("Valid model_path arguemnt is required")
    if not model_path.is_dir():
        raise ValueError("model_path should point to a directory contianing `model.pt` and `model.scripted.pt`")
        

    init_model (model_dir=model_path)

    char_seq_tensor, char_lengths_tensor = text_to_tensor(args.text, device)

    # a warmp up turn 
    sample(char_seq_tensor, char_lengths=char_lengths_tensor, max_gen_len=1200,bias=2)
    run_name = model_path.name  
    benchmark(args.text, run_name, args.num_runs,chars=char_seq_tensor, chars_len=char_lengths_tensor, 
              max_stroke_len=args.max_stroke_len, bias=args.bias)

    wandb.finish() 

    pynvml.nvmlShutdown() 


if __name__ == '__main__':
    main() 

    
