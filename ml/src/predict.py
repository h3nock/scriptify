from csv import Error
import os
from typing import Dict, Optional, Union
import torch
import argparse
import numpy as np
from pathlib import Path
from src.utils.text_utils import construct_alphabet_list, encode_text, get_alphabet_map, load_np_strokes, load_priming_data, load_text
from src.models.rnn import HandwritingRNN, PrimingData
from src.utils.paths import RunPaths, find_latest_checkpoint, find_latest_run_checkpoint, find_latest_run_dir
from src.utils.stroke_viz import plot_offset_strokes 
from config.config import load_config 


def predict_handwriting(model: HandwritingRNN, text_to_generate: str, char_map: Dict[str, int], device, max_text_length: int, max_stroke_length=1200, bias=0.75, prime: Optional[int] = None):
    """Generates handwriting for the given text."""
    model.eval() 
    
    encoded_np_array, actual_text_length = encode_text(text=text_to_generate,
                                   char_to_index_map=char_map, 
                                   max_length=max_text_length,
                                   )
    c = torch.tensor(np.array([encoded_np_array]), dtype=torch.long, device=device) 
    c_len = torch.tensor([actual_text_length], dtype=torch.long, device=device)

    
    primingData = None 
    
    if prime is not None:
   
        priming_text, priming_strokes = load_priming_data(style=prime) 
        
        priming_stroke_tensor = torch.tensor(priming_strokes, dtype=torch.float32, device=device).unsqueeze(dim=0)
        encoded_priming_text, priming_text_len = encode_text(priming_text, char_map, max_length=len(priming_text), add_eos=False)
        encoded_priming_text_tensor = torch.tensor(encoded_priming_text, dtype=torch.long, device=device).unsqueeze(dim=0)
        priming_text_len_tensor = torch.tensor([priming_text_len], dtype=torch.long, device=device)
        primingData = PrimingData(priming_stroke_tensor, char_seq_tensors=encoded_priming_text_tensor, char_seq_lengths=priming_text_len_tensor)

    with torch.no_grad():
        # strokes: (batch_size, max_generated_length, 3)
        strokes = model.sample(c, c_len, max_length=max_stroke_length, bias=bias, prime=primingData)

    if strokes:
        # just take the first sample (since we aren't doing batch prediction here)
        # take the first sample and convert it to numpy  
        strokes = strokes[0].cpu().numpy() 
        return strokes 
    print("Warrning: model.sample returned no strokes") 
    return np.array([]) 

def validate_bias(value: float):
    fvalue = float(value)
    if fvalue < 0.5:
        raise argparse.ArgumentTypeError(f"Bias must be >= 0.5, got {fvalue}")
    return fvalue 

def main():
    parser = argparse.ArgumentParser(description="Generate handwriting from text using a trained model.")
    parser.add_argument("--run", type=str, default=None, help="Path to the specific run folder pointing to the run to use for the prediction. The run folder should have `checkpoints/` and `config/` folder. By default the latest run folder is used.")
    parser.add_argument("--checkpoint", type=int, default=None, help="The specific checkpoint (model) number to use from the `checkpoints/` folder. By default the latest model is used.")
    parser.add_argument("--text", type=str, required=True, help="Text to generate handwriting for.")
    parser.add_argument("--max_length", type=int, default=1200, help="Maximum length of the generated stroke sequence.")
    parser.add_argument("--bias", type=float, default=1, help="Sampling bias (temperature). Lower values make it more deterministic.")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save the generated strokes as cha .npy file.")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available.")
    parser.add_argument("--style", type=int, default=None, help="Optional style choice index from list of styles provided") 
    args = parser.parse_args()

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for prediction.")
    else:
        device = torch.device("cpu")
        print("Using CPU for prediction.")

    try:
        run_path = None 
        if args.run is None:
            run_path = find_latest_run_dir() 
        else:
            run_path = Path(args.run) 
        
        if run_path is None:
            raise FileNotFoundError(f"Run directory is not found at {run_path}")
        elif not run_path.is_dir():
            raise NotADirectoryError(f"Run path is not directory: {run_path}") 
        
        checkpoints_subdir = run_path / "checkpoints"
        config_subdir = run_path / "config"

        if not checkpoints_subdir.is_dir():
            raise FileNotFoundError(f"Requreid 'checkpoints' subdirectory not find in run directory:{run_path} or not a directory. ")

        if not config_subdir.is_dir():
            raise FileNotFoundError(f"Requreid 'config' subdirectory not find in run directory:{run_path} or not a directory. ")

        checkpoint_path = None 
        if args.checkpoint is None:
            checkpoint_path = find_latest_checkpoint(checkpoints_subdir ) 
        else:
            checkpoint = checkpoints_subdir / f"model-{args.checkpoint}" 
        
        if checkpoint_path is None or (not checkpoint_path.exists()):
            raise FileNotFoundError(f"Checkpoint is not found at {checkpoint_path}")
        elif checkpoint_path.is_dir(): 
            raise FileNotFoundError(f"Checkpoint file expected but provided directory path: {checkpoint_path}")
        
        try: 
            run_name = run_path.name 
            base_outputs_dir = run_path.parent 
            run_paths = RunPaths(run_name=run_name, base_outputs_dir=base_outputs_dir)

            if not run_paths.config_copy.exists(): 
                raise FileNotFoundError(f"Config file not found at {run_paths.config_copy}")
            run_config = load_config(config_path=run_paths.config_copy) 
            
        except Exception as e:
            raise Error(f"Unexpected error occured while preparing checkpoint and config files: {e}") 
        
        max_text_len_for_encoding = run_config.dataset.max_text_len 
        model, char_map = HandwritingRNN.load_from_checkpoint(checkpoint_path,run_config, device)

        generated_strokes = predict_handwriting(
            model, 
            args.text, 
            char_map, 
            device, 
            max_text_length=max_text_len_for_encoding,
            max_stroke_length=args.max_length, 
            bias=args.bias, 
            prime=args.style
        )
        
        if generated_strokes.size > 0:
            print(f"Generated Strokes Shape: {generated_strokes.shape}")
            plot_offset_strokes(generated_strokes,run_paths.get_sample_plot_path(text=args.text), plot_only_text=True)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()