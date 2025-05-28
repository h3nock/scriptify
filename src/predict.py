import os
from typing import Union
import torch
import argparse
import numpy as np
from pathlib import Path
from src.models.rnn import HandwritingRNN
from src.data.dataloader import ProcessedHandwritingDataset
from src.utils.paths import RunPaths, find_latest_run_checkpoint
from src.utils.stroke_viz import plot_offset_strokes 
from config.config import load_config 

    
def load_model_for_prediction(checkpoint_path: Union[str, Path],config, device):
    """Loads a trained model from a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_params  = config.model_params 
    model = HandwritingRNN(
        lstm_size=model_params.lstm_size,
        output_mixture_components=model_params.output_mixture_components, 
        attention_mixture_components=model_params.attention_mixture_components,
        alphabet_size=ProcessedHandwritingDataset.get_alphabet_size()
    )
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v 
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model

def encode_text(text, alphabet, device):
    """Encodes text into a tensor of character indices."""
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    idxs = [char_to_idx.get(c, 0) for c in text] + [0]      
    c   = torch.tensor([idxs], dtype=torch.long, device=device)  
    c_len  = torch.tensor([len(idxs)], dtype=torch.long, device=device)
    return c, c_len 

def predict_handwriting(model, text_to_generate, alphabet, device, max_length=1200, bias=0.75):
    """Generates handwriting for the given text."""
    model.eval()
    c, c_len = encode_text(text_to_generate, alphabet, device)
    
    with torch.no_grad():
        strokes = model.sample(c, c_len, max_length=max_length, bias=bias)
    if strokes:
        return strokes[0].cpu().numpy()  
    print("Warrning: model.sample returned no strokes") 
    return np.array([]) 

def validate_bias(value: float):
    fvalue = float(value)
    if fvalue < 0.5:
        raise argparse.ArgumentTypeError(f"Bias must be >= 0.5, got {fvalue}")
    return fvalue 

def main():
    parser = argparse.ArgumentParser(description="Generate handwriting from text using a trained model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint file (eg., .../checkpoints/model-XXXX).")
    parser.add_argument("--text", type=str, required=True, help="Text to generate handwriting for.")
    parser.add_argument("--max_length", type=int, default=1200, help="Maximum length of the generated stroke sequence.")
    parser.add_argument("--bias", type=float, default=1, help="Sampling bias (temperature). Lower values make it more deterministic.")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save the generated strokes as a .npy file.")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available.")
    
    args = parser.parse_args()

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for prediction.")
    else:
        device = torch.device("cpu")
        print("Using CPU for prediction.")

    alphabet = ProcessedHandwritingDataset.get_alphabet()

    try:
        checkpoint_path = None
        if args.checkpoint is None:
            checkpoint_path = find_latest_run_checkpoint()
        else:
            checkpoint_path = Path(args.checkpoint) 
        
        if checkpoint_path: 
            run_config = None 
            run_paths = None 
            
            try:
                run_name = checkpoint_path.parent.parent.name 
                base_outputs_dir = checkpoint_path.parent.parent.parent
                run_paths = RunPaths(run_name=run_name, base_outputs_dir=base_outputs_dir)
                if run_paths.run_dir.exists() and run_paths.config_copy.exists():
                    run_config = load_config(run_paths.config_copy) 
                else:
                    run_config = load_config() 
                    if not run_paths.run_dir.exists():
                        run_paths = None 
            except IndexError:
                run_config = load_config()
                run_paths = None 
            
            if run_paths is None:
                run_config = load_config()
                run_paths = RunPaths(run_name=f"predict_{checkpoint_path.stem}", base_outputs_dir=run_config.paths.outputs_dir)
                run_paths.create_directories()
            
            model = load_model_for_prediction(checkpoint_path,run_config, device)
            generated_strokes = predict_handwriting(
                model, 
                args.text, 
                alphabet, 
                device, 
                max_length=args.max_length, 
                bias=args.bias
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