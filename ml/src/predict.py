import os
from typing import Dict, Optional, Union
import torch
import argparse
import numpy as np
from pathlib import Path
from src.utils.text_utils import construct_alphabet_list, encode_text, get_alphabet_map, load_np_strokes, load_priming_data, load_text
from src.models.rnn import HandwritingRNN, PrimingData
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
    alphabet = construct_alphabet_list(config.dataset.alphabet_string) 
    alphabet_size = len(alphabet) 
    char_map = get_alphabet_map(alphabet_list=alphabet)
    
    model = HandwritingRNN(
        lstm_size=model_params.lstm_size,
        output_mixture_components=model_params.output_mixture_components, 
        attention_mixture_components=model_params.attention_mixture_components,
        alphabet_size=alphabet_size
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
    return model, char_map  

# def encode_text(text, alphabet, device):
#     """Encodes text into a tensor of character indices."""
#     char_to_idx = {c: i for i, c in enumerate(alphabet)}
#     idxs = [char_to_idx.get(c, 0) for c in text] + [0]      
#     c   = torch.tensor([idxs], dtype=torch.long, device=device)  
#     c_len  = torch.tensor([len(idxs)], dtype=torch.long, device=device)
#     return c, c_len 

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
    
    if prime:
   
        priming_text, priming_strokes = load_priming_data(style=prime) 
        
        priming_stroke_tensor = torch.tensor(priming_strokes, dtype=torch.float32, device=device).unsqueeze(dim=0)
        encoded_priming_text, priming_text_len = encode_text(priming_text, char_map, max_length=len(priming_text), add_eos=False)
        encoded_priming_text_tensor = torch.tensor(encoded_priming_text, dtype=torch.long, device=device).unsqueeze(dim=0)
        priming_text_len_tensor = torch.tensor([priming_text_len], dtype=torch.long, device=device)
        primingData = PrimingData(priming_stroke_tensor, char_seq_tensors=encoded_priming_text_tensor, char_seq_lengths=priming_text_len_tensor)

    with torch.no_grad():
        strokes = model.sample(c, c_len, max_length=max_stroke_length, bias=bias, prime=primingData)

    if strokes:
        # just take the first sample (since we aren't doing batch prediction)
        strokes = [stroke.squeeze(0).cpu().numpy() for stroke in strokes]
        return np.array(strokes)
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
            
            max_text_len_for_encoding = run_config.dataset.max_text_len 
            model, char_map = load_model_for_prediction(checkpoint_path,run_config, device)

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