import torch
import numpy as np
import argparse
import os

from src.models.handwriting_rnn import HandwritingRNN
from src.data.dataloader import ProcessedHandwritingDataset
from src.utils.stroke_viz import plot_offset_strokes # Assuming you have this utility

def load_model_for_prediction(checkpoint_path, device):
    """Loads a trained model from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = HandwritingRNN(
        lstm_size=400,
        output_mixture_components=20, 
        attention_mixture_components=10,
        alphabet_size=ProcessedHandwritingDataset.get_alphabet_size()
    )
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove `module.`
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

def predict_handwriting(model, text_to_generate, alphabet, device, max_length=1000, bias=0.75):
    """Generates handwriting for the given text."""
    model.eval()
    c, c_len = encode_text(text_to_generate, alphabet, device)
    
    with torch.no_grad():
        strokes = model.sample(c, c_len, max_length=max_length, bias=bias)
    if strokes:
        return strokes[0].cpu().numpy()  
    print("Warrning: model.sample returned no strokes") 
    return np.array([]) 

def main():
    parser = argparse.ArgumentParser(description="Generate handwriting from text using a trained model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file (eg., checkpoints/model-XXXX).")
    parser.add_argument("--text", type=str, required=True, help="Text to generate handwriting for.")
    parser.add_argument("--max_length", type=int, default=1000, help="Maximum length of the generated stroke sequence.")
    parser.add_argument("--bias", type=float, default=0.5, help="Sampling bias (temperature). Lower values make it more deterministic.")
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
        model = load_model_for_prediction(args.checkpoint, device)
        
        print(f"\nGenerating handwriting for: '{args.text}'")
        generated_strokes = predict_handwriting(
            model, 
            args.text, 
            alphabet, 
            device, 
            max_length=args.max_length, 
            bias=args.bias
        )
        
        print(f"Generated strokes shape: {generated_strokes.shape}")

        plot_offset_strokes(generated_strokes, title=f"Generated: '{args.text}'")
        
        if args.output_file:
            np.save(args.output_file, generated_strokes)
            print(f"Generated strokes saved to {args.output_file}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()