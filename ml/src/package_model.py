from typing import Optional
import torch 
import argparse 
import traceback
from pathlib import Path 
from config.config import load_config, PROJECT_ROOT
from src.data.dataloader import ProcessedHandwritingDataset
from src.models.rnn import HandwritingRNN
from src.utils.paths import RunPaths, find_latest_checkpoint

def package_model(checkpoint_path: Path, 
                  output_base_name: str,
                  output_dir: Path,
                  config_path: Optional[Path] = None):
    """package trained model by saving its state_dict, model params and TorchScript traced version."""
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found at {checkpoint_path}")
    
    if not config_path or not config_path.exists():
        raise FileNotFoundError(f"config file not found at {config_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_config = load_config(config_path) 
    print(f"Loaded specified config from {config_path}")

    model_params = run_config.model_params 
    alphabet_size = ProcessedHandwritingDataset.get_alphabet_size() 
    
    model = HandwritingRNN(
        lstm_size=model_params.lstm_size, 
        output_mixture_components=model_params.output_mixture_components, 
        attention_mixture_components=model_params.attention_mixture_components,
        alphabet_size=alphabet_size, 
        dropout_prob=model_params.dropout_prob
    )

    # load the checkpoint 
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint_data['model_state_dict']
    new_state_dict = {} 
    is_ddp_model = any(k.startswith('module.') for k in state_dict.keys())
    for k, v in state_dict.items():
        if is_ddp_model and k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval() 

    output_dir.mkdir(parents=True, exist_ok=True) 
    original_pkg_path = output_dir / f"{output_base_name}.pt" 
    pkg_data = {
        'model_params': model_params.model_dump(), 
        'model_state_dict': model.state_dict(), 
        'alphabet_size': alphabet_size, 
        "config_full": run_config.model_dump()
    }
    
    torch.save(pkg_data, original_pkg_path)
    print(f"Original model package saved to: {original_pkg_path}")

    
    max_char_len_for_tracing = run_config.dataset.max_text_len

    example_batch_size = 1
    example_x_seq_len = 1
    
    example_x = torch.zeros(example_batch_size, example_x_seq_len, 3, device=device)
    example_x[:, 0, 2] = 1.0

    example_c = torch.randint(0, alphabet_size, (example_batch_size, max_char_len_for_tracing), device=device, dtype=torch.long)
    example_c_len = torch.tensor([max_char_len_for_tracing] * example_batch_size, device=device, dtype=torch.long)

    print(f"Tracing model with example inputs: x_shape={example_x.shape}, c_shape={example_c.shape}, c_len_shape={example_c_len.shape}")

    try:
        traced_model = torch.jit.trace(model, (example_x, example_c, example_c_len))
        traced_model_path = output_dir / f"{output_base_name}.traced.pt"
        torch.jit.save(traced_model, traced_model_path)
        print(f"TorchScript traced model saved to: {traced_model_path}")
    except Exception as e:
        print(f"Error during TorchScript tracing or saving: {e}")
        traceback.print_exc()
        print("TorchScript model was NOT saved due to an error.")
    
def main():
    parser = argparse.ArgumentParser(description="Package a trained handwriting model")
     
    parser.add_argument(
        "--checkpoint",
        type=str, 
        default=None, 
        help="Path to the model checkpoint file" 
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="packaged_models",
        help="Directory to save the packaged model files" 
    )

    parser.add_argument(
        "--pkg_name", 
        type=str, 
        default="handwriting_model", 
        help="Base name for the packaged model files" 
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="Path to the specific config.yaml file for the model" 
    )
    args = parser.parse_args() 
    checkpoint_path = None 
    config_file_path = args.config 

    if args.checkpoint: 
        checkpoint_path = Path(args.checkpoint) 
        if config_file_path: 
            config_file_path = Path(config_file_path) 
        
        if not config_file_path or not config_file_path.exists():
            print(f"Provided {config_file_path} as a config file path. Please provide a valid one")
            
    else:
        print("No checkpoint specified, trying to find the latest...")
        try: 
            default_config = load_config()
            outputs_dir = default_config.paths.outputs_dir
            print(f"Searching checkpoints inside {outputs_dir}")
        except Exception as e:
           outputs_dir = PROJECT_ROOT / "outputs"  
           print(f"Falling back to searching inside {outputs_dir}")

        latest_run = RunPaths.find_latest_run(base_outputs_dir=outputs_dir) 
        if latest_run:
            checkpoint_path = find_latest_checkpoint(latest_run.checkpoints_dir)
            config_file_path = latest_run.config_copy 
        else:
            print(f"Coun't find checkpoint. Please specify one")            
    
    if not checkpoint_path or not checkpoint_path.exists():
        print("Checkpoint path `{checkpoint_path}` doesn't exist.")
        return 
    if not config_file_path or not config_file_path.exists():
        print(f"Config file path `{config_file_path}` doesn't exist.")
        return 
    

    pkg_output_dir = PROJECT_ROOT / args.output_dir 

    try:
        package_model(checkpoint_path, args.pkg_name, pkg_output_dir, config_file_path)
    except Exception as e:
        print(f"An error occurred pakcing the model: {e}")
        traceback.print_exc()
    
    

if __name__ == '__main__':
    main()