import torch
import pandas as pd 
from pathlib import Path
from torch.utils.data import DataLoader 
from config.config import Config, load_config
from src.training.loss import gaussian_mixture_loss
from src.data.dataloader import ProcessedHandwritingDataset
from src.models.rnn import HandwritingRNN
from src.utils.paths import RunPaths, find_latest_checkpoint, find_latest_run_checkpoint, find_latest_run_dir 
from src.data.raw_data_reader import SourceDocumentInfo, collect_and_organize_docs, get_stroke_seqs


def evaluate_losses(config: Config, checkpoint_path: Path) -> dict:

    
    device = torch.device(device="cpu")
    if torch.cuda.is_available():
        device = torch.device(device="cuda")

    model, char_map = HandwritingRNN.load_from_checkpoint(checkpoint_path=checkpoint_path,config= config, device = device)

    model.eval()

    dataset = ProcessedHandwritingDataset(processed_dir=config.paths.processed_data_dir)

    data_loader = DataLoader(dataset, 
                             batch_size=config.training_params.batch_sizes[0] if config.training_params.batch_sizes else 128, 
                             shuffle=False
                            )
    filename_to_loss = {}
    with torch.no_grad():

        for batch_idx, batch_data in enumerate(data_loader):
            strokes_batch = batch_data['stroke'].to(device)
            strokes_len_batch = batch_data['stroke_len'].to(device) 
            chars_batch = batch_data['chars'].to(device)
            chars_len_batch = batch_data['chars_len'].to(device)
            stroke_filenames_batch = batch_data['stroke_filename'] 
            
            y_target_batch = strokes_batch[:, 1:, :]
            x_input_batch = strokes_batch[:, :-1, :]
            
            y_lengths_batch = strokes_len_batch - 1
            y_lengths_batch = torch.clamp(y_lengths_batch, min=0)
            
            valid_indices = y_lengths_batch > 0
            
            if not valid_indices.any(): 
                print(f"Skipping batch {batch_idx} as all samples have zero target length after shift.")
                continue

            x_input_batch_valid = x_input_batch[valid_indices]
            y_target_batch_valid = y_target_batch[valid_indices]
            chars_batch_valid = chars_batch[valid_indices]
            chars_len_batch_valid = chars_len_batch[valid_indices]
            y_lengths_batch_valid = y_lengths_batch[valid_indices]
            stroke_filenames_batch_valid = [stroke_filenames_batch[i] for i, v in enumerate(valid_indices) if v]


            if x_input_batch_valid.size(0) == 0: 
                continue

            # forward pass
            pis, sigmas, rhos, mus, es, _ = model(x_input_batch_valid, chars_batch_valid, chars_len_batch_valid)
            
            # sequence_loss with dim = (batch_size) where each element is the avg loss for that sequence
            sequence_losses_for_batch, _ = gaussian_mixture_loss(
                y_target_batch_valid, y_lengths_batch_valid, pis, sigmas, rhos, mus, es
            )
            
            for i, filename in enumerate(stroke_filenames_batch_valid):
                if filename in filename_to_loss:
                    # this shouldn't occur if shuffle is false since filenames are unique 
                    print(f"Duplicate filename '{filename}' encountered. Overwriting loss.")
                filename_to_loss[filename] = sequence_losses_for_batch[i].item()
            
            if (batch_idx + 1) % 20 == 0:
                 print(f"Processed batch {batch_idx + 1}/{len(data_loader)}")

    return filename_to_loss 
            
    
def main():
    run_path = find_latest_run_dir() 
    
    checkpoints_subdir = run_path / "checkpoints"

    checkpoint_path = find_latest_checkpoint(checkpoints_dir=checkpoints_subdir)
    config_path = run_path / "config" / "config.yaml" 
    config = load_config(config_path=config_path)
    
    filename_to_loss = evaluate_losses(
        config=config, 
        checkpoint_path=checkpoint_path 
    ) 
    if filename_to_loss:
        output_df = pd.DataFrame(list(filename_to_loss.items()), columns=['stroke_filename', 'average_sequence_loss'])
    
   
    output_dir = config.paths.processed_data_dir 
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_csv_path = output_dir / "stroke_file_losses.csv"
    output_df.to_csv(output_csv_path, index=False)
    print(f"Saved stroke file losses to: {output_csv_path}")
        

if __name__ == "__main__":
    main()