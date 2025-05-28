import os
from typing import Optional
import torch
import wandb 
from datetime import datetime 
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
import torch.distributed as dist
from src.models.rnn import HandwritingRNN
from src.training.trainer import HandwritingTrainer
from src.data.dataloader import ProcessedHandwritingDataset
from config.config import load_config 
from src.utils.paths import RunPaths

config_global = load_config()

def setup(rank, world_size):
    """Initialize distributed training process group"""
    #TODO: handle distributed training config params later 
    dist_config = config_global.distributed_training
    os.environ['MASTER_ADDR'] = dist_config.master_addr
    os.environ['MASTER_PORT'] = str(dist_config.master_port)
    dist.init_process_group(dist_config.backend, rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training process group"""
    dist.destroy_process_group()

def train(rank, world_size):
    """Training function to run on each GPU"""
    # setup process group
    setup(rank, world_size)
    
    # set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # load dataset
    full_dataset = ProcessedHandwritingDataset(config_global.paths.processed_data_dir)
    
    # split dataset 
    generator = torch.Generator().manual_seed(config_global.dataset.random_seed)
    dataset_size = len(full_dataset)
    train_size = int(config_global.dataset.train_split_ration * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    run_paths = RunPaths(base_outputs_dir=config_global.paths.outputs_dir)
    wandb_run = None 
    if rank == 0: 
        run_paths.create_directories() 
        run_paths.copy_config_file(config_global.paths.config_file_path)
        
        if config_global.wandb.enabled:
            hyperparams = {
                **config_global.dataset.model_dump(),
                **config_global.model_params.model_dump(),
                **config_global.training_params.model_dump(),
                **config_global.distributed_training.model_dump(), 
                "world_size": world_size,
                "device_type": device.type,
            }
                
            wandb_run = wandb.init(
                project=config_global.wandb.project_name,
                name= run_paths.run_name, 
                config=hyperparams, 
                tags=config_global.wandb.tags, 
                notes=config_global.wandb.notes,
                resume="allow"
            )
            
        print(f"Dataset loaded: {dataset_size} samples")
        print(f"Training set: {train_size} samples")
        print(f"Validation set: {val_size} samples")
    
    # init model
    model = HandwritingRNN(
        lstm_size=config_global.model_params.lstm_size,
        output_mixture_components=config_global.model_params.output_mixture_components,
        attention_mixture_components= config_global.model_params.attention_mixture_components,
        alphabet_size= ProcessedHandwritingDataset.get_alphabet_size() 
    )
    
    model.to(device)
    if rank == 0 and wandb_run and config_global.wandb.watch_model_log_freq > 0:
        wandb.watch(model, log="gradients", log_freq=config_global.wandb.watch_model_log_freq)
    ddp_model = DDP(model, device_ids=[rank])
    
    # init trainer 
    trainer = HandwritingTrainer(
        model=ddp_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_params=config_global.training_params, 
        config_file_path=config_global.paths.config_file_path,
        run_paths=run_paths, 
        wandb_config=config_global.wandb,
        device=device,
        world_size=world_size,
        rank=rank
    )
    
    # start training
    if rank == 0:
        print("Starting training...")
    best_step = trainer.fit()

    if rank == 0:
        print(f"Training completed. Best model at step {best_step}")
        if wandb_run:
            wandb_run.finish() 
    
    # clean up
    cleanup()

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        return
    try:
        # set the multiprocessing start method
        torch.multiprocessing.set_start_method('spawn', force=True)
        
        # get the number of available GPUs
        world_size = torch.cuda.device_count()
        print(f"Found {world_size} GPUs")
        if world_size == 1:
            print("Only one GPU found, running without DDP")
            train(0, 1)
        else:
            mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)  # type: ignore
    except Exception as e:
        print(f"Error in distributed training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()