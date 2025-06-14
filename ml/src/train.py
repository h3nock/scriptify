import os
from typing import Optional, Union
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

RANK = 'RANK' # global rank of the process 
WORLD_SIZE = 'WORLD_SIZE' # total number of processes across all nodes  
LOCAL_RANK = 'LOCAL_RANK' # local rank of the process on the current node 

def setup(rank, world_size):
    """Initialize distributed training process group"""
    backend = os.environ.get('SCRIPTIFY_DIST_BACKEND', 'nccl')
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

def train(rank,local_rank, world_size):
    """Training function to run on each GPU"""
    if world_size > 1:
        # setup process group
        setup(rank, world_size)
    
    # set device for this process
    device = torch.device(f"cuda:{local_rank}")
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
    
    run_name_container: list[Union[str,None]] = [None]
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        run_name_container[0] = f"run_{timestamp}"

    if world_size > 1:
        dist.broadcast_object_list(run_name_container,0)

    run_name = run_name_container[0]
    run_paths = RunPaths(run_name, base_outputs_dir=config_global.paths.outputs_dir)
    wandb_run = None 
    if rank == 0: 
        run_paths.create_directories() 
        run_paths.copy_config_file(config_global.paths.config_file_path)
        
        if config_global.wandb.enabled:
            hyperparams = {
                **config_global.dataset.model_dump(),
                **config_global.model_params.model_dump(),
                **config_global.training_params.model_dump(),
                # Add environment info
                "world_size": world_size,
                "device_type": device.type,
                "local_rank": local_rank,
                "global_rank": rank,
                "master_addr": os.environ.get('MASTER_ADDR', 'unknown'),
                "master_port": os.environ.get('MASTER_PORT', 'unknown'),
                "hostname": os.environ.get('HOSTNAME', 'unknown'),
                "pytorch_version": torch.__version__,
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
    
    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank])
    else:
        ddp_model = model 
    
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

    if world_size > 1: 
        # clean up
        cleanup()

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        return 1

    if all(key in os.environ for key in [RANK, WORLD_SIZE, LOCAL_RANK]):

        rank = int(os.environ[RANK]) 
        world_size = int(os.environ[WORLD_SIZE])
        local_rank = int(os.environ[LOCAL_RANK])
        
        try:
            # set the multiprocessing start method
            torch.multiprocessing.set_start_method('spawn', force=True)
            
            train(rank, local_rank, world_size) 
        except Exception as e:
            print(f"Error in distributed training: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("Use torchrun to run this script") 
        return 1  

    return 0

if __name__ == "__main__":
    exit(main())