import os
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from src.models.handwriting_rnn import HandwritingRNN
from src.trainer import HandwritingTrainer
from src.data.dataloader import ProcessedHandwritingDataset

def setup(rank, world_size):
    """Initialize distributed training process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

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
    print(f"Running training on GPU {rank}")
    
    # load dataset
    processed_dir = 'data/processed'
    full_dataset = ProcessedHandwritingDataset(processed_dir)
    
    # split dataset (90% train, 10% validation)
    generator = torch.Generator().manual_seed(42)
    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    if rank == 0:
        print(f"Dataset loaded: {dataset_size} samples")
        print(f"Training set: {train_size} samples")
        print(f"Validation set: {val_size} samples")
    
    # init model
    model = HandwritingRNN(
        lstm_size=400,
        output_mixture_components=20,
        attention_mixture_components=10,
        char_embedding_size=32,
        alphabet_size=82
    )
    
    model.to(device)
    ddp_model = DDP(model, device_ids=[rank])
    
    # init trainer 
    trainer = HandwritingTrainer(
        model=ddp_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_sizes=[32, 64, 64],
        learning_rates=[0.0001, 0.00005, 0.00002],
        beta1_decays=[0.9, 0.9, 0.9],
        patiences=[1500, 1000, 500],
        optimizer_type='adam',
        grad_clip=10.0,
        num_training_steps=100000,
        checkpoint_dir='checkpoints',
        log_dir='logs',
        log_interval=20,
        device=device,
        world_size=world_size,
        rank=rank
    )
    
    # start training
    if rank == 0:
        print("Starting distributed training...")
    best_step = trainer.fit()
    if rank == 0:
        print(f"Training completed. Best model at step {best_step}")
    
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