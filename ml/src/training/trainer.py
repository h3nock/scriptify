import time  
import wandb
import numpy as np  
import torch  
from torch.utils.data import DataLoader  
from collections import deque  
import logging
from torch.utils.data.distributed import DistributedSampler 
from src.data.dataloader import ProcessedHandwritingDataset
from .loss import gaussian_mixture_loss  
from .optimizer import get_optimizer, get_lr_scheduler  
from config.config import TrainingParams, WandBConfig
from pathlib import Path 
from src.utils.paths import RunPaths
 
class HandwritingTrainer:  
    def __init__(  
        self,  
        model,  
        train_dataset,  
        val_dataset,  
        training_params: TrainingParams, 
        run_paths: RunPaths,
        config_file_path: Path, 
        wandb_config: WandBConfig,
        device=None, 
        world_size =1, 
        rank=0  
    ):  
        self.model = model
        self.world_size = world_size 
        self.rank = rank   
        self.train_dataset = train_dataset  
        self.val_dataset = val_dataset  
        self.training_params = training_params
        self.config_file_path = config_file_path
        self.wandb_config = wandb_config
        self.run_paths = run_paths 
        # unpack training params 
        self.batch_sizes = training_params.batch_sizes  
        self.learning_rates = training_params.learning_rates  
        self.beta1_decays = training_params.beta1_decays  
        self.patiences = training_params.patiences  
        self.optimizer_type = training_params.optimizer_type  
        self.grad_clip = training_params.grad_clip  
        self.num_training_steps = training_params.num_training_steps  
        self.log_interval = training_params.log_interval  
        # unpack paths 
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging  
        logging.basicConfig(  
            level=logging.INFO,  
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  
            handlers=[  
                logging.FileHandler(self.run_paths.training_log),  
                logging.StreamHandler()  
            ]  
        )  
        self.logger = logging.getLogger('HandwritingTrainer')  
          
        # Move model to device  
        self.model.to(device)  
          
        # Initialize training phase  
        self.restart_idx = 0  
        self.update_train_params()  
          
    def update_train_params(self):  
        """Update training parameters for current phase"""  
        self.batch_size = self.batch_sizes[self.restart_idx]
        self.learning_rate = self.learning_rates[self.restart_idx]
        self.learning_rate *= np.sqrt(self.world_size) 
        self.beta1_decay = self.beta1_decays[self.restart_idx]
        self.patience = self.patiences[self.restart_idx]
        
        if self.world_size > 1:
            # use distributed samplers for multi-GPU training
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
        else:
            # use regular samplers for single GPU training
            train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
            val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )

        if not hasattr(self, "optimizer") or self.optimizer is None:
            self.optimizer = get_optimizer(  
                self.model,  
                optimizer_type=self.optimizer_type,  
                learning_rate=self.learning_rate,  
                beta1=self.beta1_decay  
            )  
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                if 'betas' in param_group: # for Adam, AdamW
                    param_group['betas'] = (self.beta1_decay, param_group['betas'][1])
        
        
        self.scheduler = get_lr_scheduler(  
            self.optimizer,  
            scheduler_type='plateau',  
            patience=self.patience // 10,  
            factor=0.5  
        )  
        
        self.logger.info(f"Updated training parameters: batch_size={self.batch_size}, "  
                        f"learning_rate={self.learning_rate}, beta1={self.beta1_decay}")
    def train_step(self, batch):  
        """Perform a single training step"""  
        self.model.train()  
          
        x = batch['stroke'].to(self.device)  
        # for training, y is the next point in the sequence  
        # we need to shift the input by one timestep  
        y = x[:, 1:, :]  # target is input shifted by 1  
        x = x[:, :-1, :]  # input is all but the last timestep  
          
        x_len = batch['stroke_len'].to(self.device) - 1  # adjust length for the shift  
        c = batch['chars'].to(self.device)  
        c_len = batch['chars_len'].to(self.device)  
          
        # forward pass  
        pis, sigmas, rhos, mus, es, _ = self.model(x, c, c_len)  
          
        # calculate loss  
        sequence_loss, element_loss = gaussian_mixture_loss(y, x_len, pis, sigmas, rhos, mus, es)  
        loss = element_loss  
          
        # backward pass  
        self.optimizer.zero_grad()  
        loss.backward()  
          
        # gradient clipping  
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  
          
        # update weights  
        self.optimizer.step()  
          
        return loss.item(), sequence_loss.detach().cpu().numpy()  
      
    def validate(self):  
        """Validate the model on the validation set"""  
        self.model.eval()  
        val_losses = []  
          
        with torch.no_grad():  
            for batch in self.val_loader:  
                x = batch['stroke'].to(self.device)  
                # for validation, y is the next point in the sequence  
                y = x[:, 1:, :]  
                x = x[:, :-1, :] 
                  
                x_len = batch['stroke_len'].to(self.device) - 1 
                c = batch['chars'].to(self.device)  
                c_len = batch['chars_len'].to(self.device)  
                  
                pis, sigmas, rhos, mus, es, _ = self.model(x, c, c_len)  
                  
                _, element_loss = gaussian_mixture_loss(y, x_len, pis, sigmas, rhos, mus, es)  
                val_losses.append(element_loss.item())  

        local_avg_loss = np.mean(val_losses) 
        # gather and average losses across all GPUs
        if self.world_size > 1:
            local_loss_tensor = torch.tensor([local_avg_loss]).to(self.device)
            all_losses = [torch.zeros_like(local_loss_tensor) for _ in range(self.world_size)]
            torch.distributed.all_gather(all_losses, local_loss_tensor)
            
            # average losses from all GPUs
            global_avg_loss = torch.mean(torch.stack(all_losses)).item()
            return global_avg_loss, local_avg_loss
        else:
            return local_avg_loss, local_avg_loss 
      
    def save_checkpoint(self, epoch, step, val_loss):  
        """Save model checkpoint"""  
        if self.rank == 0:
            file_name = f'model-{step}'
            checkpoint_path = self.run_paths.checkpoints_dir / file_name  
            torch.save({  
                'step': step,  
                'model_state_dict': self.model.state_dict(),  
                'optimizer_state_dict': self.optimizer.state_dict(),  
                'val_loss': val_loss,  
                'restart_idx': self.restart_idx  
            }, checkpoint_path)  
            self.logger.info(f"Saved checkpoint locally to {checkpoint_path}")  
            
            # save to wandb 
            if self.wandb_config.enabled and wandb.run:
                model_artifact = wandb.Artifact(
                    name="scriptify", 
                    type="model", 
                    description=f"Handwriting model checkpoint. Step: {step}, Epoch: {epoch}, Val Loss: {val_loss:.4f}",
                    metadata={
                        'step': step,
                        'epoch': epoch,
                        'validation_loss': val_loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                    }
                )
                model_artifact.add_file(str(checkpoint_path),name=file_name)
                model_artifact.add_file(str(self.config_file_path), name="config.yaml") 
               
                wandb.log_artifact(model_artifact) 
                self.logger.info(f"Logged model artifact to WandB")
    def load_checkpoint(self, step=None):  
        """Load model checkpoint"""  
        if step is None:  
            # find latest checkpoint 
            checkpoints = [p for p in self.run_paths.checkpoints_dir.glob(f"model-*") if p.is_file()]
            if not checkpoints:  
                self.logger.info("No checkpoints found, starting from scratch")  
                return 0  
              
            # extract step numbers and find the latest  
            steps = [int(f.name.split('-')[1]) for f in checkpoints]  
            step = max(steps)  
          
        checkpoint_path = self.run_paths.checkpoints_dir / f'model-{step}'
        if not checkpoint_path.exists():  
            self.logger.error(f"Checkpoint {checkpoint_path} not found")  
            return 0  
          
        checkpoint = torch.load(checkpoint_path, map_location=self.device)  
        self.model.load_state_dict(checkpoint['model_state_dict'])  
        self.restart_idx = checkpoint['restart_idx']  
        self.update_train_params()  
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
          
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")  
        return step  
      
    def fit(self, warm_start_step=None):  
        """Train the model"""  
        # load checkpoint if warm starting  
        if warm_start_step is None:
            self.logger.info("warm_start_step is None, attempting to load the latest checkpoint") 
            step = self.load_checkpoint()
        elif warm_start_step > 0:
            step = self.load_checkpoint(warm_start_step)
        else:
            step = 0 
            self.logger.info("warm_start_step is 0, starting training from scratch.")
          
        # initialize training history  
        train_loss_history = deque(maxlen=100)  

        # initialize best validation metrics  
        best_val_loss = float('inf')  
        best_val_step = step  
        
        batch_size_per_gpu = self.batch_size
        total_batch_size = batch_size_per_gpu * self.world_size
        steps_per_epoch = max(1, len(self.train_dataset) // total_batch_size)
        max_epochs = (self.num_training_steps - step) // steps_per_epoch + 1

        self.logger.info(f"Starting training for {max_epochs} epochs ({steps_per_epoch} steps per epoch)")

        epoch = 0

        while step < self.num_training_steps and epoch < max_epochs:  

            epoch_start_time = time.time() 
            epoch_train_losses = [] 

            if self.world_size > 1:
                self.train_loader.sampler.set_epoch(epoch)  # type: ignore
            
            # train for one epoch 
            self.model.train() 
            for batch_idx, batch in enumerate(self.train_loader):
                if step >= self.num_training_steps:
                    break 
                train_start = time.time()
                train_loss , _ = self.train_step(batch) 
                train_time = time.time() - train_start
                
                epoch_train_losses.append(train_loss)
                if self.rank == 0: 
                    train_loss_history.append(train_loss) 

                if step % self.log_interval == 0 and self.rank == 0:
                    avg_train_loss_rank0 = sum(train_loss_history) / len(train_loss_history) if train_loss_history else float('nan')

                    self.logger.info(
                        f"epoch {epoch}, step {step}: "
                        f"train loss (rank 0 current step): {train_loss:.6f} "
                        f"train loss (rank 0 recent avg): {avg_train_loss_rank0:.6f}, "
                        f"lr: {self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                    if self.wandb_config.enabled and wandb.run:
                        wandb.log({
                            "train/loss_step_rank0": train_loss, # current step 
                            "train/loss_avg_hist_rank0": avg_train_loss_rank0, # avg local loss on rank 0 (considering the recent 100 or less steps)
                            "learning_rate": self.optimizer.param_groups[0]['lr'],
                            "epoch": epoch
                        }, step=step)
                        
                step += 1 

            val_start = time.time()
            val_loss, local_val_loss = self.validate() # global validation loss, current rank validation loss
            val_time = time.time() - val_start

            # log epoch summary
            avg_local_epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else float('nan')
            global_avg_epoch_train_loss = avg_local_epoch_train_loss 
            if self.world_size > 1:
                local_loss_tensor = torch.tensor([avg_local_epoch_train_loss], dtype=torch.float32, device=self.device)
                
                # avg across all processes
                torch.distributed.all_reduce(local_loss_tensor, op=torch.distributed.ReduceOp.AVG)
                global_avg_epoch_train_loss = local_loss_tensor.item()

            epoch_duration = time.time() - epoch_start_time
            
            if self.rank == 0:
                log_message_epoch = (
                    f"epoch {epoch} completed in {epoch_duration:.2f}s: "
                    f"train loss (global avg): {global_avg_epoch_train_loss:.6f}, "
                    f"val loss (global avg): {val_loss:.6f}"
                )
                if self.world_size > 1:
                    log_message_epoch += (
                        f", train loss (rank 0 local avg): {avg_local_epoch_train_loss:.6f}"
                        f", val loss (rank 0 local avg): {local_val_loss:.6f}" 
                    )

                self.logger.info(log_message_epoch)

                if self.wandb_config.enabled and wandb.run:
                    log_dict = {
                        "epoch/train_loss_global_avg": global_avg_epoch_train_loss,
                        "epoch/val_loss_global_avg": val_loss,
                        "epoch/num": epoch, 
                        "epoch/duration_sec": epoch_duration, 
                        "epoch/val_time_sec": val_time 
                    }
                    if self.world_size > 1:
                        log_dict["epoch/train_loss_rank0_avg"] = avg_local_epoch_train_loss 
                        log_dict["epoch/val_loss_rank0_avg"] = local_val_loss
                    wandb.log(log_dict, step=step)
            
            # check for new best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_step = step
                self.save_checkpoint(epoch,step, val_loss)
                
            # check for early stopping
            if step - best_val_step > self.patience:
                if self.restart_idx < len(self.batch_sizes) - 1:
                    # load best checkpoint 
                    self.logger.info(f"validation loss plateaued. moving to next training phase.")
                    self.load_checkpoint(best_val_step)
                    self.restart_idx += 1
                    self.update_train_params()
                    best_val_loss = float('inf')
                    best_val_step = step
                else:
                    self.logger.info(f"early stopping at step {step}. best validation loss: {best_val_loss:.6f} at step {best_val_step}")
                    break
            
            # update learning rate 
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
                
            epoch += 1
          
        self.logger.info(f"Training completed after {step} steps and {epoch} epochs")
        return best_val_step  
      
    def predict(self, text_batch, max_length=1000, bias=0.5):  
        """Generate handwriting for a batch of text"""  
        self.model.eval()  
          
        alphabet = ProcessedHandwritingDataset.get_alphabet()  
          
        # create a function to encode text  
        def encode_ascii(text):  
            char_to_index = {char: idx for idx, char in enumerate(alphabet)}  
            return np.array([char_to_index.get(c, 0) for c in text], dtype=np.int8)  
          
        # prepare input  
        c = torch.tensor([encode_ascii(text) for text in text_batch], device=self.device)  
        c_len = torch.tensor([len(text) for text in text_batch], device=self.device)  
          
        # generate strokes  
        with torch.no_grad():  
            strokes = self.model.sample(c, c_len, max_length=max_length, bias=bias)  
          
        return strokes.cpu().numpy()