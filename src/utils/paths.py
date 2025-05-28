import shutil 
from datetime import datetime  
from pathlib import Path 
from typing import Union, Optional  

class RunPaths:
    """Centralized path management for a single training runs."""
    def __init__(self, run_name: Optional[str] = None, base_outputs_dir: Union[str, Path] = "outputs"):
        self.base_outputs_dir = Path(base_outputs_dir)
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
            run_name = f"run_{timestamp}"
        self.run_name = run_name 
        self.run_dir = self.base_outputs_dir / run_name 
        
        # subdirectories per run dir 
        self.checkpoints_dir = self.run_dir / "checkpoints" 
        self.logs_dir = self.run_dir / "logs" 
        self.plots_dir = self.run_dir / "plots" 
        self.config_dir = self.run_dir / "config" 

        # plot organization 
        self.training_plots_dir = self.plots_dir / "training" 
        self.sample_plots_dir = self.plots_dir / "samples" 

        # standard file paths 
        self.training_log = self.logs_dir / "training.log" 
        self.loss_plot = self.training_plots_dir / "training_loss.png" 
        self.config_copy = self.config_dir / "config.yaml" 
    
    def create_directories(self):
        """Create all necessary directories"""    
        dirs = [
            self.run_dir, self.checkpoints_dir, self.logs_dir, self.plots_dir, self.config_dir, self.training_plots_dir, self.sample_plots_dir
        ]
        for dir in dirs:
            dir.mkdir(parents=True, exist_ok=True)

    def copy_config_file(self, source_config_path: Union[str, Path]):
        source_path = Path(source_config_path) 
        if source_path.exists():
            shutil.copy2(source_path, self.config_copy) 
            print(f"Config file copied to {self.config_copy}")
        else:
            print(f"Warnning: Source config filt not found at {source_path}")
         
    def get_training_plot_path(self, plot_name: str) -> Path:
        if not plot_name.endswith(".png"):
            plot_name += '.png' 
        return self.training_plots_dir / plot_name  
    
    def get_sample_plot_path(self, text: str, step: Optional[int] = None,
                             suffix: str = "") -> Path:
        clean_text = "".join(c for c in text if c.isalnum() or c in (' ', '_')).strip() 
        clean_text = clean_text.replace(' ', '_')[:30] 
        filename_parts = [clean_text] 
        
        if step is not None:
            filename_parts.append(f"step_{step}") 
        if suffix:
            filename_parts.append(suffix) 
        
        filename = "_".join(filename_parts) + ".png" 
        return self.sample_plots_dir / filename 
            
            
    
    @classmethod 
    def find_latest_run(cls, base_outputs_dir: str = "outputs") -> Optional['RunPaths']:
        latest_run_dir = find_latest_run_dir(base_outputs_dir) 
        if latest_run_dir is None:
            return None 
        return cls(run_name=latest_run_dir.name, base_outputs_dir=base_outputs_dir) 

def find_latest_run_dir(base_outputs_dir: Union[str, Path] = "outputs") -> Optional['Path']: 
    """Find the latest run directory in the outputs folder."""
    outputs_path = Path(base_outputs_dir) 
    if not outputs_path.exists():
        print(f"Error: Outputs directory '{base_outputs_dir}' not found.") 
        return None 
    
    run_dirs = list(outputs_path.glob("run_*")) 

    if not run_dirs:
        print(f"Error: No run directories found in '{base_outputs_dir}'")
        return None 
    
    run_dirs.sort(key=lambda x : x.stat().st_mtime, reverse=True) 
    latest_run = run_dirs[0] 
    
    return latest_run

def find_latest_checkpoint(checkpoints_dir: Union[str, Path]) -> Optional[Path]:
    """Find the latest checkpoint in `checkpoints_dir` directory"""
    checkpoints_path = Path(checkpoints_dir) 
    if not checkpoints_path.exists():
        return None 
    checkpoint_files = list(checkpoints_path.glob('model-*'))
    if not checkpoint_files:
        return None 
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True) 
    return checkpoint_files[0] 

def find_latest_run_checkpoint(base_outputs_dir: Union[str,Path] = "outputs") -> Optional[Path]:
    """Find the latest checkpoint from the latest run."""
    latest_run = find_latest_run_dir(base_outputs_dir=base_outputs_dir) 

    if latest_run is None:
        return None 
    checkpoints_dir = latest_run / "checkpoints" 
    return find_latest_checkpoint(checkpoints_dir)