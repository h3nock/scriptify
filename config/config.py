import os
import yaml 
from pathlib import Path
from typing import Union 
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field, ValidationInfo, field_validator

load_dotenv()

PROJECT_ROOT  =  Path(__file__).resolve().parent.parent 

class Paths(BaseModel):
    raw_data_root: Path
    raw_ascii_subdir_name: str = "ascii"
    raw_line_strokes_name: str = "lineStrokes"
    raw_original_xml_subdir_name: str = "original"
    log_filename: str = "training.log"
    processed_data_dir: Path 
    outputs_dir: Path
    
    @field_validator(
        "raw_data_root",
        "outputs_dir",
        "processed_data_dir",
        mode='before')
    @classmethod 
    def resolve_to_abs_path(cls, v: Union[str, Path], info: ValidationInfo) -> Union[str,Path]: 
        if isinstance(v, str):
            path_obj = Path(v) 
        elif isinstance(v, Path):
            path_obj = v
        else:
            raise ValueError("Path must be a string or Path object")
        
        if not path_obj.is_absolute():
                return (PROJECT_ROOT / path_obj).resolve()
        return path_obj.resolve() 
    
    @property 
    def raw_ascii_dir(self) -> Path:
        return self.raw_data_root / self.raw_ascii_subdir_name 

    @property 
    def raw_line_strokes_dir(self) -> Path:
        return self.raw_data_root / self.raw_line_strokes_name 
    
    @property 
    def raw_original_xml_dir(self) -> Path:
        return self.raw_data_root / self.raw_original_xml_subdir_name
    
    @property 
    def runs_dir(self) -> Path:
        return self.outputs_dir / "runs"
class Dataset(BaseModel):
    train_split_ration: float = Field(0.9, ge=0.0, le=1.0) 
    random_seed: int = 42
    max_stroke_len: int = Field(1200, gt=0)
    max_text_len: int = Field(80, gt=0)
    offset_filter_threshold: int = Field(60, gt=0)

class ModelParams(BaseModel): 
    lstm_size: int = Field(400, ge=0)
    output_mixture_components: int = Field(20, gt=0)
    attention_mixture_components: int = Field(10, gt=0) 
    dropout_prob: float = Field(0.2, gt=0.0, le=1.0)
    eps: float = Field(1e-8, ge=0) 
    sigma_eps: float = Field(1e-4, ge=0)

class TrainingParams(BaseModel): 
    batch_sizes: list[int] = [192, 192, 192]
    learning_rates: list[float] = [1e-4, 5 * 1e-5, 2*1e-5]
    beta1_decays: list[float] = [0.9, 0.9, 0.9]
    patiences: list[int] = [1500, 1000, 500]
    optimizer_type: str = "adam"
    weight_decay: float = Field(1e-5, gt=0)
    grad_clip: int = Field(10, gt=0)
    num_training_steps: int = Field(10 ** 5, gt=0)
    log_interval: int = Field(10, gt=0, description="Number of steps between logs")
    
class PredictionParams(BaseModel):
    max_length: int = Field(1000, gt=0)

class DistributedTrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='SCRIPTIFY_DIST_', extra='ignore', case_sensitive=False)

    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: int = 29500

class WandBConfig(BaseModel):
    enabled: bool = Field(True, description="enable weights and biases logging") 
    project_name: str = Field("Scriptify")
    run_name: Union[str, None] = Field(None)
    tags: list[str] = Field(default_factory=list)
    notes: Union[str, None] = Field(None)
    log_model_checkpoint_freq: int = Field(0)
    watch_model_log_freq: int = Field(100)

class Config(BaseModel):
    paths: Paths
    dataset: Dataset 
    model_params: ModelParams 
    training_params: TrainingParams 
    prediction_params: PredictionParams 
    distributed_training: DistributedTrainingConfig = Field(default_factory=DistributedTrainingConfig)
    wandb: WandBConfig
     
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_VAR_CONFIG_PATH_NAME = "SCRIPTIFY_CONFIG_PATH"

def load_config(config_path: Union[str, Path, None] = None) -> Config:
    final_config_path: Path 
    if config_path is not None:
        if isinstance(config_path, str):
            final_config_path = Path(config_path) 
        elif isinstance(config_path, Path):
            final_config_path = config_path 
        else:
            raise TypeError(f"config_path must be a string or Path obj or None to use the default path config/config.yaml")
    else:
        env_path = os.getenv(ENV_VAR_CONFIG_PATH_NAME) 
        if env_path:
            final_config_path = Path(env_path) 
        else:
            final_config_path = DEFAULT_CONFIG_PATH 

    if not final_config_path.is_absolute():
        final_config_path = (PROJECT_ROOT / final_config_path).resolve()
    else:
        final_config_path.resolve() 
    
    if not final_config_path.exists():
        raise FileNotFoundError(f"Config filie not found at {final_config_path}") 
    
    if not final_config_path.is_file():
        raise IsADirectoryError(f"Config path {final_config_path} is a directory")
    
    try: 
        with open(final_config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if config_data is None:
            raise ValueError(f"Config file {final_config_path} is empty")
        return Config(**config_data) 
    except Exception as e:
        raise Exception(f"Error occurred while parsing yaml. Error: {e}")

if __name__ == "__main__":
    config = load_config() 
    print(config.paths) 
    print(config.model_params)
    print(config.prediction_params)