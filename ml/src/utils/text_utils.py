from pathlib import Path
from typing import Dict, Union
import numpy as np 

NULL_CHAR = '\x00'

def construct_alphabet_list(alphabet_string: str) -> list[str]:
    if not isinstance(alphabet_string, str):
        raise TypeError("alphabet_string must be a string") 
    
    char_list = list(alphabet_string) 
    return [NULL_CHAR] + char_list 

def get_alphabet_map(alphabet_list: list[str]) -> Dict[str, int]:
    """creates a char to index map from full alphabet list"""
    return {char: idx for idx, char in enumerate(alphabet_list)}  

def encode_text(text: str, char_to_index_map: Dict[str, int], 
                max_length: int, add_eos: bool = True, eos_char_index: int = 0
                ) -> tuple[np.ndarray, int]:
    """Encode a text string into a sequence of integer indices"""
    encoded = [char_to_index_map.get(c, eos_char_index) for c in text] 
    if add_eos:
        encoded.append(eos_char_index) 

    true_length = len(encoded)

    if true_length <= max_length: 
        padded_encoded = np.full(max_length, eos_char_index, dtype=np.int64) 
        padded_encoded[:true_length] = encoded 
    else:
        padded_encoded = np.array(encoded[:max_length], dtype=np.int64) 
        true_length = max_length 
    
    return padded_encoded, true_length


def load_np_strokes(stroke_path: Union[Path, str]) -> np.ndarray:
    """loads stroke sequence from stroke_path"""
    stroke_path = Path(stroke_path)
    if not stroke_path.exists():
        raise FileNotFoundError(f"style strokes file not found at {stroke_path}")
    
    return np.load(stroke_path)

def load_text(text_path: Union[Path, str]) -> str:
    """loads text from a text_path""" 
    text_path = Path(text_path) 
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found at {text_path}")
    if not text_path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file.")
    
    try: 
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read() 
        return content 

    except Exception as e:
        raise IOError(f"Error reading text file {text_path}: {e}")

def load_priming_data(style: int):
    
    priming_text = load_text(f"./data/samples/sample{style}.txt")
    priming_strokes = load_np_strokes(f"./data/samples/sample{style}.npy")
    
    return priming_text, priming_strokes 