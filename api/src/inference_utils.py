from typing import Dict
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
    
    return np.array([padded_encoded]), true_length


def convert_offsets_to_absolute_coords(stroke_offsets: list[list[float]]) -> list[list[float]]:
    if not stroke_offsets:
        return []
    
    # convert to numpy for vectorized operations
    strokes_array = np.array(stroke_offsets)
    
    # vectorized cumulative sum for x and y 
    strokes_array[:, 0] = np.cumsum(strokes_array[:, 0])  # cumulative dx
    strokes_array[:, 1] = np.cumsum(strokes_array[:, 1])  # cumulative dy
    
    return strokes_array.tolist()