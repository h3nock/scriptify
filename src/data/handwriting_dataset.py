import os
from idna import encode
import numpy as np 
from src.data.loader import get_writerID, get_text_line_by_line, get_stroke_seqs, list_files 
from src.data.preprocessing import normalize, pad_line, create_mask, stroke_coords_to_offsets
from src.utils.stroke_viz import plot_stroke_seq


FIXED_MAX_STROKE_LEN = 1200
FIXED_MAX_TEXT_LEN = 80
FILTER_THRESHOLD = 60

class OnlineHandwritingDataset:
    """
    Dataset class for online handwriting. 

    Each sample corresponds to a text line from .txt file in the ascii folder and 
    its associated stroke data obtained from lineStrokes folder. 
    """

    def __init__(self, ascii_root, extreme_threshold = FILTER_THRESHOLD):
        """
        Params: 
            ascii_root: Dirctory containing text files. 
            extreme_threshold: max allowed distance between two consecutive stroke points
        """

        self.ascii_files = list_files(ascii_root) 
        self.extreme_threshold = extreme_threshold 
        self.MAX_STROKE_LENGTH = FIXED_MAX_STROKE_LEN
        self.MAX_TEXT_LENGTH = FIXED_MAX_TEXT_LEN
        self.ALPHABET = ['\x00',' ', '!', '"', '#', "'", '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', 
                    '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
                    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 
                    'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 
                    'v', 'w', 'x', 'y', 'z']
        self.samples_loaded = False
        self.char_to_index = {char: idx for idx, char in enumerate(self.ALPHABET)}
        self.ALPHABET_SIZE = len(self.ALPHABET)
    
    def _encode_text(self, text):
        """
        Encodes a text string into a sequence of integer indices based on the dataset alphabet.
        Characters not found in the alphabet default to index 0.
        """
        encoded = [self.char_to_index.get(c,0) for c in text]
        encoded.append(0) # append 0 as null for eos marker 
        encoded = np.array(encoded, dtype=np.int64) 
        return encoded[:self.MAX_TEXT_LENGTH] 
     
    def _load_samples(self):
        """
        Loads and processes all samples from the ASCII and stroke files.
        
        Returns:
            A dictionary with processed numpy arrays for strokes (offsets), stroke lengths,
            encoded text, text lengths, writer IDs, and a mask for valid stroke positions. 
        """
        texts_per_line = []
        strokes_per_line = []
        writerIDs = []
        for ascii_file in self.ascii_files: 
            # sample ascii_file: ascii/a01/a01-000/a01-000u.txt 

            text_lines = get_text_line_by_line(ascii_file) 

            # head: ascii/a01/a01-000/, tail: a01-000u.txt 
            head, tail = os.path.split(ascii_file) 

            # stroke_file_base: lineStrokes/a01/a01-000 
            stroke_files_base = head.replace("ascii", "lineStrokes")
            
            last_char = os.path.splitext(tail)[0][-1] 
            if not last_char.isalpha():
                last_char = "" 
            if not os.path.isdir(stroke_files_base):
                continue 
            # stroke_file_name_prefix: a01-000u-
            stroke_file_name_prefix = os.path.split(stroke_files_base)[-1] + last_char + "-"

            # list of filenames where each file corresponds to stroke respresentation of a single text line 
            stroke_files = sorted([os.path.join(stroke_files_base, f) for f in os.listdir(stroke_files_base) if f.startswith(stroke_file_name_prefix)]) 

            if not stroke_files or len(stroke_files) != len(text_lines):
                continue

            original_root_folder = head.replace("ascii", "original")
            original_xml_path = os.path.join(original_root_folder,"strokes" + last_char + ".xml")  
            writerID = get_writerID(original_xml_path) 

            # curr_text_strokes = []

            for i,stroke_file in enumerate(stroke_files):
                strokes = get_stroke_seqs(stroke_file)
                offsets = stroke_coords_to_offsets(strokes)
                offsets = offsets[:self.MAX_STROKE_LENGTH] 

                offsets = normalize(offsets) # normalize offsets 

                # filter based on offset magnitude 
                offset_magnitudes = np.linalg.norm(offsets[:,:2], axis = 1)  
                if np.any(offset_magnitudes > self.extreme_threshold):
                    continue # skip this sample 
                texts_per_line.append(text_lines[i])
                strokes_per_line.append(offsets) 
                writerIDs.append(writerID) 
                # curr_text_strokes.append(strokes)
            # plot_stroke_seq(curr_text_strokes) 
        
        assert len(texts_per_line) == len(strokes_per_line) == len(writerIDs), \
            "Mismatch between texts, strokes, and writer IDs." 
        
        num_samples = len(texts_per_line)

        strokes_padded = np.zeros([num_samples, self.MAX_STROKE_LENGTH, 3], dtype=np.float32)
        strokes_length = np.zeros([num_samples], dtype=np.int16)

        for i, stroke in enumerate(strokes_per_line):
            length = len(stroke)
            strokes_padded[i, :length, :] = stroke
            strokes_length[i] = length  

        # encode and pad each text line.
        char_sequences = np.zeros([num_samples, self.MAX_TEXT_LENGTH], dtype=np.int64)
        char_lengths = np.zeros([num_samples], dtype=np.int16)

        for i, text in enumerate(texts_per_line):
            encoded_text = self._encode_text(text)
            length = len(encoded_text) 
            char_sequences[i, :length] = encoded_text
            char_lengths[i] = length

        processed_data = {
            'strokes': strokes_padded,
            'strokes_len': strokes_length,
            'chars': char_sequences,
            'chars_len': char_lengths,
            'writer_ids': np.array(writerIDs, dtype=np.int16),
        }

        self.samples_loaded = True
        return processed_data


if __name__ == "__main__":
    dataset = OnlineHandwritingDataset("data/raw/ascii")
    data = dataset._load_samples()
    
    print(f"ALPHABET_SIZE: {dataset.ALPHABET_SIZE}")
    print(f"MAX_TEXT_LENGTH: {dataset.MAX_TEXT_LENGTH}")
    print(f"MAX_STROKE_LENGTH: {dataset.MAX_STROKE_LENGTH}")

    processed_dir = "data/processed"
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)
    
    np.save(os.path.join(processed_dir, "strokes.npy"), data['strokes'])
    np.save(os.path.join(processed_dir, "strokes_len.npy"), data['strokes_len'])
    np.save(os.path.join(processed_dir, "chars.npy"), data['chars'])
    np.save(os.path.join(processed_dir, "chars_len.npy"), data['chars_len'])
    np.save(os.path.join(processed_dir, "writer_ids.npy"), data['writer_ids'])

    print("Processed data saved successfully.")
