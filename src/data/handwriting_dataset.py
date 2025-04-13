import os
import numpy as np 
from src.data.loader import get_writerID, get_text_line_by_line, get_stroke_seqs, list_files 
from src.data.preprocessing import pad_line, create_mask, stroke_coords_to_offsets
from src.utils.stroke_viz import plot_stroke_seq

class OnlineHandwritingDataset:
    """
    Dataset class for online handwriting. 

    Each sample corresponds to a text line from .txt file in the ascii folder and 
    its associated stroke data obtained from lineStrokes folder. 
    """

    def __init__(self, ascii_root, extreme_threshold = 100):
        """
        Params: 
            ascii_root: Dirctory containing text files. 
            extreme_threshold: max allowed distance between two consecutive stroke points
        """

        self.ascii_files = list_files(ascii_root) 
        self.extreme_threshold = extreme_threshold 
        self.MAX_STROKE_LENGTH = 0
        self.MAX_TEXT_LENGTH = 0
        self.ALPHABET = ['\x00',' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', 
                    '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
                    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 
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
        return np.array([self.char_to_index.get(c, 0) for c in text], dtype=np.int8)
    
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
                # print(f"The current lines in the text and stroke files don't match. File name: {ascii_file}")
                continue

            original_root_folder = head.replace("ascii", "original")
            original_xml_path = os.path.join(original_root_folder,"strokes" + last_char + ".xml")  
            writerID = get_writerID(original_xml_path) 

            # curr_text_strokes = []
            texts_per_line.extend(text_lines)
            self.MAX_TEXT_LENGTH = max( self.MAX_TEXT_LENGTH, max([len(x) for x in text_lines]))

            for stroke_file in stroke_files:
                strokes = get_stroke_seqs(stroke_file)
                offsets = stroke_coords_to_offsets(strokes)
                # print("Len(strokes): ", len(strokes))
                self.MAX_STROKE_LENGTH = max(self.MAX_STROKE_LENGTH, len(strokes))
                strokes_per_line.append(offsets) 
                writerIDs.append(writerID) 
                # curr_text_strokes.append(strokes)
            # plot_stroke_seq(curr_text_strokes) 
        
        assert len(texts_per_line) == len(strokes_per_line) == len(writerIDs), \
            "Mismatch between texts, strokes, and writer IDs. File name: {fname}" 
        
        num_samples = len(texts_per_line)

        strokes_padded = np.zeros([num_samples, self.MAX_STROKE_LENGTH, 3], dtype=np.float32)
        strokes_length = np.zeros([num_samples], dtype=np.int16)

        for i, stroke in enumerate(strokes_per_line):
            padded_stroke = pad_line(stroke, max_points=self.MAX_STROKE_LENGTH)
            strokes_padded[i] = padded_stroke
            strokes_length[i] = min(len(stroke), self.MAX_STROKE_LENGTH)

        # encode and pad each text line.
        char_sequences = np.zeros([num_samples, self.MAX_TEXT_LENGTH], dtype=np.int8)
        char_lengths = np.zeros([num_samples], dtype=np.int8)

        for i, text in enumerate(texts_per_line):
            encoded_text = self._encode_text(text)
            length = min(len(encoded_text), self.MAX_TEXT_LENGTH)
            char_sequences[i, :length] = encoded_text[:length]
            char_lengths[i] = length

        # create a mask for the stroke data
        stroke_mask = np.array([create_mask(strokes_per_line[i], max_points= self.MAX_STROKE_LENGTH)
                                  for i in range(num_samples)], dtype=np.float32)
        processed_data = {
            'strokes': strokes_padded,
            'strokes_len': strokes_length,
            'chars': char_sequences,
            'chars_len': char_lengths,
            'writer_ids': np.array(writerIDs, dtype=np.int16),
            'mask': stroke_mask
        }

        self.samples_loaded = True
        return processed_data


if __name__ == "__main__":
    dataset = OnlineHandwritingDataset("data/raw/ascii")
    data = dataset._load_samples()
    
    processed_dir = "data/processed"
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)
    
    np.save(os.path.join(processed_dir, "strokes.npy"), data['strokes'])
    np.save(os.path.join(processed_dir, "strokes_len.npy"), data['strokes_len'])
    np.save(os.path.join(processed_dir, "chars.npy"), data['chars'])
    np.save(os.path.join(processed_dir, "chars_len.npy"), data['chars_len'])
    np.save(os.path.join(processed_dir, "writer_ids.npy"), data['writer_ids'])
    np.save(os.path.join(processed_dir, "mask.npy"), data['mask'])

    print("Processed data saved successfully.")
