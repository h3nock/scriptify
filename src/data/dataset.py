import os
import numpy as np 
from src.data.loader import get_writerID, get_text_line_by_line, get_stroke_seqs, list_files 
from src.data.preprocessing import pad_line, create_mask 
from src.utils.stroke_viz import plot_stroke_seq

class OnlineHandwritingDataset:
    """
    """
    def __init__(self, ascii_root, extreme_threshold = 100):
        self.ascii_files = list_files(ascii_root) 
        self.extreme_threshold = extreme_threshold 
        self.MAX_STROKE_LENGTH = 0
        self.MAX_TEXT_LENGTH = 0
        self.ALPHABET = [] 
        self.samples_loaded = False
        self.ALPHABET_SIZE = 83 
    
    def _encode_text(self, text):
        #TODO: One hot encoding 
        return np.array([ord(c) for c in text], dtype=np.int8) 
    
    def _load_samples(self):
        texts_per_line = []
        strokes_per_line = []
        writerIDs = []
        for ascii_file in self.ascii_files[:2]:
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
                print(f"The current lines in the text and stroke files don't match. File name: {ascii_file}")
                continue
            
            original_root_folder = head.replace("ascii", "original")
            original_xml_path = os.path.join(original_root_folder,"strokes" + last_char + ".xml")  
            writerID = get_writerID(original_xml_path) 

            # curr_text_strokes = []
            texts_per_line.extend(text_lines)
            self.MAX_TEXT_LENGTH = max( self.MAX_TEXT_LENGTH, max([len(x) for x in text_lines]))

            for stroke_file in stroke_files:
                strokes = get_stroke_seqs(stroke_file)
                # print("Len(strokes): ", len(strokes))
                self.MAX_STROKE_LENGTH = max(self.MAX_STROKE_LENGTH, len(strokes))
                strokes_per_line.append(strokes) 
                writerIDs.append(writerID) 
                # curr_text_strokes.append(strokes)
            # plot_stroke_seq(curr_text_strokes) 
        
        assert len(texts_per_line) == len(strokes_per_line) == len(writerIDs), \
            "Mismatch between texts, strokes, and writer IDs." 
        
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

        processed_data = {
            'strokes': strokes_padded,
            'strokes_len': strokes_length,
            'chars': char_sequences,
            'chars_len': char_lengths,
            'writer_ids': np.array(writerIDs, dtype=np.int16),
            'mask': np.array([create_mask(strokes_per_line[i], self.MAX_STROKE_LENGTH) 
                            for i in range(num_samples)])
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
        