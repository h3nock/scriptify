import numpy as np 
from pathlib import Path
from src.data.loader import get_writerID, get_text_line_by_line, get_stroke_seqs, list_files 
from src.data.preprocessing import deskew_line, has_outlier, normalize, smooth_strokes, stroke_coords_to_offsets
from config.config import Paths as PathsConfig, Dataset as DatasetParams
from src.utils.text_utils import encode_text, get_alphabet_map, construct_alphabet_list
class OnlineHandwritingDataset:
    """
    Dataset class for online handwriting. 

    Each sample corresponds to a text line from .txt file in the ascii folder and 
    its associated stroke data obtained from lineStrokes folder. 
    """

    def __init__(self, paths_config: PathsConfig, dataset_params: DatasetParams):
        """
        Params: 
            ascii_root: Dirctory containing text files. 
            extreme_threshold: max allowed distance between two consecutive stroke points
        """
        self.paths_config = paths_config
        self.ascii_root_path: Path = paths_config.raw_ascii_dir
        self.ascii_files: list[Path] = list_files(self.ascii_root_path) 
        self.extreme_threshold = dataset_params.offset_filter_threshold 
        self.MAX_STROKE_LENGTH = dataset_params.max_stroke_len
        self.MAX_TEXT_LENGTH = dataset_params.max_text_len

        if not dataset_params.alphabet_string:
            raise ValueError("alphabet_string not provided in dataset_params")
        self.ALPHABET = construct_alphabet_list(dataset_params.alphabet_string) 
        self.samples_loaded = False
        self.char_to_index = get_alphabet_map(self.ALPHABET)
        self.ALPHABET_SIZE = len(self.ALPHABET)
    
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
        for ascii_file_path in self.ascii_files: 
            # sample ascii_file_path: ascii/a01/a01-000/a01-000u.txt 

            text_lines: list[str] = get_text_line_by_line(ascii_file_path)  
            
            # relative_dir = a01/a01-000/, given ascii_file_path = ascii/a01/a01-000/a01-000u.txt  
            relative_dir = ascii_file_path.parent.relative_to(self.ascii_root_path)

            # stroke_file_base: .../lineStrokes/a01/a01-000 
            stroke_files_base = self.paths_config.raw_line_strokes_dir / relative_dir 
            # orginal_xml_base_dir 
            original_xml_base_dir = self.paths_config.raw_original_xml_dir / relative_dir 
            
            last_char = ascii_file_path.stem[-1]
            if not last_char.isalpha():
                last_char = "" 
            
            if not stroke_files_base.is_dir():
                continue 

            # stroke_file_name_prefix: a01-000u-
            stroke_file_name_prefix = ascii_file_path.parent.name + last_char + "-"
            # list of filenames where each file corresponds to stroke respresentation of a single text line 
            stroke_files: list[Path] = sorted(stroke_files_base.glob(f"{stroke_file_name_prefix}*"))

            if not stroke_files or len(stroke_files) != len(text_lines):
                continue

            original_xml_path = original_xml_base_dir / f"strokes{last_char}.xml"

            if not original_xml_path.exists():
                continue 

            writerID = get_writerID(original_xml_path) 
            # curr_text_strokes = []

            for i,stroke_file in enumerate(stroke_files):
                strokes = get_stroke_seqs(stroke_file)
                strokes = deskew_line(strokes) 
                strokes = smooth_strokes(strokes=strokes)  
                
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
            encoded_text, true_length = encode_text(text=text, 
                                       max_length=self.MAX_TEXT_LENGTH,
                                       char_to_index_map=self.char_to_index)
            char_sequences[i, :] = encoded_text
            char_lengths[i] = true_length

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
    from config.config import load_config 
    config = load_config() 

    dataset = OnlineHandwritingDataset(paths_config=config.paths, dataset_params=config.dataset)
    data = dataset._load_samples()
    
    print(f"ALPHABET_SIZE: {dataset.ALPHABET_SIZE}")
    print(f"MAX_TEXT_LENGTH: {dataset.MAX_TEXT_LENGTH}")
    print(f"MAX_STROKE_LENGTH: {dataset.MAX_STROKE_LENGTH}")

    processed_dir = config.paths.processed_data_dir
    if not processed_dir.exists():
        processed_dir.mkdir(parents=True,exist_ok=True)
    
    np.save(processed_dir/ "strokes.npy", data['strokes'])
    np.save(processed_dir/ "strokes_len.npy", data['strokes_len'])
    np.save(processed_dir/ "chars.npy", data['chars'])
    np.save(processed_dir/ "chars_len.npy", data['chars_len'])
    np.save(processed_dir/ "writer_ids.npy", data['writer_ids'])

    print("Processed data saved successfully.")
