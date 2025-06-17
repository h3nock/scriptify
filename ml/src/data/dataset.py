import numpy as np 
from pathlib import Path
from src.data.raw_data_reader import SourceDocumentInfo, collect_and_organize_docs, get_stroke_seqs
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
        self.dataset_params = dataset_params 
        
        self.source_docs: list[SourceDocumentInfo] = collect_and_organize_docs(data_root_path=paths_config.raw_data_root)
        
        if not self.source_docs:
            print(f"No source documents were collected")

        self.extreme_threshold = dataset_params.offset_filter_threshold 
        self.MAX_STROKE_LENGTH = dataset_params.max_stroke_len
        self.MAX_TEXT_LENGTH = dataset_params.max_text_len

        if not dataset_params.alphabet_string:
            raise ValueError("alphabet_string not provided in dataset_params")
        self.ALPHABET = construct_alphabet_list(dataset_params.alphabet_string) 

        self.char_to_index = get_alphabet_map(self.ALPHABET)
        self.ALPHABET_SIZE = len(self.ALPHABET)
    
    def _load_samples(self):
        """
        Loads and processes all samples from the ASCII and stroke files.
        
        Returns:
            A dictionary with processed numpy arrays for strokes (offsets), stroke lengths,
            encoded text, text lengths, writer IDs, and a mask for valid stroke positions. 
        """

        all_lines_text = []
        all_lines_strokes_offsets = [] 
        all_lines_writer_ids = [] 
        
        if not self.source_docs:
            raise FileNotFoundError("No source documents to load samples from")
        
        for doc_info in self.source_docs:
            writerID = doc_info.writer_id 
            
            for i, line_text in enumerate(doc_info.text_lines):
                stroke_file_path = doc_info.line_stroke_file_paths[i] 
                
                try:
                    strokes = get_stroke_seqs(stroke_file_path) 
                    if len(strokes)== 0:
                        continue
                    strokes = deskew_line(coords=strokes) 
                    strokes = smooth_strokes(strokes=strokes) 
                    
                    offsets = stroke_coords_to_offsets(strokes)
                    
                    offsets = offsets[:self.MAX_STROKE_LENGTH] 

                    # normalize offsets 
                    offsets = normalize(offsets) 

                    all_lines_text.append(line_text) 
                    all_lines_strokes_offsets.append(offsets) 
                    all_lines_writer_ids.append(writerID)
                except FileNotFoundError:
                    print(f"Stroke file {stroke_file_path} not found, so skipping it.") 
                    continue 
                except Exception as e:
                    print(f"Error occured processing stroke file {stroke_file_path}. Skipping the file. The detail error: {e}")
                    continue 
              
        if not all_lines_text:
            print("No valid lines was processed from the source documents") 
                 
         
        num_samples = len(all_lines_text)

        strokes_padded = np.zeros([num_samples, self.MAX_STROKE_LENGTH, 3], dtype=np.float32)
        strokes_length = np.zeros([num_samples], dtype=np.int16)

        for i, stroke_offsets in enumerate(all_lines_strokes_offsets):
            length = len(stroke_offsets)
            strokes_padded[i, :length, :] = stroke_offsets
            strokes_length[i] = length  

        # encode and pad each text line.
        char_sequences = np.zeros([num_samples, self.MAX_TEXT_LENGTH], dtype=np.int64)
        char_lengths = np.zeros([num_samples], dtype=np.int16)

        for i, text in enumerate(all_lines_text):
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
            'writer_ids': np.array(all_lines_writer_ids, dtype=np.int16),
        }

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
