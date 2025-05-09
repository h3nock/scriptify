import os 
import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader

from src.utils.stroke_viz import plot_offset_strokes

class ProcessedHandwritingDataset(Dataset):
    def __init__(self, processed_dir):
        self.strokes = np.load(os.path.join(processed_dir, "strokes.npy"))
        self.strokes_len = np.load(os.path.join(processed_dir, "strokes_len.npy"))
        self.chars = np.load(os.path.join(processed_dir, "chars.npy"))
        self.chars_len = np.load(os.path.join(processed_dir, "chars_len.npy"))
        self.writer_ids = np.load(os.path.join(processed_dir, "writer_ids.npy"))
        print(f"Loaded {self.strokes.shape[0]} samples.")
        plot_offset_strokes(self.strokes[5:6]) 

    @staticmethod
    def get_alphabet():
        return ['\x00',' ', '!', '"', '#', "'", '(', ')', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', 
                    '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
                    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 
                    'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 
                    'v', 'w', 'x', 'y', 'z']
    @staticmethod
    def get_alphabet_size():
        return len(ProcessedHandwritingDataset.get_alphabet()) 
    
    def __len__(self):
        return self.strokes.shape[0] 
    
    def __getitem__(self, idx):
        return {
            'stroke': torch.tensor(self.strokes[idx], dtype=torch.float32),
            'stroke_len': torch.tensor(self.strokes_len[idx], dtype=torch.int),
            'chars': torch.tensor(self.chars[idx], dtype=torch.long),
            'chars_len': torch.tensor(self.chars_len[idx], dtype = torch.int), 
            'writer_id': torch.tensor(self.writer_ids[idx], dtype= torch.int), 
        }

if __name__ == '__main__':
    processed_dir = 'data/processed' 
    dataset = ProcessedHandwritingDataset(processed_dir=processed_dir) 
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

    for batch in dataloader:
        print("Strokes batch shape:", batch['stroke'].shape)
        print("Text batch shape:", batch['chars'].shape)
        break