import os 
import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader

class ProcessedHandwritingDataset(Dataset):
    def __init__(self, processed_dir):
        self.strokes = np.load(os.path.join(processed_dir, "strokes.npy"))
        self.strokes_len = np.load(os.path.join(processed_dir, "strokes_len.npy"))
        self.chars = np.load(os.path.join(processed_dir, "chars.npy"))
        self.chars_len = np.load(os.path.join(processed_dir, "chars_len.npy"))
        self.writer_ids = np.load(os.path.join(processed_dir, "writer_ids.npy"))
        self.mask = np.load(os.path.join(processed_dir, "mask.npy"))
    
    def __len__(self):
        return self.strokes.shape[0] 
    
    def __getitem__(self, idx):
        return {
            'stroke': torch.tensor(self.strokes[idx], dtype=torch.float32),
            'stroke_len': torch.tensor(self.strokes_len[idx], dtype=torch.int),
            'chars': torch.tensor(self.chars[idx], dtype=torch.int),
            'chars_len': torch.tensor(self.chars_len[idx], dtype = torch.int), 
            'writer_id': torch.tensor(self.writer_ids[idx], dtype= torch.int), 
            'mask': torch.tensor(self.mask[idx], dtype = torch.float32) 
        }

if __name__ == '__main__':
    processed_dir = 'data/processed' 
    dataset = ProcessedHandwritingDataset(processed_dir=processed_dir) 
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

    for batch in dataloader:
        print("Strokes batch shape:", batch['stroke'].shape)
        print("Text batch shape:", batch['chars'].shape)
        print("Mask shape:", batch['mask'].shape)
        break