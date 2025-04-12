import numpy as np

def stroke_coords_to_offsets(stroke_coords):
    """
    Convert absolute coordinates to offsets
    
    Parameters:
        stroke_coords: numpy array of shape (n, 3) where each row is (x, y, eos)
    
    Returns:
        offsets: numpy array where the first row is [0, 0, 1] and subsequent rows are the differences in (x, y)
                 along with the corresponding eos flag
    """
    diffs = stroke_coords[1:, :2] - stroke_coords[:-1, :2]
    eos_flags = stroke_coords[1:, 2:3]
    offsets = np.concatenate([diffs, eos_flags], axis=1)
    start_offset = np.array([[0, 0, 1]])
    return np.concatenate([start_offset, offsets], axis=0)

def offsets_to_stroke_coords(offsets):
    """
    Convert offsets back to absolute coordinates
    
    Parameters:
        offsets: numpy array of shape (n, 3), where each row represents (dx, dy, eos)
    
    Returns:
        stroke_coords: numpy array of shape (n, 3) where the x and y coordinates are reconstructed by cumulative sum
    """
    stroke_coords_xy = np.cumsum(offsets[:, :2], axis=0)
    return np.concatenate([stroke_coords_xy, offsets[:, 2:3]], axis=1)

def pad_line(line_stroke, max_points=2048, pad_value=0):
    """
    Pad a stroke (or an entire line of stroke coordinates) to a fixed shape (max_points, 3)
    
    Parameters:
        line_stroke: numpy array of shape (n, 3)
        max_points: the fixed number of points desired
        pad_value: value to use for padding
    
    Returns:
        padded: numpy array of shape (max_points, 3)
    """
    n, d = line_stroke.shape
    if n >= max_points:
        return line_stroke[:max_points]
    padded = np.full((max_points, d), pad_value, dtype=line_stroke.dtype)
    padded[:n, :] = line_stroke
    return padded

def create_mask(line_stroke, max_points=2048):
    """
    Create a binary mask for a padded stroke
    
    Parameters:
        line_stroke: numpy array of shape (n, 3) (before padding or after padding)
        max_points: total desired length after padding
    
    Returns:
        mask: numpy array of shape (max_points,) with 1 for valid points and 0 for padding
    """
    n = line_stroke.shape[0]
    mask = np.zeros(max_points, dtype=np.float32)
    mask[:min(n, max_points)] = 1
    return mask