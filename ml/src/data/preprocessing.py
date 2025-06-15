from typing import Literal
import numpy as np
from scipy.signal import medfilt, savgol_filter

SmoothMethod = Literal['median', 'savgol']

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

def normalize(offsets):  
    """  
    normalizes strokes to median unit norm  
    """ 
    esp = 1e-6

    offsets = offsets.astype(np.float32) 
    xy_offsets = offsets[:,:2] 
    norms = np.linalg.norm(xy_offsets, axis=1) 

    median_norm = np.median(norms)
    if median_norm > esp:
        offsets[:,:2] /= median_norm 
    return offsets

def deskew_line(coords: np.ndarray) -> np.ndarray:
    """
    Rotate a full‑line (x,y,eos) array so that its main axis is straight horizontal line.
    """
    if coords.shape[0] < 2:
        return coords.copy()

    xy  = coords[:, :2].astype(np.float32)
    ctr = xy.mean(axis=0, keepdims=True)         # line centroid
    centered = xy - ctr

    # principal component of the centered cloud
    cov      = np.cov(centered.T)
    _, vecs  = np.linalg.eigh(cov)
    principal = vecs[:, 1]                       # largest eigen vector

    # force it to point rightwards to avoid 180 degree flips
    if principal[0] < 0:
        principal = -principal

    angle = np.arctan2(principal[1], principal[0])  # radians to horizontal

    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    rotated = centered @ R.T + ctr                 # rotate about centroid
    return np.concatenate([rotated, coords[:, 2:3]], axis=1)

def has_outlier(offsets: np.ndarray, threshold: float) -> bool:
    """check if any consecutive points have Euclidean distance > threshold"""
    deltas = offsets[1:,:2] - offsets[:-1,:2] 
    mags = np.linalg.norm(deltas, axis=1)

    return bool(np.any(mags > threshold)) 

def smooth_strokes(strokes: np.ndarray, method: SmoothMethod = 'savgol',*,
                   kernel_size: int = 5, window_length: int = 5, 
                   polyorder: int = 2) -> np.ndarray:
    """
    Smoothes stroke offset. 
    - method: 'median' or 'savgol' 
    - window_length: is kernel_size for median and window_length for savgol 
    - polyorder: polynomial order for savgol 
    """
    if method == 'savgol' and window_length % 2 == 0:
        window_length += 1

    # split at pen‑ups (eos==1)
    cuts = np.where(strokes[:, 2] == 1)[0] + 1
    segments = np.split(strokes, cuts)

    out_segments = []
    for seg in segments:
        if seg.size == 0:
            continue
        x, y = seg[:, 0].copy(), seg[:, 1].copy()

        if method == 'median':
            k = min(window_length, len(x))
            if k % 2 == 0:
                k -= 1
            x_s = medfilt(x, kernel_size=k)
            y_s = medfilt(y, kernel_size=k)
        else:  # savgol
            if len(x) < window_length:
                x_s, y_s = x, y
            else:
                x_s = savgol_filter(x, window_length, polyorder, mode='interp')
                y_s = savgol_filter(y, window_length, polyorder, mode='interp')

        seg[:, 0] = x_s
        seg[:, 1] = y_s
        out_segments.append(seg)

    # re‑concatenate all stroke segments
    return np.vstack(out_segments) 