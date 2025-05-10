import os
import numpy as np
import matplotlib.pyplot as plt

from src.data.preprocessing import offsets_to_stroke_coords

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

def split_into_stroke_segments(segments):
    """
    Converts a flat numpy array of shape (n, 3) where each row is (x, y, flag)
    into a list of lists of (x, y) tuples. A flag value of 1 marks the end of a stroke segment.
    """
    strokes = []
    current_segment = []
    for line_segs in segments: 
        for x, y, flag in line_segs:
            current_segment.append((x, y))
            if flag == 1:
                strokes.append(current_segment)
                current_segment = []
    
        # In case the last segment doesn't end with a flag, add it as well
        if current_segment:
            strokes.append(current_segment)
        
    return strokes

def plot_stroke_seq(stroke_segments, title="Handwriting Stroke Visualization"):
    """
    Plot the handwriting stroke sequence from a list of stroke segments.
    
    Each stroke segment is a list of tuples (x, y).
    """
    os.makedirs(OUTPUT_DIR, exist_ok = True) 
    stroke_segments = split_into_stroke_segments(stroke_segments)
    plt.figure(figsize=(8, 6))
    for segment in stroke_segments:
        x_vals, y_vals = [], [] 
        for x,y in segment:
            x_vals.append(x) 
            y_vals.append(y) 
        plt.plot(x_vals, y_vals,'k')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # save the plot 
    filename = title.replace(' ', '_').replace(":", "").replace("'", "") 
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.png")
    plt.savefig(filepath, dpi=100)
    plt.close()
    
    print(f"View the plot at: {os.path.abspath(filepath)}")

def plot_offset_strokes(offset_strokes, title="Offset Stroke Visualization"):
    """
    Visualize stroke sequences in their offset representation.
    """
    if isinstance(offset_strokes, np.ndarray) and offset_strokes.ndim == 2:
        offset_strokes = [offset_strokes] 
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    
    for i, stroke in enumerate(offset_strokes):
        dx_vals = [point[0] for point in stroke]
        dy_vals = [point[1] for point in stroke]
        pen_states = [point[2] for point in stroke]
        
        plt.quiver(
            range(len(dx_vals)), 
            [0] * len(dy_vals), 
            dx_vals, 
            dy_vals, 
            scale=1, 
            scale_units='xy', 
            angles='xy',
            width=0.004,
            color='blue',
            alpha=0.7
        )
        
        # mark pen up points with red dots
        pen_up_indices = [i for i, flag in enumerate(pen_states) if flag == 1]
        if pen_up_indices:
            plt.scatter([pen_up_indices], [0] * len(pen_up_indices), color='red', s=50, label='Pen Up')
    
    plt.title("Raw Offset Visualization (dx, dy)")
    plt.xlabel("Sequence Index")
    plt.ylabel("Offset Value")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    
    # convert offsets to absolute coordinates for comparison
    abs_coords = []
    for stroke in offset_strokes:
        curr_x, curr_y = 0, 0
        stroke_coords = []
        
        for dx, dy, pen in stroke:
            curr_x += dx
            curr_y += dy
            stroke_coords.append((curr_x, curr_y, pen))
        
        abs_coords.append(stroke_coords)
    
    stroke_segments = split_into_stroke_segments(abs_coords)
    
    for segment in stroke_segments:
        x_vals, y_vals = zip(*segment) if segment else ([], [])
        plt.plot(x_vals, y_vals, 'k-')
    
    plt.title("Reconstructed Absolute Coordinates")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    # save the plot
    filename = title.replace(' ', '_').replace(":", "").replace("'", "")
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.png")
    plt.savefig(filepath, dpi=100)
    plt.close()
    
    print(f"View the offset visualization at: {os.path.abspath(filepath)}")

def plot_stroke(stroke, title = "Stroke"):
    plt.figure(figsize=(8,6))
    x_vals, y_vals = [], []
    for x,y, _ in stroke:
        x_vals.append(x) 
        y_vals.append(y) 
    plt.plot(x_vals, y_vals, 'k') 
    plt.title(title) 
    plt.xlabel("X") 
    plt.ylabel("Y") 
    plt.legend() 
    plt.show() 