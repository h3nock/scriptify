import os
from pathlib import Path
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from src.data.preprocessing import offsets_to_stroke_coords


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

def plot_stroke_seq(stroke_segments,filepath: Union[str, Path], title: str = "Handwriting Stroke Visualization"):
    """
    Plot the handwriting stroke sequence from a list of stroke segments.
    
    Each stroke segment is a list of tuples (x, y).
    """
    stroke_segments = split_into_stroke_segments(stroke_segments)
    plt.figure(figsize=(10, 6))
    for segment in stroke_segments:
        x_vals, y_vals = [], [] 
        for x,y in segment:
            x_vals.append(x) 
            y_vals.append(y) 
        plt.plot(x_vals, y_vals,'k', linewidth = 2)
    plt.title(title, fontsize=20)
    plt.axis('equal') 
    plt.xticks([]) 
    plt.yticks([])
    plt.axis('off')
    # plt.legend()

    # save the plot 
    plt.savefig(filepath, dpi=200)
    plt.close()
    
    print(f"View the plot at: {os.path.abspath(filepath)}")

def plot_offset_strokes(offset_strokes_data: np.ndarray, filepath: Union[str, Path],title: str = "Offset Stroke Visualization", show_reconstructed_axes: bool = False, save_plot: bool = True, plot_only_text: bool = False):
    """
    Visualize stroke sequences in their offset representation.
    """
    try:
        filepath = Path(filepath) 
    except:
        print(f"file path must of a valid path string or path")
        return 
    
    if save_plot:
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True) 
        except Exception as e: 
            raise e
            
    
    dx_vals = offset_strokes_data[:, 0]
    dy_vals = offset_strokes_data[:, 1]
    pen_states = offset_strokes_data[:, 2]
    sequence_indices = np.arange(len(dx_vals))
    
    if plot_only_text:
        fig = plt.figure(figsize=(10,6))
        ax_text = fig.add_subplot(1,1,1) 
    else:
        fig = plt.figure(figsize=(12, 10))
    
        # plot 1 dx
        ax1 = fig.add_subplot(4, 1, 1)
        ax1.plot(sequence_indices, dx_vals, label='dx', color='dodgerblue', linewidth=1.5)
        ax1.set_ylabel('dx')
        ax1.set_title('Offset Components')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper right')

        # plot 2 dy
        ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
        ax2.plot(sequence_indices, dy_vals, label='dy', color='mediumseagreen', linewidth=1.5)
        ax2.set_ylabel('dy')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='upper right')

        # plot 3 pen_state
        ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
        ax3.step(sequence_indices, pen_states, label='pen_state (0=down, 1=up)', color='coral', where='post', linewidth=1.5)
        ax3.set_ylabel('Pen State')
        ax3.set_xlabel('Sequence Index')
        ax3.set_yticks([0, 1])
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend(loc='upper right')
    
        # plot 4 reconstructed Absolute Coordinates
        ax_text = fig.add_subplot(4, 1, 4)
    
    abs_coords_sample = []
    current_x, current_y = 0, 0
    for dx, dy, pen in offset_strokes_data:
        current_x += dx
        current_y += dy
        abs_coords_sample.append((current_x, current_y, pen))
            
    if abs_coords_sample:
        stroke_segments_to_plot = split_into_stroke_segments([abs_coords_sample])
        if not stroke_segments_to_plot:
            ax_text.text(0.5, 0.5, "No strokes to plot.", ha='center', va='center')
        else:
            for segment in stroke_segments_to_plot:
                if not segment or len(segment) < 2: continue
                x_plot, y_plot = zip(*segment)
                ax_text.plot(x_plot, y_plot, 'k-', linewidth=1.5)
    
    ax_text.set_title( "" if plot_only_text else "Reconstructed Absolute Coordinates")
    ax_text.axis('equal')

    if show_reconstructed_axes:
        ax_text.set_xlabel("X")
        ax_text.set_ylabel("Y")
        ax_text.grid(True, linestyle='--', alpha=0.6)
    else:
        ax_text.set_xticks([])
        ax_text.set_yticks([])
        ax_text.axis('off')
        
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.suptitle(title, fontsize=18, y=0.99)
    
    if save_plot:
        plt.savefig(filepath, dpi=200)
        print(f"Offset visualization saved to: {filepath}")
    
    plt.show()
    plt.close(fig) 
    
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