import matplotlib.pyplot as plt

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
    plt.gca().invert_yaxis()  # To match the whiteboard's coordinate convention
    plt.legend()
    plt.show()

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
    plt.gca().invert_yaxis() 
    plt.legend() 
    plt.show() 