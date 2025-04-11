import matplotlib.pyplot as plt

def plot_stroke_seq(stroke_segments, title="Handwriting Stroke Visualization"):
    """
    Plot the handwriting stroke sequence from a list of stroke segments.
    
    Each stroke segment is a list of tuples (x, y).
    """
    plt.figure(figsize=(8, 6))
    for segment in stroke_segments:
        x_vals, y_vals = [], [] 
        for x,y,_ in segment:
            x_vals.append(x) 
            y_vals.append(y) 
        plt.plot(x_vals, y_vals,'k')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()  # to match the whiteboard's coordinate convention
    plt.legend()
    plt.show()