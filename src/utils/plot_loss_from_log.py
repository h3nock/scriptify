import re
import os
import matplotlib.pyplot as plt

script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
LOG_FILE_PATH = os.path.join(script_dir, 'training_log.txt')
LOG_FILE_PATH = os.path.normpath(LOG_FILE_PATH)
OUTPUT_FILE_PATH = os.path.join(script_dir, 'outputs', 'epoch_loss_plot.png')


def parse_epoch_losses_from_log(log_file_path):
    """
    Parses epoch level training and validation loss values from the specified training log file focusing on epoch completion lines.
    """
    epochs = []
    train_losses_epoch = []
    val_losses_epoch = []

    epoch_completed_pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+completed.*?Train Loss:\s*(?P<train_loss>[\d.]+).*?Val Loss:\s*(?P<val_loss>[\d.]+)",
        re.IGNORECASE
    )

    if not os.path.exists(log_file_path):
        print(f"Error: Log file '{log_file_path}' not found.")
        return [],[],[]
    
    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                match = epoch_completed_pattern.search(line)
                if match:
                    try:
                        epoch = int(match.group("epoch"))
                        train_loss = float(match.group("train_loss"))
                        val_loss = float(match.group("val_loss"))
                        
                        epochs.append(epoch)
                        train_losses_epoch.append(train_loss)
                        val_losses_epoch.append(val_loss)
                    except ValueError as e:
                        print(f"Warning: couldn't parse float from matched groups on line {line_num}: {line.strip()} - {e}")
                    except IndexError:
                        print(f"Warning: regex matched but could not find all groups on line {line_num}: {line.strip()}")
    except FileNotFoundError:
        print(f"Log file '{log_file_path}' not found during read attempt.")
        return [], [], []
    except Exception as e:
        print(f"Unexpected error occurred while reading the log file: {e}")
        return [], [], []
                        
    return epochs, train_losses_epoch, val_losses_epoch

def plot_epoch_losses(epochs, train_losses, val_losses, output_file_path="epoch_loss_plot.png"):
    """
    Plots the epoch level training and validation losses and saves the plot to a file
    """
    if not epochs:
        print("No epoch data found to plot. Please check your log file format and the regex pattern.")
        return

    plt.figure(figsize=(12, 7))
    
    plt.plot(epochs, train_losses, marker='o', linestyle='-', label='Epoch Training Loss')
    plt.plot(epochs, val_losses, marker='x', linestyle='--', label='Epoch Validation Loss')
            
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    
    if len(epochs) > 20:
        tick_spacing = max(1, len(epochs) // 10) 
        plt.xticks(epochs[::tick_spacing])
    elif epochs:
        plt.xticks(epochs)

    plt.tight_layout()
    
    try:
        plt.savefig(output_file_path)
        print(f"Plot saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving plot to {output_file_path}: {e}")
    
    plt.close() 

if __name__ == "__main__":
    log_file_path = LOG_FILE_PATH 
    output_file_path = OUTPUT_FILE_PATH
    parsed_epochs, parsed_train_losses, parsed_val_losses = parse_epoch_losses_from_log(log_file_path=log_file_path)
    
    if parsed_epochs: # check if any epoch data was successfully parsed
        plot_epoch_losses(parsed_epochs, parsed_train_losses, parsed_val_losses, output_file_path = output_file_path)
    else:
        print(f"No data was extracted from '{log_file_path}' or the file was not found.")
