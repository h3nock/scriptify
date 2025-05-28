from email.mime import base
import re
import os
import argparse 
import numpy as np
from typing import Union
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils.paths import RunPaths

def parse_log_for_steps_and_losses(log_file_path):
    """
    Parses epoch completion logs to get training/validation losses and calculates the cumulative training steps at the end of each of these logged epochs"""
    steps_at_epoch_end = []
    train_losses_epoch = []
    val_losses_epoch = []
    
    steps_per_epoch = None
    
    # find steps per epoch 
    steps_per_epoch_pattern = re.compile(r"Starting training for \d+ epochs \((\d+) steps per epoch\)")
    
    # regex for completed epoch logs 
    epoch_completed_pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+completed.*?Train Loss:\s*(?P<train_loss>[-]?[\d.]+).*?Val Loss:\s*(?P<val_loss>[-]?[\d.]+)",
        re.IGNORECASE
    )

    if not os.path.exists(log_file_path):
        print(f"Error: Log file '{log_file_path}' not found.")
        return [], [], []

    # try to find steps_per_epoch from the log file
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match_spe = steps_per_epoch_pattern.search(line)
                if match_spe:
                    steps_per_epoch = int(match_spe.group(1))
                    break 
            if steps_per_epoch is None:
                print("Could not find 'steps per epoch' in the log file ('Starting training for ... (X steps per epoch)').")
                return [], [], []
    except Exception as e:
        print(f"Error reading log file to find 'steps per epoch': {e}")
        return [], [], []

    # parse the epoch completion lines and calculate cumulative steps
    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                match_ec = epoch_completed_pattern.search(line)
                if match_ec:
                    try:
                        epoch = int(match_ec.group("epoch")) # 0-indexed epoch number
                        train_loss = float(match_ec.group("train_loss"))
                        val_loss = float(match_ec.group("val_loss"))
                        
                        current_total_steps = (epoch + 1) * steps_per_epoch
                        
                        steps_at_epoch_end.append(current_total_steps)
                        train_losses_epoch.append(train_loss)
                        val_losses_epoch.append(val_loss)
                    except ValueError as e:
                        print(f"couldn't parse line {line_num}: {line.strip()} - {e}")
                    except IndexError:
                        print(f"regex matched but could not find all groups on line {line_num}: {line.strip()}")
    except Exception as e:
        print(f"unexpected error occurred while reading the log file for epoch data: {e}")
        return [], [], []
                        
    return steps_at_epoch_end, train_losses_epoch, val_losses_epoch

def plot_losses_vs_steps(steps_data, train_losses, val_losses, output_file_path="steps_loss_plot_32.png"):
    """
    Plots the training and validation losses against training steps and saves the plot"""
    if not steps_data:
        print("No data found to plot, check your log file format")
        return

    plt.figure(figsize=(14, 8))
    
    plt.plot(steps_data, train_losses, linestyle='-', linewidth=2, label='Training Loss')
    plt.plot(steps_data, val_losses, linestyle='--', linewidth=2, label='Validation Loss')
            
    plt.title('Training and Validation Loss vs. Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)
    
    min_step = min(steps_data) if steps_data else 0
    max_step = max(steps_data) if steps_data else 1
    # add a small padding to xlim
    padding = (max_step - min_step) * 0.01 if (max_step - min_step) > 0 else 1
    plt.xlim(min_step - padding, max_step + padding)

    num_ticks_desired = 10
    if len(steps_data) > num_ticks_desired:
        tick_indices = np.linspace(0, len(steps_data) - 1, num_ticks_desired, dtype=int)
        x_ticks = [steps_data[i] for i in tick_indices]
        if steps_data[0] not in x_ticks: 
             x_ticks[0] = steps_data[0]
        if steps_data[-1] not in x_ticks: 
             x_ticks[-1] = steps_data[-1]
        x_ticks = sorted(list(set(x_ticks))) 
        plt.xticks(x_ticks, rotation=30, ha='right')
    elif steps_data:
        plt.xticks(steps_data, rotation=30, ha='right')

    plt.tight_layout(pad=1.5)
    
    try:
        plt.savefig(output_file_path, dpi=200, bbox_inches='tight')
        print(f"Plot saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving plot to {output_file_path}: {e}")
    
    plt.close()

def plot_loss(log_file_path, output_file_path):
    parsed_steps, parsed_train_losses, parsed_val_losses = parse_log_for_steps_and_losses(log_file_path=log_file_path)
    plot_losses_vs_steps(parsed_steps, parsed_train_losses, parsed_val_losses, output_file_path = output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name", 
        type=str, 
        default=None, 
        help="Specific run name to plot"
    )
    
    parser.add_argument(
        "--base_outputs_dir", 
        type=str, 
        default="outputs", 
        help="Base outputs directory"
    )
    
    args = parser.parse_args() 
    
    if args.run_name:
        run_path = RunPaths(run_name=args.run_name,base_outputs_dir=args.base_outputs_dir) 
    else:
        run_paths = RunPaths.find_latest_run(base_outputs_dir=args.base_outputs_dir)
        
    if run_paths is None:
        print("No runs found. Please specify --run-name")
        exit(1) 
    
    output_file_path = run_paths.loss_plot 
    output_file_path.parent.mkdir(parents=True, exist_ok=True) 

    log_file_path = run_paths.training_log 
    
    if not log_file_path.exists():
        print(f"Error: Log file not found at {log_file_path}")
        exit(1) 
    
    print(f"Using run: {run_paths.run_name}")
    print(f"Reading from: {log_file_path}")
    print(f"Saving to: {output_file_path}")

    plot_loss(log_file_path, output_file_path)