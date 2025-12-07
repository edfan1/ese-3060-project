import torch
import os
import glob

# Find the most recent log directory
log_dirs = glob.glob('logs/*')
if log_dirs:
    latest_log = max(log_dirs, key=os.path.getctime)
    log_path = os.path.join(latest_log, 'log.pt')
    
    # Load and view
    log = torch.load(log_path)
    print(f"Loaded from: {log_path}")
    print(f"Accuracies: {log['accs']}")
    print(f"Mean: {log['accs'].mean():.4f}")
    print(f"Std: {log['accs'].std():.4f}")
    print(f"Min: {log['accs'].min():.4f}")
    print(f"Max: {log['accs'].max():.4f}")