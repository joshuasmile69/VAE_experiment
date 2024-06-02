import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--log_path', type=str, required=True, help='Path to the log file')
parser.add_argument('--out_path', type=str, required=True, help='Output path for the CSV and PNG files')
args = parser.parse_args()

log_path = args.log_path
out_path = args.out_path

train_loss_list = []
dev_loss_list = []

with open(log_path, 'r') as f:
    next(f)  # Skip the header line
    for line in f:
        epoch, mtype, total_loss = line.strip().split(',')
        total_loss = float(total_loss)
        
        if mtype == "Train":
            train_loss_list.append(total_loss)
        elif mtype == "DEV":
            dev_loss_list.append(total_loss)

# 길이 맞추기
min_length = min(len(train_loss_list), len(dev_loss_list))
train_loss_list = train_loss_list[:min_length]
dev_loss_list = dev_loss_list[:min_length]

# Save to CSV
csv_path = out_path.replace('.png', '.csv')
with open(csv_path, 'w') as f:
    f.write("epoch,Train,Dev\n")
    for epoch in range(min_length):
        f.write(f"{epoch},{train_loss_list[epoch]},{dev_loss_list[epoch]}\n")

# Plot the graph
epochs = range(min_length)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss_list, label='Train Total Loss')
plt.plot(epochs, dev_loss_list, label='Dev Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training and Development Total Loss')
plt.legend()
plt.savefig(out_path)
plt.show()



