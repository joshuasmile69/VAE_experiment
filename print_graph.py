import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--log_path', type=str)
parser.add_argument('--out_path', type=str)
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

with open(out_path, 'w') as f:
    f.write("epoch,Train,Dev\n")
    for epoch in range(len(train_loss_list)):
        f.write(f"{epoch},{train_loss_list[epoch]},{dev_loss_list[epoch]}\n")

# 그래프 그리기
epochs = range(len(train_loss_list))
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss_list, label='Train Total Loss')
plt.plot(epochs, dev_loss_list, label='Dev Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training and Development Total Loss')
plt.legend()
plt.savefig(out_path.replace('.csv', '.png'))
plt.show()
