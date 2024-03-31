#!/usr/bin/env python
import torch
import matplotlib.pyplot as plt

pth = "./2024-04-01-01_41_45-20-76.79-0.0218.pth"

checkpoint = torch.load(pth)
train_losses = checkpoint["train_losses"]
val_losses = checkpoint["val_losses"]
val_accuracies = checkpoint["val_accuracies"]

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.close()  # 关闭图形，避免在下一个图中显示

# 绘制验证准确率
plt.figure(figsize=(10, 6))
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc.png')
plt.close()

