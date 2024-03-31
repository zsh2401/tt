#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CifarDataset
from torch.utils.data import Dataset, DataLoader
from model import Cifar18
import datetime

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

# 课后题的超参
lr_period = 50
lr_decay = 0.1
lr = 0.1
num_epochs = 100
batch_size = 128
wd = 5e-4

# 正常实验的超参
lr_period = 4
lr_decay = 0.9
wd = 5e-4
lr = 2e-4
num_epochs = 20
batch_size = 32


train_loader = DataLoader(CifarDataset(0, 0.8), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CifarDataset(0.8, 1), batch_size=batch_size, shuffle=True)

model = Cifar18()
optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=wd)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
criterion = torch.nn.CrossEntropyLoss().to(device)
# criterion.sate
model = torch.nn.DataParallel(model).to(device)

print(f"runnin on {device}")
train_losses = []
val_losses = []
val_accuracies = []
for epoch in range(1,num_epochs + 1):
    train_loss = 0
    val_loss = 0
    val_acc = 0
    model.train()

    for i, (labels, images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")):
        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

    with torch.no_grad():
        correct = 0
        total = 0
        val_loss = 0
        for i, (labels, images) in enumerate(tqdm(val_loader, desc=f"Validation")):
            labels = labels.to(device)
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取每个样本的最大logit值索引作为预测结果

            val_loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_acc = correct / total
        # print(f'***Validation Set Accuracy: {accuracy * 100:.2f}% ***')
        val_loss /= len(val_loader.dataset)

    print(f"\nTrain loss {train_loss}, Val loss {val_loss}, Accuracy {val_acc * 100:.2f}%")
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    file_name = f"checkpoints/{datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}-{epoch}-{val_acc * 100:.2f}-{val_loss:.4f}.pth"
    if epoch % 5 != 0:
        continue
    torch.save({
        "model": model,
        "optimizer": optimizer.state_dict(),
        "schedular": scheduler.state_dict(),
        "criterion": criterion.state_dict(),
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "train_losses": train_losses
    },file_name)
    print("Saved as " + file_name)
