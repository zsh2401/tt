#!/usr/bin/env python
import os

import torch

import torchvision.transforms as VT
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from dataset import id2label,classes

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

batch_size =90
# 默认超参最好的版本
pth_name = "checkpoints/2024-04-01-00_26_38-4-64.35-0.0051.pth"
checkpoint = torch.load(pth_name)

model = checkpoint["model"]
model.to(device)
model.eval()
print(f"Loaded {pth_name}")


# class Testset(torch.utils.DataSet)

class Testset(Dataset):
    def __init__(self):
        self.images = [(name[:-4], f"./dataset/test/{name}") for name in os.listdir("./dataset/test")]
        self.transform = VT.Compose([
            VT.Resize(256),
    # 从图像中心裁切224x224大小的图片
            VT.CenterCrop(224),
            VT.ToTensor(),
            VT.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010])])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        id, path = self.images[index]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return id, image


test_loader = DataLoader(Testset(), shuffle=True, batch_size=batch_size)

with open("submission.csv", "w") as f:
    f.write('id,' + ','.join(classes))
    for ids, images in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        
        prob = torch.softmax(outputs,dim=1).cpu().detach().numpy()
        for i, p in enumerate(prob):
            # print(p)
            f.write(f"\n{ids[i]},{','.join([str(num) for num in p])}")
