import os

import torch

import torchvision.transforms as VT
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from dataset import id2label

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

batch_size = 90
# 默认超参最好的版本
# pth_name = "checkpoints/2024-04-01-01_41_45-20-76.79-0.0218.pth"

# 无增广
pth_name = "checkpoints/2024-04-01-04_38_40-20-81.37-0.0204.pth"
# 100 epoch
# pth_name = "checkpoints/2024-04-01-04_43_09-100-74.62-0.0060.pth"

checkpoint = torch.load(pth_name)

model = checkpoint["model"]
model.to(device)
model.eval()
# model.no_
print(f"Loaded {pth_name}")


# class Testset(torch.utils.DataSet)

class Testset(Dataset):
    def __init__(self):
        self.images = [(name[:-4], f"./dataset/test/{name}") for name in os.listdir("./dataset/test")]
        self.transform = VT.Compose([
            # VT.Resize(40),
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


test_loader = DataLoader(Testset(), shuffle=False, batch_size=batch_size)

with open("无增广submission.csv", "w") as f:
    f.write("id,label")
    for ids, images in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        preds = []
        with torch.no_grad():
            outputs = model(images)
        preds.extend(outputs.argmax(dim=1).type(torch.int32).cpu().numpy())
        for i, prediction in enumerate(preds):
            label = id2label(prediction)
            # print(f"\n{ids[i]},{label}")
            f.write(f"\n{ids[i]},{label}")
