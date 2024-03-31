from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TV
from PIL import Image


class CifarDataset(Dataset):
    def __init__(self, start_percent, end_percent) -> None:
        super().__init__()
        self.data = split_dataset(start_percent, end_percent)
        self.transform = TV.Compose([
            # 在高度和宽度上将图像放大到40像素的正方形
            TV.Resize(40),
            # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
            # 生成一个面积为原始图像面积0.64～1倍的小正方形，
            # 然后将其缩放为高度和宽度均为32像素的正方形
            # 不使用增广进行训练
            # TV.RandomResizedCrop(32, scale=(0.64, 1.0),
            #                      ratio=(1.0, 1.0)),
            # TV.RandomHorizontalFlip(),
            TV.ToTensor(),
            # 标准化图像的每个通道
            TV.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label, path = self.data[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        result = label2id(label), image
        return result


def split_dataset(start_percent, end_percent):
    start_index = int(len(original_csv) * start_percent)  # 计算 60% 的位置
    end_index = int(len(original_csv) * end_percent)  # 计算 80% 的位置
    data = []
    for i in range(start_index, end_index):
        data.append((original_csv[str(i + 1)], f"./dataset/train/{i + 1}.png"))
    return data


def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))


original_csv = read_csv_labels("./dataset/trainLabels.csv")
classes = [clazz for clazz in set(original_csv.values())]
classes.sort()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
numeric_labels = encoder.fit_transform(classes)


def id2label(id):
    # return numeric_labels
    return classes[id]


def label2id(label):
    for i in range(len(classes)):
        if label == classes[i]:
            return numeric_labels[i]

    raise "Label not found"
