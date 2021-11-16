# step1:导入包
import os
import glob
import random
from PIL import Image
from torch.utils.data import  Dataset
import torchvision.transforms as transforms

# step2:定义Dataset
class myDataset(Dataset):
    def __init__(self, root = "", transform = None, model = "train"):
        self.mytransforms = transforms.Compose(transform)

        self.pathA = os.path.join(root, model, "A/*")
        self.pathB = os.path.join(root, model, "B/*")

        self.listA = glob.glob(self.pathA)
        self.listB = glob.glob(self.pathB)

    def __getitem__(self, i):
        img_pathA = self.listA[i % len(self.listA)]
        img_pathB = random.choice(self.listB)

        imgA = self.mytransforms(Image.open(img_pathA))
        imgB = self.mytransforms(Image.open(img_pathB))

        return {"A":imgA, "B":imgB}

    def __len__(self):
        lenA = len(self.listA)
        lenB = len(self.listB)
        return max(lenA, lenB)

