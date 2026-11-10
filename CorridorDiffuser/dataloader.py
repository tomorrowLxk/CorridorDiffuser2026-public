import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split


class SafetyCorridorDataset(Dataset):
    def __init__(self, map_dir, target_dir, map_size=256, train=False):
        """
        map_dir: 地图路径
        target_dir: 安全速度栅格路径
        map_size: 输入大小
        max_speed: 对应最大速度
        train: 是否训练模式（可选数据增强）
        """
        super().__init__()
        self.map_dir = map_dir
        self.target_dir = target_dir
        self.map_paths = sorted(glob.glob(os.path.join(map_dir, "map_*.png")))
        self.map_size = map_size
        self.train = train
        self.to_tensor = T.ToTensor()  # 转换为 [0,1]
        self.idxs = []

        # 构建 map 对应的 target 列表
        self.map_targets = []
        for map_path in self.map_paths:
            idx = os.path.splitext(os.path.basename(map_path))[0].split("_")[1]
            target_paths = sorted(glob.glob(os.path.join(target_dir, f"target_{idx}_*.png")))
            for tpath in target_paths:
                self.map_targets.append((map_path, tpath))
                self.idxs.append(idx)

    def __len__(self):
        return len(self.map_targets)

    def __getitem__(self, idx):
        map_path, target_path = self.map_targets[idx]

        img = Image.open(map_path).convert('RGB')
        img_np = np.array(img, dtype=np.float32)
        map_img = torch.from_numpy(np.transpose(img_np, (2, 0, 1))) * 2.0 - 1.0
        target_img = self.to_tensor(Image.open(target_path).convert("L")) * 2.0 - 1.0

        if self.train:
            if random.random() < 0.5:
                map_img = TF.hflip(map_img)
                target_img = TF.hflip(target_img)
            if random.random() < 0.5:
                map_img = TF.vflip(map_img)
                target_img = TF.vflip(target_img)

        return {
            "Y_gt": target_img.float(),  # 速度栅格
            "map": map_img.float(),  # 地图
            "idx": float(self.idxs[idx]),
        }


def generate_dataloader(dataset, config):
    total_len = len(dataset)
    train_len = int(total_len * config.TRAIN_RATIO)
    val_len = int(total_len * config.VAL_RATIO)
    test_len = total_len - train_len - val_len
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # torch.cuda.manual_seed_all(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len], generator=generator)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader
