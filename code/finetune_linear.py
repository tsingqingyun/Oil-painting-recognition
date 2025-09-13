import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models import ClipEmbedder
from preprocess import preprocess_for_painting
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, embedder):
        self.image_paths = image_paths
        self.labels = labels
        self.embedder = embedder

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img_tensor = preprocess_for_painting(img_path)
        embedding = self.embedder.encode_images([img_tensor])
        return torch.tensor(embedding[0]), torch.tensor(label)

def finetune_linear_classifier(image_paths, labels, embedder, output_dir="ckpts"):
    dataset = CustomDataset(image_paths, labels, embedder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 定义线性分类器
    linear_classifier = nn.Linear(embedder.model.context_length, len(set(labels)))  # 类别数
    optimizer = optim.Adam(linear_classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练
    linear_classifier.train()
    for epoch in range(10):  # 训练10轮
        total_loss = 0
        for imgs, lbls in dataloader:
            optimizer.zero_grad()
            outputs = linear_classifier(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
    
    # 保存模型
    torch.save(linear_classifier.state_dict(), f"{output_dir}/linear_classifier.pth")
    print(f"Model saved to {output_dir}/linear_classifier.pth")

# 假设你有 metadata.csv 文件和图片路径，你可以加载标签并进行微调
if __name__ == "__main__":
    # 加载标签和图片路径
    image_paths = ["data/gallery/authorA/xxx.jpg", "data/gallery/authorB/yyy.jpg"]  # 示例
    labels = [0, 1]  # 示例标签，实际要从 metadata.csv 中加载

    embedder = ClipEmbedder()
    finetune_linear_classifier(image_paths, labels, embedder)
