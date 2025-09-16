# finetune_linear.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models import ClipEmbedder
from preprocess import preprocess_full_for_clip

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, embedder):
        self.image_paths = image_paths
        self.labels = labels
        self.embedder = embedder

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = int(self.labels[idx])
        pil_img = preprocess_full_for_clip(img_path)
        embedding = self.embedder.encode_images([pil_img])  # (1, d) float32
        x = torch.from_numpy(embedding[0])                  # (d,)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

def finetune_linear_classifier(image_paths, labels, embedder, output_dir="ckpts", epochs=10, batch_size=32, lr=1e-3):
    dataset = CustomDataset(image_paths, labels, embedder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 动态确定输入维度
    with torch.no_grad():
        x0, _ = dataset[0]
        in_dim = x0.numel()

    num_classes = int(len(set(labels)))
    linear_classifier = nn.Linear(in_dim, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    linear_classifier = linear_classifier.to(device)

    optimizer = optim.Adam(linear_classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    linear_classifier.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0
        for xs, ys in dataloader:
            xs = xs.to(device)
            ys = ys.to(device)

            optimizer.zero_grad()
            outputs = linear_classifier(xs)
            loss = criterion(outputs, ys)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xs.size(0)
            total_samples += xs.size(0)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/total_samples:.6f}")

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "linear_classifier.pth")
    torch.save(linear_classifier.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # 示例：请替换为你真实的数据与标签
    image_paths = ["data/gallery/authorA/xxx.jpg", "data/gallery/authorB/yyy.jpg"]
    labels = [0, 1]

    embedder = ClipEmbedder()
    finetune_linear_classifier(image_paths, labels, embedder)
