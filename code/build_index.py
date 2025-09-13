import faiss
import numpy as np
import os
from tqdm import tqdm
from preprocess import preprocess_for_painting
from models import ClipEmbedder

def list_images(root):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    files = []
    ids = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(exts):
                files.append(os.path.join(dp, f))
                ids.append(os.path.relpath(os.path.join(dp, f), root))
    return files, ids
def build_faiss_index(image_paths, output_dir, model_name="ViT-H-14"):
    # 载入模型
    embedder = ClipEmbedder(model_name=model_name)
    
    # 读取所有图像并转换为 embedding
    embeddings = []
    for img_path in tqdm(image_paths):
        pil_img = preprocess_for_painting(img_path)  # 确保返回的是 PIL 图像
        emb = embedder.encode_images([pil_img])      # 预期是 PIL 格式传给 preprocess
        embeddings.append(emb)
    
    embeddings = np.vstack(embeddings).astype("float32")

    # 构建 FAISS 索引
    dim = embeddings.shape[1]  # 向量的维度
    index = faiss.IndexFlatL2(dim)  # 使用 L2 距离
    index.add(embeddings)  # 添加到索引中

    # 保存索引和 ID
    faiss.write_index(index, os.path.join(output_dir, "faiss.index"))
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    return embeddings

if __name__ == "__main__":
    gallery_dir = "data/gallery"  # 数据库图像文件夹
    output_dir = "index"  # FAISS 索引保存路径
    image_paths, _ = list_images(gallery_dir)
    build_faiss_index(image_paths, output_dir)
