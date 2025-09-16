# build_index.py
import faiss
import numpy as np
import os
from tqdm import tqdm
from preprocess import preprocess_full_for_clip
from models import ClipEmbedder

def list_images(root):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    files, ids = [], []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(exts):
                full = os.path.join(dp, f)
                files.append(full)
                ids.append(os.path.relpath(full, root))  # 相对 gallery 根目录的路径
    return files, ids

def build_faiss_index(image_paths, ids, output_dir, model_name="ViT-H-14"):
    os.makedirs(output_dir, exist_ok=True)

    embedder = ClipEmbedder(model_name=model_name)
    embs = []
    for p in tqdm(image_paths, desc="Embedding gallery"):
        pil_img = preprocess_full_for_clip(p)
        emb = embedder.encode_images([pil_img])  # (1, d) float32/float64 -> 统一到 float32
        embs.append(emb)

    embs = np.vstack(embs).astype("float32")
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)

    faiss.write_index(index, os.path.join(output_dir, "faiss.index"))
    np.save(os.path.join(output_dir, "embeddings.npy"), embs)
    np.save(os.path.join(output_dir, "ids.npy"), np.array(ids))  # 新增：保存路径映射
    print(f"Saved index, embeddings and ids to: {output_dir}")

if __name__ == "__main__":
    gallery_dir = "data/gallery"
    output_dir = "index"
    image_paths, ids = list_images(gallery_dir)
    build_faiss_index(image_paths, ids, output_dir)
