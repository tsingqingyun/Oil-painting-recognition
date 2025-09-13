import faiss
import numpy as np
from models import ClipEmbedder
from preprocess import preprocess_for_painting

def search_image(query_img_path, index_dir, top_k=5):
    # 载入 FAISS 索引
    index = faiss.read_index(f"{index_dir}/faiss.index")
    embeddings = np.load(f"{index_dir}/embeddings.npy")

    # 载入模型并对查询图像进行向量化
    embedder = ClipEmbedder()
    query_img = preprocess_for_painting(query_img_path)
    query_emb = embedder.encode_images([query_img])

    # 使用 FAISS 检索最相似的图像
    distances, indices = index.search(query_emb, top_k)
    print("Top-k Similar Images:")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. Image ID: {idx}, Similarity: {distances[0][i]:.4f}")

if __name__ == "__main__":
    query_image_path = "D:\\GitHub\\whole procedure\\data\\queries\\02 (2).jpg"  # 查询图像路径
    index_dir = "index"  # FAISS 索引存放目录
    search_image(query_image_path, index_dir, top_k=5)
