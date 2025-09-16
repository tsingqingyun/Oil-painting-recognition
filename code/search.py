# search.py  —— 批量查询 + 打印命中文件路径
import argparse
import glob
import os
import faiss
import numpy as np
from models import ClipEmbedder
from preprocess import preprocess_full_for_clip

def list_query_images(query_dir):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
    paths = []
    for ptn in exts:
        paths.extend(glob.glob(os.path.join(query_dir, ptn)))
    return sorted(paths)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_dir", "-qd", type=str, default="data/queries", help="查询图像所在目录")
    ap.add_argument("--index", "-i", type=str, default="index", help="FAISS 索引目录")
    ap.add_argument("--topk", "-k", type=int, default=5, help="返回前 K 个结果")
    args = ap.parse_args()

    index_path = os.path.join(args.index, "faiss.index")
    ids_path = os.path.join(args.index, "ids.npy")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"缺少索引文件: {index_path}，请先运行 build_index.py")
    if not os.path.isfile(ids_path):
        raise FileNotFoundError(f"缺少 ids 映射: {ids_path}，请使用新的 build_index.py 重新构建索引（会生成 ids.npy）")

    index = faiss.read_index(index_path)
    gallery_ids = np.load(ids_path, allow_pickle=True)

    embedder = ClipEmbedder()

    queries = list_query_images(args.query_dir)
    if not queries:
        raise FileNotFoundError(f"在目录中未找到查询图片: {args.query_dir}")

    print(f"[info] 将对 {len(queries)} 张查询图执行检索（Top-{args.topk}）\n")

    for qi, qpath in enumerate(queries, 1):
        qimg = preprocess_full_for_clip(qpath)
        qemb = embedder.encode_images([qimg]).astype("float32")
        if qemb.ndim == 1:
            qemb = qemb[None, :]

        distances, indices = index.search(qemb, args.topk)

        print(f"=== Query {qi}: {qpath}")
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
            # 打印命中图片的相对路径（构建索引时保存的）
            hit_path_rel = str(gallery_ids[idx])
            print(f"{rank}. {hit_path_rel}    L2: {dist:.4f}")
        print()

if __name__ == "__main__":
    main()
