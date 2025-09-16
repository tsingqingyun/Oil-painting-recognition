# Oil-painting-recognition
---

# 油画图像识别与检索流程

本项目实现了一个基于 **CLIP** 的油画图像检索与分类系统，并结合 **OpenCV** 的完整预处理流程（透视矫正 + 去边框 + 增强），用于提升图像质量与特征一致性。

## 📦 环境准备

```bash
# 创建并激活虚拟环境（推荐）
conda create -n painting python=3.10 -y
conda activate painting

# 安装依赖
pip install torch torchvision torchaudio
pip install open-clip-torch faiss-cpu pillow opencv-python tqdm
```

> 如果有 GPU，建议安装对应 CUDA 版本的 `torch` 和 `faiss-gpu`。

---

## 📂 数据准备

项目目录结构建议如下：

```
project_root/
├── code/
│   ├── build_index.py
│   ├── search.py
│   ├── finetune_linear.py
│   ├── models.py
│   ├── preprocess.py
├── data/
│   ├── gallery/      # 图库（建立索引用）
│   │   ├── authorA/xxx.jpg
│   │   ├── authorB/yyy.png
│   │   └── ...
│   └── queries/      # 查询集（用来检索测试）
│       ├── q1.jpg
│       ├── q2.png
│       └── ...
├── index/            # 索引文件（自动生成）
└── ckpts/            # 微调保存的模型（自动生成）
```

* **data/gallery/**：存放图库图片，支持 `.jpg/.jpeg/.png/.bmp/.tif/.tiff/.webp`。
* **data/queries/**：存放要检索的查询图片。

---

## 🛠 模块说明

### 1. 预处理模块（`preprocess.py`）

* **`preprocess_full_for_clip(path)`**
  完整 OpenCV 流程：

  * CLAHE 对比度增强
  * Canny 边缘检测
  * 四边形透视校正（失败则自动退化为边框裁剪）
  * LAB-CLAHE 增强
  * 输出 `PIL.Image(RGB)`，供 CLIP 模型使用。

### 2. 向量提取（`models.py`）

* **`ClipEmbedder`** 封装了 `open_clip` 模型（默认 `ViT-H-14`）。
* 方法：

  * `encode_images(pil_imgs)` → 返回 `float32` numpy 向量。

### 3. 索引构建（`build_index.py`）

* 遍历 `data/gallery/`，逐图调用 `preprocess_full_for_clip`，提取特征向量。
* 保存：

  * `faiss.index`：向量索引
  * `embeddings.npy`：所有 embedding
  * `ids.npy`：图片路径映射

运行：

```bash
python code/build_index.py
```

结果保存在 `index/` 文件夹。

### 4. 图像检索（`search.py`）

* 遍历 `data/queries/` 下的所有图片
* 每张图执行完整预处理 + CLIP 向量化
* 用 FAISS 搜索前 K 个最相似的图库图片
* 打印命中路径及 L2 距离（越小越相似）

运行示例：

```bash
python code/search.py --query_dir data/queries --index index --topk 5
```

输出示例：

```
[info] 将对 2 张查询图执行检索（Top-5）

=== Query 1: data/queries/q1.jpg
1. authorA/img001.jpg    L2: 0.4435
2. authorB/img023.jpg    L2: 0.6096
...

=== Query 2: data/queries/q2.png
1. authorC/img102.jpg    L2: 0.5123
...
```

### 5. 线性分类微调（`finetune_linear.py`）

* 自定义 `Dataset`：对图片做完整预处理并提取特征
* 在线性层上微调，完成作者/类别分类

运行示例（需准备标签列表）：

```python
# finetune_linear.py 中 main 函数示例
image_paths = ["data/gallery/authorA/img1.jpg", "data/gallery/authorB/img2.jpg"]
labels = [0, 1]  # 与 image_paths 一一对应

embedder = ClipEmbedder()
finetune_linear_classifier(image_paths, labels, embedder)
```

模型会保存到 `ckpts/linear_classifier.pth`。

---

## 🚀 推荐使用流程

1. 准备图库 → 放入 `data/gallery/`
2. 构建索引

   ```bash
   python code/build_index.py
   ```
3. 准备查询集 → 放入 `data/queries/`
4. 执行检索

   ```bash
   python code/search.py --query_dir data/queries --index index --topk 5
   ```
5. （可选）准备带标签的训练数据，执行线性分类微调

   ```bash
   python code/finetune_linear.py
   ```

---

## 📌 注意事项

* 预处理耗时比单纯读图要长，图库大时可考虑先离线批处理保存。
* 距离度量使用 **L2 距离**，值越小越相似。
* 如需更快检索，可将 `faiss.IndexFlatL2` 换成 `IVF` 或 `HNSW` 索引。

---