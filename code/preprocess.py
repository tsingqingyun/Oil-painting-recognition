import cv2
import numpy as np
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from PIL import Image

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def adjust_brightness_contrast(image, brightness=10, contrast=15):
    """调整图像亮度和对比度"""
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    
    return image

def preprocess_image(image):
    """预处理图像以提高边缘检测效果"""
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 使用CLAHE增强对比度
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 中值滤波减少噪声
    gray = cv2.medianBlur(gray, 5)
    
    return gray

def find_edges(image):
    """查找图像中的边缘"""
    # 使用自适应阈值处理
    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 边缘检测
    edges = cv2.Canny(thresh, 75, 150)
    
    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    return edges

def find_largest_quadrilateral(edges, original_image, min_area_ratio=0.05):
    """在边缘图像中寻找最大的四边形，增加参数控制"""
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # 按面积排序轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = original_image.shape[0] * original_image.shape[1]
    min_area = image_area * min_area_ratio
    
    # 寻找近似四边形的轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:  # 提前过滤小面积轮廓
            continue
            
        perimeter = cv2.arcLength(contour, True)
        # 动态调整近似精度
        epsilon = max(0.02 * perimeter, 3)  # 确保至少有3像素的精度
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 检查是否为四边形
        if len(approx) == 4:
            if cv2.isContourConvex(approx):
                return approx
    
    # 尝试放宽条件，允许非凸四边形
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)  # 更宽松的近似
        if len(approx) == 4:
            return approx
    
    return None

def order_points(pts):
    """对四个点进行排序：左上，右上，右下，左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def auto_crop_border(image, border_color_threshold=30):
    """自动裁剪图像边缘的有色边框"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 将接近边框颜色的区域设为白色，其他为黑色
    _, thresh = cv2.threshold(gray, border_color_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 获取最大轮廓的边界矩形
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 确保裁剪后的图像不会太小
        if w > image.shape[1] * 0.5 and h > image.shape[0] * 0.5:
            # 裁剪图像
            cropped = image[y:y+h, x:x+w]
            return cropped
    
    return image

def detect_and_warp(image, quadrilateral):
    """检测四边形并执行透视变换，增加边界检查"""
    # 对点进行排序
    rect = order_points(quadrilateral.reshape(4, 2))
    
    # 计算新图像的宽度和高度
    (tl, tr, br, bl) = rect
    
    # 计算宽度和高度（增加异常值处理）
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))
    
    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)
    max_height = max(int(height_left), int(height_right))
    
    # 防止尺寸为零的异常情况
    if max_width == 0 or max_height == 0:
        return None
        
    # 限制最大尺寸，防止内存溢出
    max_dimension = 4096
    if max_width > max_dimension or max_height > max_dimension:
        scale = max_dimension / max(max_width, max_height)
        max_width = int(max_width * scale)
        max_height = int(max_height * scale)
    
    # 定义目标点
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    
    try:
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(rect, dst)
        # 应用透视变换
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        return warped
    except Exception as e:
        logger.warning(f"透视变换失败: {str(e)}")
        return None
    
def enhance_image(image):
    """增强图像质量"""
    # 转换为LAB颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 分离通道
    l, a, b = cv2.split(lab)
    
    # 对L通道应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    
    # 合并通道
    lab = cv2.merge((l, a, b))
    
    # 转换回BGR颜色空间
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def perspective_correction(image_path, output_path):
    """执行透视矫正和裁剪"""
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return False
        
        orig = image.copy()
        
        # 预处理图像
        processed = preprocess_image(image)
        
        # 查找边缘
        edges = find_edges(processed)
        
        # 查找最大的四边形
        quadrilateral = find_largest_quadrilateral(edges, image)
        
        if quadrilateral is None:
            # 尝试使用霍夫直线检测来寻找边界
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 
                threshold=100, minLineLength=100, maxLineGap=10
            )
            
            if lines is not None:
                # 创建一个空白图像绘制所有直线
                line_image = np.zeros_like(image)
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                
                # 从直线图像中再次查找轮廓
                line_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
                quadrilateral = find_largest_quadrilateral(line_gray, image)
        
        if quadrilateral is None:
            # 无法检测到四边形，使用原始图像但尝试裁剪边框
            logger.warning(f"无法检测到四边形，将尝试裁剪边框: {image_path}")
            final = auto_crop_border(orig)
            # 增强图像
            final = enhance_image(final)
            cv2.imwrite(output_path, final)
            return True
        
        # 执行透视变换
        warped = detect_and_warp(orig, quadrilateral)
        
        # 自动裁剪黑色边框
        final = auto_crop_border(warped)
        
        # 增强图像
        final = enhance_image(final)
        
        # 保存结果
        cv2.imwrite(output_path, final)
        return True
        
    except Exception as e:
        logger.error(f"处理图像 {image_path} 时出错: {str(e)}")
        return False

def process_single_file(input_path, output_path):
    """处理单个文件的包装函数，用于多线程处理"""
    filename = os.path.basename(input_path)
    success = perspective_correction(input_path, output_path)
    return (filename, success)

def process_directory(input_dir, output_dir, max_workers=None):
    """处理目录中的所有图像，支持多线程"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 支持的图像格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')
    
    # 收集所有图像文件
    image_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            # 确保是文件而不是目录
            if os.path.isfile(input_path):
                image_files.append(input_path)
    
    if not image_files:
        logger.error(f"在目录 {input_dir} 中未找到任何支持的图像文件")
        return
    
    logger.info(f"发现 {len(image_files)} 个图像文件，开始处理...")
    
    # 确定最大工作线程数，默认为CPU核心数
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(image_files))
    
    # 使用线程池处理文件
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建任务
        futures = []
        for input_path in image_files:
            filename = os.path.basename(input_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_corrected{ext}")
            futures.append(executor.submit(process_single_file, input_path, output_path))
        
        # 显示进度并处理结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            filename, success = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                logger.error(f"处理失败: {filename}")
    
    logger.info(f"处理完成。成功: {success_count}, 失败: {fail_count}")
    logger.info(f"处理结果已保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='自动透视矫正和裁剪美术馆照片')
    parser.add_argument('--input', '-i', required=True, help='输入图像路径或目录')
    parser.add_argument('--output', '-o', required=True, help='输出图像路径或目录')
    parser.add_argument('--threads', '-t', type=int, default=None, help='处理线程数，默认使用CPU核心数')
    parser.add_argument('--debug', '-d', action='store_true', help='启用调试模式，显示中间处理步骤')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        # 处理整个目录
        process_directory(args.input, args.output, args.threads)
    else:
        # 处理单个文件
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        success = perspective_correction(args.input, args.output)
        if success:
            logger.info(f"处理完成，结果已保存到: {args.output}")
        else:
            logger.error("处理失败!")

def preprocess_for_painting(image_path):
    """
    供 CLIP 嵌入使用的最小预处理：读取路径 -> 返回 RGB PIL.Image
    保持和 ClipEmbedder.encode_images 的预处理流水线兼容。
    """
    img = Image.open(image_path).convert("RGB")
    return img

# 完整 OpenCV 流程（透视矫正 + 去边框 + 增强）→ 转 PIL.RGB，供 CLIP 使用
def preprocess_full_for_clip(image_path):
    """
    读取 image_path -> 透视矫正/去边框/增强（内存中完成，不写磁盘）
    -> 转为 PIL RGB 图像返回。
    """
    # 1) 读原图（BGR）
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    orig = image.copy()

    # 2) 预处理 + 边缘（与你现有函数一致）
    processed = preprocess_image(image)             # 灰度 + CLAHE + 中值滤波:contentReference[oaicite:0]{index=0}
    edges = find_edges(processed)                   # 自适应阈值 + Canny + 形态学:contentReference[oaicite:1]{index=1}
    quad = find_largest_quadrilateral(edges, image) # 最大四边形候选:contentReference[oaicite:2]{index=2}

    # 3) 若没找到四边形，尝试霍夫直线再找一次（复用你已有逻辑）
    if quad is None:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        if lines is not None:
            line_image = np.zeros_like(image)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            line_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
            quad = find_largest_quadrilateral(line_gray, image)

    # 4) 透视变换/裁剪
    if quad is not None:
        warped = detect_and_warp(orig, quad)        # 失败返回 None:contentReference[oaicite:3]{index=3}
        img2 = warped if warped is not None else orig
    else:
        img2 = auto_crop_border(orig)               # 找不到四边形就直接去边框:contentReference[oaicite:4]{index=4}

    # 5) 增强（LAB-CLAHE）:contentReference[oaicite:5]{index=5}
    img3 = enhance_image(img2)

    # 6) OpenCV(BGR) → PIL(RGB)
    img_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return pil_img

if __name__ == "__main__":
    main()