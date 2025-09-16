import os
import argparse
import re

def rename_images(directory, prefix="", start_index=1):
    """
    重命名目录中的图片文件为连续数字格式
    
    参数:
        directory: 图片所在的目录路径
        prefix: 文件名前缀(可选)
        start_index: 起始编号(默认为1)
    """
    # 支持的图片扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    # 获取目录中所有图片文件
    image_files = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(filename)
    
    # 按文件名排序
    image_files.sort()
    
    # 重命名文件
    counter = start_index
    renamed_count = 0
    
    for filename in image_files:
        ext = os.path.splitext(filename)[1]
        new_name = f"{prefix}{counter:04d}{ext}"
        
        # 构建完整路径
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        
        # 重命名文件
        try:
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_name}")
            counter += 1
            renamed_count += 1
        except OSError as e:
            print(f"无法重命名 {filename}: {e}")
    
    print(f"\n成功重命名 {renamed_count} 个文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量重命名图片文件为0001, 0002等格式')
    parser.add_argument('directory', help='包含图片的目录路径')
    parser.add_argument('-p', '--prefix', default="", help='文件名前缀(可选)')
    parser.add_argument('-s', '--start', type=int, default=1, help='起始编号(默认为1)')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.isdir(args.directory):
        print(f"错误: 目录 '{args.directory}' 不存在")
        exit(1)
    
    rename_images(args.directory, args.prefix, args.start)