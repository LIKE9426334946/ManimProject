import os
import shutil
from pathlib import Path

# ==================== 配置区域 ====================

SOURCE_DIR = Path(r"D:\08_Projects\ManimProject\media\videos")  # 源目录
TARGET_DIR = Path(r"D:\07_Images\07_线性代数")  # 目标目录

MAX_DEPTH = 2  # 最多遍历2层子目录

# ==================================================


def get_unique_path(path):
    """避免目标目录中出现同名文件覆盖。"""
    if not path.exists():
        return path

    index = 1
    while True:
        new_path = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not new_path.exists():
            return new_path
        index += 1


def move_mp4():
    source = SOURCE_DIR.resolve()
    target = TARGET_DIR.resolve()

    if not source.is_dir():
        print(f"源目录不存在：{source}")
        return

    if source == target:
        print("源目录和目标目录不能相同！")
        return

    target.mkdir(parents=True, exist_ok=True)

    count = 0

    for current_dir, dirs, files in os.walk(source):

        current_path = Path(current_dir)

        # 当前目录相对于源目录的深度
        depth = len(current_path.relative_to(source).parts)

        # 最多遍历 MAX_DEPTH 层
        if depth >= MAX_DEPTH:
            dirs.clear()

        # 如果目标目录位于源目录内部，则跳过目标目录
        dirs[:] = [d for d in dirs if (current_path / d).resolve() != target]

        for filename in files:

            # 只处理 MP4 文件（不区分扩展名大小写）
            if Path(filename).suffix.lower() != ".mp4":
                continue

            src_file = current_path / filename

            # 避免覆盖同名文件
            dst_file = get_unique_path(target / filename)

            try:
                shutil.move(str(src_file), str(dst_file))
                count += 1
                print(f"已移动：{src_file} -> {dst_file}")

            except Exception as e:
                print(f"移动失败：{src_file}")
                print(f"原因：{e}")

    print("\n========== 执行完成 ==========")
    print(f"成功移动文件数量：{count}")
    print(f"目标目录：{target}")


if __name__ == "__main__":
    move_mp4()
