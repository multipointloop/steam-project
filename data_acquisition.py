"""
=============================================================================
模块1：数据获取 (Data Acquisition)
=============================================================================
数据来源: Kaggle - Steam Games Dataset
数据集地址: https://www.kaggle.com/datasets/nikatomashvili/steam-games-dataset
包含字段: 游戏名称、价格、评分、发行日期、开发商、发行商、标签、类别等

合规说明: Kaggle是公开数据平台，该数据集为社区公开共享，符合
         "免费公开、可下载"的要求，不涉及付费/涉密/商业私有数据。
=============================================================================
"""

import os
import sys
import kagglehub
import pandas as pd
from datetime import datetime


# ========== 配置 ==========
OUTPUT_DIR = "data"
DATASET_PATH = "nikatomashvili/steam-games-dataset"  # Kaggle上的数据集标识
RAW_CSV = os.path.join(OUTPUT_DIR, "steam_games_raw.csv")


def ensure_dir(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def download_steam_dataset():
    """
    通过 kagglehub 下载 Steam Games 数据集。

    工作原理：
    - kagglehub 会自动从 Kaggle 拉取公开数据集并缓存到本地
    - 返回下载后的本地路径
    - 不需要 Kaggle API Key（公开数据集可直接下载）
    """
    print("=" * 60)
    print("[数据获取] 正在从 Kaggle 下载 Steam Games 数据集...")
    print(f"           数据集: {DATASET_PATH}")
    print(f"           下载时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        # kagglehub 一行搞定：自动下载 + 返回本地路径
        download_path = kagglehub.dataset_download(DATASET_PATH)
        print(f"\n[OK] 下载成功！原始文件路径: {download_path}")

        # 列出下载的文件
        files = os.listdir(download_path)
        print(f"  文件列表: {files}")

        # 找到CSV文件
        csv_files = [f for f in files if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"数据集中未找到CSV文件，文件列表: {files}")

        return download_path, csv_files

    except Exception as e:
        print(f"\n[ERROR] 下载失败: {e}")
        raise


def load_and_save_raw(download_path, csv_files):
    """
    从下载目录读取原始CSV，标准化后保存到项目 data/ 目录。
    实现"数据保存"模块要求——本地存放，方便实时调取。
    """
    ensure_dir(OUTPUT_DIR)

    all_dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(download_path, csv_file)
        print(f"\n[数据保存] 读取: {csv_file}")

        df = pd.read_csv(file_path)
        print(f"           行数: {len(df)}, 列数: {len(df.columns)}")
        print(f"           列名: {list(df.columns)[:10]}...")

        all_dfs.append(df)

    # 合并多个CSV（如果有的话）
    if len(all_dfs) == 1:
        combined = all_dfs[0]
    else:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\n  合并后总行数: {len(combined)}")

    # 保存到本地
    combined.to_csv(RAW_CSV, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 已保存到本地: {os.path.abspath(RAW_CSV)}")
    print(f"  总记录数: {len(combined)}")
    print(f"  总字段数: {len(combined.columns)}")

    return combined


def main():
    """数据获取主流程"""
    print("\n" + "█" * 60)
    print("█  2026实践学期 - Steam游戏数据分析项目")
    print("█  模块1: 数据获取与本地保存")
    print("█" * 60 + "\n")

    # Step 1: 下载
    download_path, csv_files = download_steam_dataset()

    # Step 2: 本地保存
    df = load_and_save_raw(download_path, csv_files)

    print("\n" + "=" * 60)
    print("[数据获取] 完成！可以继续运行 data_exploration.py 查看数据概况。")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
