"""
=============================================================================
模块2：数据探索与初步分析 (EDA - Exploratory Data Analysis)
=============================================================================
对 Steam Games 数据集进行初步探查，了解：
- 数据规模与字段构成
- 缺失值与异常值分布
- 关键指标的统计特征
- 确定预测目标与特征工程方向
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号显示

RAW_CSV = "data/steam_games_raw.csv"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def load_data():
    """加载本地保存的原始数据"""
    print(f"[加载数据] {RAW_CSV}")
    df = pd.read_csv(RAW_CSV, encoding='utf-8-sig')
    print(f"  记录数: {len(df)}, 字段数: {len(df.columns)}")
    return df


def data_overview(df):
    """数据总览"""
    print("\n" + "=" * 60)
    print("一、数据总览")
    print("=" * 60)

    print(f"\n数据集维度: {df.shape[0]} 行 × {df.shape[1]} 列\n")

    print("全部字段名称及类型:")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_rate = (1 - non_null / len(df)) * 100
        print(f"  [{i:2d}] {col:25s} | 类型: {str(dtype):10s} | 非空: {non_null:6d} | 缺失率: {null_rate:5.1f}%")

    # 缺失率汇总
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(1)
    missing_df = pd.DataFrame({
        '缺失数': missing[missing > 0],
        '缺失率(%)': missing_pct[missing > 0]
    })
    if len(missing_df) > 0:
        print(f"\n⚠ 存在缺失的字段 ({len(missing_df)} 个):")
        print(missing_df.to_string())

    return missing_df


def numeric_summary(df):
    """数值字段统计"""
    print("\n" + "=" * 60)
    print("二、数值字段统计描述")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print(f"\n数值字段数: {len(numeric_cols)}")
        desc = df[numeric_cols].describe().round(2)
        print(desc.to_string())


def price_analysis(df):
    """价格分布分析——重点探索"""
    print("\n" + "=" * 60)
    print("三、价格分布分析（核心探索）")
    print("=" * 60)

    price_cols = [c for c in df.columns if 'price' in c.lower()]
    if not price_cols:
        print("  未找到价格相关字段")
        return

    for col in price_cols:
        series = df[col]

        # 处理非数值
        if series.dtype == 'object':
            print(f"\n  字段 '{col}' 为文本类型，前20个唯一值:")
            print(f"  {series.value_counts().head(20).to_dict()}")
            # 尝试提取数值
            numeric = pd.to_numeric(series, errors='coerce')
            valid = numeric.dropna()
            if len(valid) > 0:
                print(f"\n  提取到 {len(valid)} 个有效数值:")
                print(f"    min={valid.min():.2f}, max={valid.max():.2f}, mean={valid.mean():.2f}, median={valid.median():.2f}")
                print(f"    免费游戏数: {(valid == 0).sum()}")
            continue

        valid = series.dropna()
        print(f"\n  字段 '{col}' 统计:")
        print(f"    总数: {len(valid)}, 缺失: {series.isna().sum()}")
        print(f"    min={valid.min():.4f}, max={valid.max():.4f}")
        print(f"    mean={valid.mean():.4f}, median={valid.median():.4f}")
        print(f"    25%: {valid.quantile(0.25):.4f}, 75%: {valid.quantile(0.75):.4f}")

        # 特殊值
        zeros = (valid == 0).sum()
        if zeros > 0:
            print(f"    零值(免费?): {zeros} ({zeros/len(valid)*100:.1f}%)")


def rating_analysis(df):
    """评分分布分析——预测目标候选"""
    print("\n" + "=" * 60)
    print("四、评分/评价分布分析")
    print("=" * 60)

    # 找评分相关字段
    rating_cols = [c for c in df.columns
                   if any(k in c.lower() for k in ['rating', 'score', 'review', 'positive', 'negative',
                                                    'recommend', 'metacritic', 'steam'])]

    for col in rating_cols[:8]:  # 最多看8个
        series = df[col]
        if series.dtype == 'object':
            try:
                numeric = pd.to_numeric(series, errors='coerce')
                valid = numeric.dropna()
                if len(valid) > 0 and len(valid) > len(df) * 0.1:
                    print(f"\n  字段 '{col}' (从文本提取数值):")
                    print(f"    有效值: {len(valid)}, 均值: {valid.mean():.2f}, "
                          f"中位数: {valid.median():.2f}, 范围: [{valid.min():.2f}, {valid.max():.2f}]")
                else:
                    print(f"\n  字段 '{col}': 文本类型，前5个值: {series.dropna().head().tolist()}")
            except Exception:
                print(f"\n  字段 '{col}': 文本类型，前5个值: {series.dropna().head().tolist()}")
        else:
            valid = series.dropna()
            if len(valid) > 0:
                print(f"\n  字段 '{col}': mean={valid.mean():.2f}, median={valid.median():.2f}, "
                      f"min={valid.min():.2f}, max={valid.max():.2f}")


def categorical_analysis(df):
    """类别字段分析"""
    print("\n" + "=" * 60)
    print("五、类别/标签字段分析")
    print("=" * 60)

    object_cols = df.select_dtypes(include=['object']).columns.tolist()

    # 挑可能有分类意义的字段
    skip_keywords = ['url', 'image', 'icon', 'logo', 'header', 'background', 'screenshot',
                     'thumbnail', 'avatar', 'description', 'about', 'detail', 'legal', 'support',
                     'website', 'link', 'movie', 'path', 'filename']
    cat_cols = [c for c in object_cols
                if not any(sk in c.lower() for sk in skip_keywords)]

    for col in cat_cols[:10]:
        series = df[col].dropna()
        unique_count = series.nunique()
        sample_values = series.head(5).tolist()

        print(f"\n  字段 '{col}'")
        print(f"    唯一值数: {unique_count}, 非空数: {len(series)}")
        print(f"    前5个值: {sample_values}")

        # 如果唯一值较少，展示分布
        if unique_count <= 20 and unique_count > 1:
            top = series.value_counts().head(10)
            print(f"    分布: {dict(top)}")


def main():
    print("\n" + "█" * 60)
    print("█  模块2: 数据探索与初步分析 (EDA)")
    print("█" * 60)

    df = load_data()

    data_overview(df)
    numeric_summary(df)
    price_analysis(df)
    rating_analysis(df)
    categorical_analysis(df)

    print("\n" + "=" * 60)
    print("[EDA完成] 请根据以上探索结果确定预测目标和特征工程策略。")
    print("=" * 60)


if __name__ == "__main__":
    main()
