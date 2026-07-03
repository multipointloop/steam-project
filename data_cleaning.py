"""
=============================================================================
模块3：数据清洗 (Data Cleaning)
=============================================================================
将原始Steam数据集中的文本字段解析为结构化数值/类别数据：
- 价格字段：去除$符号，转为浮点数
- 评分字段：从文本中提取评分百分比和评论数
- 日期字段：解析"3 Aug, 2023"等格式
- 类别字段：解析列表型字符串，提取特征
- 缺失值处理与异常值过滤
=============================================================================
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime


RAW_CSV = "data/steam_games_raw.csv"
CLEAN_CSV = "data/steam_games_clean.csv"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_price(price_str):
    """
    价格清洗：'$9.99' -> 9.99, 'Free' -> 0.0

    处理逻辑：
    - 去除$前缀和逗号
    - 'Free' 统一为 0
    - 无法解析的返回 NaN
    """
    if pd.isna(price_str):
        return np.nan
    price_str = str(price_str).strip()
    if price_str.lower() in ['free', 'free to play', 'free demo', '']:
        return 0.0
    # 移除 $ 和逗号
    cleaned = price_str.replace('$', '').replace(',', '')
    try:
        val = float(cleaned)
        return val if val >= 0 else np.nan
    except ValueError:
        return np.nan


def parse_review_info(text):
    """
    从评论文本中提取评分百分比和评论数量。

    示例输入:
      "- 96% of the 128,900 user reviews in the last 30 days are positive."
    返回:
      (96.0, 128900)
    """
    if pd.isna(text):
        return np.nan, np.nan

    text = str(text)

    # 提取百分比
    pct_match = re.search(r'(\d+)%', text)
    pct = float(pct_match.group(1)) if pct_match else np.nan

    # 提取评论数量（可能是 "128,900" 或 "5,312"）
    num_match = re.search(r'of the ([\d,]+) user reviews', text)
    if num_match:
        num = int(num_match.group(1).replace(',', ''))
    else:
        num = np.nan

    return pct, num


SENTIMENT_MAP = {
    'Overwhelmingly Positive': 9,
    'Very Positive': 8,
    'Positive': 7,
    'Mostly Positive': 6,
    'Mixed': 5,
    'Mostly Negative': 4,
    'Negative': 3,
    'Very Negative': 2,
    'Overwhelmingly Negative': 1,
}


def map_sentiment(text):
    """
    将评分摘要映射为1-9的序数值。
    1 = 极差, 9 = 极好
    """
    if pd.isna(text):
        return np.nan
    # 尝试直接匹配
    for key, val in SENTIMENT_MAP.items():
        if key in str(text):
            return val
    return np.nan


def parse_date(date_str):
    """解析日期: '3 Aug, 2023' -> datetime"""
    if pd.isna(date_str):
        return pd.NaT
    try:
        return pd.to_datetime(date_str, format='%d %b, %Y', errors='coerce')
    except Exception:
        return pd.NaT


def parse_list_string(s):
    """将Python列表字符串安全解析为list"""
    if pd.isna(s):
        return []
    try:
        # 使用 ast.literal_eval 比 eval 更安全
        import ast
        return ast.literal_eval(str(s))
    except (ValueError, SyntaxError):
        return []


def count_items(list_str):
    """计算列表中的元素数量"""
    items = parse_list_string(list_str)
    return len(items)


def main():
    print("\n" + "=" * 60)
    print("[数据清洗] 开始处理 Steam 数据集...")
    print("=" * 60)

    # ====== 加载原始数据 ======
    print("\n[1/6] 加载原始数据...")
    df = pd.read_csv(RAW_CSV, encoding='utf-8-sig')
    print(f"  原始记录数: {len(df)}")

    # ====== 价格清洗 ======
    print("\n[2/6] 清洗价格字段...")
    df['OriginalPrice'] = df['Original Price'].apply(clean_price)
    df['DiscountedPrice'] = df['Discounted Price'].apply(clean_price)
    # 折扣率
    df['DiscountRate'] = np.where(
        df['OriginalPrice'] > 0,
        (1 - df['DiscountedPrice'] / df['OriginalPrice']) * 100,
        0
    )
    df['IsFree'] = (df['OriginalPrice'] == 0).astype(int)
    df['IsOnSale'] = (df['DiscountedPrice'] < df['OriginalPrice']).astype(int)

    valid_price = df['OriginalPrice'].notna()
    print(f"  有效价格记录: {valid_price.sum()} / {len(df)}")
    print(f"  免费游戏: {df['IsFree'].sum()}")
    print(f"  打折游戏: {df['IsOnSale'].sum()}")
    print(f"  原价范围: ${df['OriginalPrice'].min():.2f} ~ ${df['OriginalPrice'].max():.2f}")
    print(f"  原价中位数: ${df['OriginalPrice'].median():.2f}")

    # ====== 评分解析 ======
    print("\n[3/6] 解析评分数据...")
    # Recent Reviews
    recent_info = df['Recent Reviews Number'].apply(parse_review_info)
    df['RecentReviewPct'] = recent_info.apply(lambda x: x[0])
    df['RecentReviewCount'] = recent_info.apply(lambda x: x[1])
    df['RecentSentiment'] = df['Recent Reviews Summary'].apply(map_sentiment)

    # All Reviews
    all_info = df['All Reviews Number'].apply(parse_review_info)
    df['AllReviewPct'] = all_info.apply(lambda x: x[0])
    df['AllReviewCount'] = all_info.apply(lambda x: x[1])
    df['AllSentiment'] = df['All Reviews Summary'].apply(map_sentiment)

    print(f"  近期评分有效记录: {df['RecentReviewPct'].notna().sum()}")
    print(f"  全量评分有效记录: {df['AllReviewPct'].notna().sum()}")
    print(f"  近期评分均值: {df['RecentReviewPct'].mean():.1f}%")

    # ====== 日期解析 ======
    print("\n[4/6] 解析发行日期...")
    df['ReleaseDate'] = df['Release Date'].apply(parse_date)
    df['ReleaseYear'] = df['ReleaseDate'].dt.year
    df['ReleaseMonth'] = df['ReleaseDate'].dt.month
    df['ReleaseDayOfYear'] = df['ReleaseDate'].dt.dayofyear

    # 过滤异常年份
    valid_year = (df['ReleaseYear'] >= 1980) & (df['ReleaseYear'] <= 2026)
    df.loc[~valid_year, ['ReleaseYear', 'ReleaseMonth', 'ReleaseDayOfYear']] = np.nan

    print(f"  有效日期记录: {valid_year.sum()}")
    print(f"  年份范围: {df['ReleaseYear'].min():.0f} ~ {df['ReleaseYear'].max():.0f}")

    # ====== 特征解析 ======
    print("\n[5/6] 解析游戏特征、标签和语言...")
    df['NumFeatures'] = df['Game Features'].apply(count_items)
    df['NumTags'] = df['Popular Tags'].apply(count_items)
    df['NumLanguages'] = df['Supported Languages'].apply(count_items)

    # 关键特征标记
    df['HasSinglePlayer'] = df['Game Features'].str.contains('Single-player', na=False).astype(int)
    df['HasMultiplayer'] = df['Game Features'].apply(
        lambda x: 1 if any(k in str(x) for k in ['Multiplayer', 'Online PvP', 'Online Co-op', 'LAN', 'MMO']) else 0
    )
    df['HasControllerSupport'] = df['Game Features'].str.contains('controller', case=False, na=False).astype(int)
    df['HasAchievements'] = df['Game Features'].str.contains('Steam Achievements', na=False).astype(int)
    df['HasTradingCards'] = df['Game Features'].str.contains('Steam Trading Cards', na=False).astype(int)
    df['HasInAppPurchases'] = df['Game Features'].str.contains('In-App Purchases', na=False).astype(int)
    df['HasWorkshop'] = df['Game Features'].str.contains('Steam Workshop', na=False).astype(int)

    # 中文支持
    df['SupportsChinese'] = df['Supported Languages'].str.contains('Simplified Chinese|Traditional Chinese', na=False).astype(int)

    print(f"  平均特征数: {df['NumFeatures'].mean():.1f}")
    print(f"  平均标签数: {df['NumTags'].mean():.1f}")
    print(f"  平均语言数: {df['NumLanguages'].mean():.1f}")
    print(f"  支持中文: {df['SupportsChinese'].sum()} ({df['SupportsChinese'].sum()/len(df)*100:.1f}%)")

    # ====== 热门标签 One-Hot编码 ======
    print("\n[6/6] 热门标签编码...")

    # 收集所有标签
    all_tags = []
    for tags_str in df['Popular Tags'].dropna():
        all_tags.extend(parse_list_string(tags_str))

    # 取Top 30热门标签
    tag_counts = pd.Series(all_tags).value_counts()
    top_tags = tag_counts.head(30).index.tolist()
    print(f"  Top标签: {top_tags[:10]}...")

    # 为每个Top标签创建二值特征
    for tag in top_tags:
        col_name = 'Tag_' + tag.replace(' ', '_').replace('-', '_').replace("'", '')
        df[col_name] = df['Popular Tags'].apply(
            lambda x: 1 if tag in parse_list_string(x) else 0
        ).astype(int)

    # ====== 保存清洗后的数据 ======
    # 只保留清洗后的字段 + 必要原始字段
    keep_cols = [
        # 基础信息
        'Title', 'Developer', 'Publisher', 'Link',
        # 原始文本特征（前端展示用）
        'Popular Tags', 'Game Features', 'Supported Languages', 'Game Description',
        'Recent Reviews Summary', 'All Reviews Summary',
        # 价格
        'OriginalPrice', 'DiscountedPrice', 'DiscountRate', 'IsFree', 'IsOnSale',
        # 评分（预测目标）
        'RecentReviewPct', 'RecentReviewCount', 'RecentSentiment',
        'AllReviewPct', 'AllReviewCount', 'AllSentiment',
        # 日期
        'ReleaseDate', 'ReleaseYear', 'ReleaseMonth', 'ReleaseDayOfYear',
        # 特征
        'NumFeatures', 'NumTags', 'NumLanguages',
        'HasSinglePlayer', 'HasMultiplayer', 'HasControllerSupport',
        'HasAchievements', 'HasTradingCards', 'HasInAppPurchases', 'HasWorkshop',
        'SupportsChinese',
    ] + [f'Tag_{t.replace(" ", "_").replace("-", "_").replace("\'", "")}' for t in top_tags]

    clean_df = df[[c for c in keep_cols if c in df.columns]].copy()

    # 过滤：至少要有评分数据（否则没法做预测）
    clean_df = clean_df[clean_df['RecentReviewPct'].notna()].copy()
    clean_df = clean_df[clean_df['OriginalPrice'].notna()].copy()

    clean_df.to_csv(CLEAN_CSV, index=False, encoding='utf-8-sig')
    print(f"\n{'=' * 60}")
    print(f"[数据清洗完成] 清洗后记录数: {len(clean_df)}")
    print(f"  保存至: {CLEAN_CSV}")
    print(f"  字段数: {len(clean_df.columns)}")
    print(f"{'=' * 60}")

    return clean_df


if __name__ == "__main__":
    main()
