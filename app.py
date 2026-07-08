"""
=============================================================================
Steam Game Analytics - Flask Backend
=============================================================================
提供REST API：游戏列表、详情、实时预测、统计概览
启动方式: python app.py
访问地址: http://localhost:5000
=============================================================================
"""
import os
import ast
import pickle
import sqlite3
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, g
from sklearn.ensemble import IsolationForest

app = Flask(__name__, static_folder='.', static_url_path='')

# ====== 加载预训练模型 ======
MODEL_PATH = "data/model_rf.pkl"
FEATURES_PATH = "data/feature_columns.pkl"
DB_PATH = "data/steam.db"

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(FEATURES_PATH, 'rb') as f:
    feat_data = pickle.load(f)
    FEATURE_COLS = feat_data['feature_cols']
    IMPORTANCES = feat_data['importances']


def get_db():
    """获取数据库连接（每个请求一个连接）"""
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()


# ====== 预计算异常检测结果（服务启动时运行一次） ======
ANOMALY_CACHE = None

def init_anomaly_detection():
    """启动时运行IsolationForest，缓存结果"""
    global ANOMALY_CACHE
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT id, Title, OriginalPrice, NumFeatures, NumTags, NumLanguages,
               RecentReviewPct, RecentReviewCount, ReleaseYear, AllReviewPct,
               HasInAppPurchases, IsFree, Developer, Publisher, DiscountRate
        FROM games
    """).fetchall()
    db.close()

    df = pd.DataFrame([dict(r) for r in rows])
    anomaly_cols = ['OriginalPrice', 'NumFeatures', 'NumTags', 'NumLanguages',
                    'RecentReviewPct', 'RecentReviewCount', 'ReleaseYear']
    X = df[anomaly_cols].dropna()
    valid_idx = X.index

    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    labels = iso.fit_predict(X.fillna(0))
    scores = iso.score_samples(X.fillna(0))

    df['is_anomaly'] = 0
    df['anomaly_score'] = 0.0
    df.loc[valid_idx, 'is_anomaly'] = (labels == -1).astype(int)
    df.loc[valid_idx, 'anomaly_score'] = scores

    # 给异常分类原因（中文）
    def classify_reason(row):
        reasons = []
        p, s = row['OriginalPrice'], row['RecentReviewPct']
        if p > 100 and s < 30: reasons.append('高价低分异常')
        elif p == 0 and s > 95: reasons.append('免费游戏极高评分')
        elif p > 60 and s > 90: reasons.append('高价高评价')
        elif p < 2 and s < 25: reasons.append('低价低评分')
        elif p > 30 and s < 40: reasons.append('价格评分不匹配')
        if row['RecentReviewCount'] > 100000: reasons.append('极高人气')
        if row['NumLanguages'] > 20: reasons.append('极多语言支持')
        if row['NumFeatures'] > 12: reasons.append('功能密集')
        return reasons or ['异常模式']

    anom_df = df[df['is_anomaly'] == 1].copy()
    anom_df['reasons'] = anom_df.apply(classify_reason, axis=1)

    ANOMALY_CACHE = {
        'df': df,
        'anomalies': anom_df,
        'total_games': len(df),
        'anomaly_count': len(anom_df),
        'anomaly_pct': round(len(anom_df) / len(df) * 100, 1),
        'normal_avg_score': round(float(df[df['is_anomaly'] == 0]['RecentReviewPct'].mean()), 1),
        'anomaly_avg_score': round(float(anom_df['RecentReviewPct'].mean()), 1),
        'normal_avg_price': round(float(df[df['is_anomaly'] == 0]['OriginalPrice'].mean()), 2),
        'anomaly_avg_price': round(float(anom_df['OriginalPrice'].mean()), 2),
    }
    print(f"[Anomaly Detection] {ANOMALY_CACHE['anomaly_count']} anomalies found out of {ANOMALY_CACHE['total_games']} games")


def generate_review_history(game, months=60, model=None, feature_cols=None):
    """为单个游戏生成模拟的月度评论历史时序数据"""
    random.seed(game['id'] if game.get('id') else 42)
    np.random.seed(game['id'] if game.get('id') else 42)

    # 计算游戏已发行月数
    release_year = game.get('ReleaseYear') or 2020
    release_month = game.get('ReleaseMonth') or 6
    if release_year is None or np.isnan(release_year):
        release_year = 2020
    if release_month is None or np.isnan(release_month):
        release_month = 6
    release_year, release_month = int(release_year), int(release_month)

    current_year, current_month = 2026, 7
    total_months = (current_year - release_year) * 12 + (current_month - release_month)
    total_months = max(6, min(total_months, 120))

    # 获取评分数据
    all_pct = game.get('AllReviewPct')
    recent_pct = game.get('RecentReviewPct')
    all_count = game.get('AllReviewCount') or 0
    recent_count = game.get('RecentReviewCount') or 0

    if recent_pct is None or (isinstance(recent_pct, float) and np.isnan(recent_pct)):
        recent_pct = all_pct if all_pct and not (isinstance(all_pct, float) and np.isnan(all_pct)) else 75.0

    if all_pct is None or (isinstance(all_pct, float) and np.isnan(all_pct)):
        all_pct = recent_pct + random.uniform(-5, 5)

    all_pct = float(all_pct)
    recent_pct = float(recent_pct)
    total_reviews = max(float(recent_count or all_count or 100), 50)

    # S曲线分布评论量
    months_arr = np.arange(total_months)
    midpoint = total_months * 0.35
    rate = max(0.08, 3.0 / total_months)
    sigmoid = 1.0 / (1.0 + np.exp(-rate * (months_arr - midpoint)))
    sigmoid = sigmoid / sigmoid.sum()

    noise = np.random.normal(0, 0.03, total_months)
    weights = sigmoid + noise
    weights = np.maximum(weights, 0.001)
    weights = weights / weights.sum()

    monthly_total = (total_reviews * weights).astype(int)
    monthly_total = np.maximum(monthly_total, 1)

    # 评分轨迹
    start_score = all_pct + random.uniform(-3, 3)
    start_score = max(10, min(98, start_score))
    end_score = recent_pct

    scores = np.zeros(total_months)
    for i in range(total_months):
        progress = i / max(total_months - 1, 1)
        base = start_score + (end_score - start_score) * progress
        noise_s = np.random.normal(0, 2.5)
        scores[i] = base + noise_s

    # 最近30%的月份更靠近recent_pct
    recent_start = int(total_months * 0.7)
    for i in range(recent_start, total_months):
        progress = (i - recent_start) / max(total_months - recent_start - 1, 1)
        scores[i] = scores[i] * (1 - progress * 0.7) + end_score * progress * 0.7

    scores = np.clip(scores, 5, 99)
    scores[-1] = recent_pct  # 最后一个月精确匹配

    # 构建月度数据
    months_data = []
    cumulative_pos = 0
    cumulative_neg = 0

    for i in range(total_months):
        date_label = f"{release_year + (release_month - 1 + i) // 12:04d}-{(release_month - 1 + i) % 12 + 1:02d}"
        pos = int(monthly_total[i] * scores[i] / 100)
        neg = int(monthly_total[i]) - pos
        cumulative_pos += pos
        cumulative_neg += neg

        months_data.append({
            'date': date_label,
            'rating': round(float(scores[i]), 1),
            'total': int(monthly_total[i]),
            'positive': int(pos),
            'negative': int(neg),
            'cumulative_positive': int(cumulative_pos),
            'cumulative_negative': int(cumulative_neg),
        })

    # 模型预测端点（未来6个月延展）
    prediction_endpoint = None
    if model is not None and feature_cols is not None:
        try:
            X = pd.DataFrame([{c: game.get(c, 0) for c in feature_cols if c in game}])
            for c in feature_cols:
                if c not in X.columns:
                    X[c] = 0
            if 'DevEncoded' in feature_cols and 'DevEncoded' not in X.columns:
                X['DevEncoded'] = 0
            if 'PubEncoded' in feature_cols and 'PubEncoded' not in X.columns:
                X['PubEncoded'] = 0
            X = X[feature_cols].fillna(0)
            pred = float(model.predict(X)[0])
            pred = max(0, min(100, pred))

            last_date = months_data[-1]['date']
            y, m = int(last_date[:4]), int(last_date[5:7])
            future_dates = []
            for j in range(1, 7):
                nm = m + j
                ny = y + (nm - 1) // 12
                nm = (nm - 1) % 12 + 1
                future_dates.append(f"{ny:04d}-{nm:02d}")

            prediction_endpoint = {
                'dates': future_dates,
                'predicted_pct': round(pred, 1),
                'confidence_interval': [round(max(0, pred - 12), 1), round(min(100, pred + 12), 1)],
                'current_trend': 'up' if recent_pct > all_pct else ('down' if recent_pct < all_pct else 'stable'),
            }
        except Exception:
            prediction_endpoint = None

    return {
        'game_id': game.get('id'),
        'title': game.get('Title', ''),
        'months': months_data,
        'prediction_endpoint': prediction_endpoint,
        'generation_info': {
            'method': 'synthetic_s_curve',
            'has_all_time_data': all_pct is not None and not (isinstance(all_pct, float) and np.isnan(all_pct)),
            'months_generated': total_months,
            'total_reviews_modeled': int(total_reviews),
        }
    }


def analyze_gap(game, prediction, importances):
    """分析模型预测与实际评分偏差的原因"""
    actual = game.get('RecentReviewPct')
    if actual is None or prediction is None:
        return None

    actual = float(actual)
    prediction = float(prediction)
    gap = actual - prediction
    absolute_gap = abs(gap)

    if absolute_gap <= 15:
        return None

    direction = 'underestimated' if gap > 0 else 'overestimated'
    reasons = []

    # 规则1: 经典老游戏
    release_year = game.get('ReleaseYear') or 2020
    game_age = 2026 - release_year if release_year else 0
    if game_age > 10 and actual > prediction:
        reasons.append({
            'type': '经典老游戏效应',
            'text': f'该游戏发行已超过{game_age}年，已成为经典IP，老玩家情怀和历史口碑使实际评分远超模型基于技术指标的预期',
            'confidence': min(0.85, 0.5 + game_age * 0.02),
            'direction': 'positive',
        })

    # 规则2: 质量下滑
    all_pct = game.get('AllReviewPct')
    recent_pct = game.get('RecentReviewPct')
    if all_pct is not None and recent_pct is not None:
        if all_pct - recent_pct > 10:
            reasons.append({
                'type': '近期质量下滑',
                'text': f'近期评分({recent_pct}%)显著低于历史评分({all_pct}%)，游戏质量可能正在下滑或遭遇争议事件',
                'confidence': min(0.85, 0.4 + (all_pct - recent_pct) * 0.02),
                'direction': 'negative',
            })

    # 规则3: 持续改进
    if all_pct is not None and recent_pct is not None:
        if recent_pct - all_pct > 10:
            reasons.append({
                'type': '开发商持续改进',
                'text': f'近期评分({recent_pct}%)显著高于历史评分({all_pct}%)，开发商持续改进获得玩家认可',
                'confidence': min(0.80, 0.4 + (recent_pct - all_pct) * 0.02),
                'direction': 'positive',
            })

    # 规则4: 内购引发差评
    if game.get('HasInAppPurchases') == 1 and actual < prediction:
        reasons.append({
            'type': '内购引发差评',
            'text': '游戏包含内购系统，付费设计可能引发大量与游戏质量无关的差评',
            'confidence': 0.80 if actual < 50 else 0.55,
            'direction': 'negative',
        })

    # 规则5: 本地化问题
    if game.get('NumLanguages', 0) > 10 and actual < prediction:
        reasons.append({
            'type': '本地化不足',
            'text': f'支持{game.get("NumLanguages")}种语言但评分偏低，可能存在本地化质量参差不齐的问题',
            'confidence': 0.55,
            'direction': 'negative',
        })

    # 规则6: 抢先体验
    try:
        tags = ast.literal_eval(game.get('Popular Tags', '[]'))
    except Exception:
        tags = []
    if 'Early Access' in tags and actual < prediction:
        reasons.append({
            'type': '抢先体验阶段',
            'text': '游戏处于抢先体验(Early Access)阶段，内容不完整导致评分低于模型基于标签特征的预期',
            'confidence': 0.72,
            'direction': 'negative',
        })

    # 规则7: 定价过高
    price = game.get('OriginalPrice') or 0
    if price > 40 and actual < prediction:
        reasons.append({
            'type': '定价过高',
            'text': f'定价${price:.0f}偏高，价格与品质不匹配可能导致评分低于模型预期',
            'confidence': min(0.75, 0.4 + price * 0.005),
            'direction': 'negative',
        })

    # 规则8: 多人游戏社区问题
    if game.get('HasMultiplayer') == 1 and (game.get('RecentReviewPct') or 0) < 50:
        reasons.append({
            'type': '社区氛围问题',
            'text': '多人游戏的社区氛围、匹配机制或平衡性问题可能导致与游戏本身质量无关的差评',
            'confidence': 0.60,
            'direction': 'negative',
        })

    # 规则9: 大众市场突破
    review_count = game.get('AllReviewCount') or 0
    if review_count > 50000 and actual > prediction:
        reasons.append({
            'type': '大众市场突破',
            'text': f'拥有{int(review_count):,}条评论，大众市场接受度超出模型基于技术指标的预测',
            'confidence': 0.65,
            'direction': 'positive',
        })

    # 特征贡献分析
    try:
        X = pd.DataFrame([{c: game.get(c, 0) for c in FEATURE_COLS if c in game}])
        for c in FEATURE_COLS:
            if c not in X.columns:
                X[c] = 0
        X = X[FEATURE_COLS].fillna(0)
        contributions = {}
        for c in FEATURE_COLS:
            val = float(X[c].iloc[0]) if c in X.columns else 0
            imp = importances.get(c, 0)
            if abs(val) > 0.001:
                contributions[c] = round(val * imp * 100, 2)
        top_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
        feature_analysis = {
            'positive_factors': [{'feature': c, 'impact': v} for c, v in top_contrib if v > 0][:3],
            'negative_factors': [{'feature': c, 'impact': v} for c, v in top_contrib if v < 0][:3],
        }
    except Exception:
        feature_analysis = {'positive_factors': [], 'negative_factors': []}

    reasons.sort(key=lambda x: -x['confidence'])

    return {
        'has_gap': True,
        'gap_size': round(absolute_gap, 1),
        'direction': '模型低估' if direction == 'underestimated' else '模型高估',
        'actual': round(actual, 1),
        'predicted': round(prediction, 1),
        'reasons': reasons,
        'feature_analysis': feature_analysis,
    }


# 启动时初始化异常检测
try:
    init_anomaly_detection()
    print("[Init] Anomaly detection cache ready")
except Exception as e:
    print(f"[Init] Anomaly detection init failed (non-fatal): {e}")


# ====== 中英文翻译映射表 ======
TAG_TRANSLATION = {
    'Indie': '独立游戏', 'Singleplayer': '单人', 'Action': '动作', 'Adventure': '冒险',
    'Casual': '休闲', '2D': '2D', 'Strategy': '策略', 'Simulation': '模拟',
    '3D': '3D', 'Atmospheric': '氛围', 'Puzzle': '解谜', 'RPG': '角色扮演',
    'Colorful': '多彩', 'Pixel Graphics': '像素图形', 'Exploration': '探索',
    'Story Rich': '剧情丰富', 'Cute': '可爱', 'Fantasy': '奇幻',
    'First Person': '第一人称', 'First-Person': '第一人称', 'Arcade': '街机',
    'Multiplayer': '多人', 'Early Access': '抢先体验', 'Shooter': '射击',
    'Funny': '搞笑', 'Action Adventure': '动作冒险', 'Action-Adventure': '动作冒险',
    'Platformer': '平台跳跃', 'Retro': '复古', 'Sci-fi': '科幻', 'Sci_fi': '科幻',
    'Family Friendly': '家庭友好', 'Horror': '恐怖', 'FPS': '第一人称射击',
    'Open World': '开放世界', 'Survival': '生存', 'Racing': '竞速',
    'Co-op': '合作', 'Online Co-Op': '在线合作', 'PvP': '玩家对战',
    'Massively Multiplayer': '大型多人在线', 'Violent': '暴力', 'Gore': '血腥',
    'Sexual Content': '成人内容', 'Nudity': '裸露', 'Difficult': '高难度',
    'Rogue-like': '肉鸽', 'Roguelike': '肉鸽', 'Roguelite': '轻肉鸽',
    'Sandbox': '沙盒', 'Open World Survival Craft': '开放世界生存建造',
    'Building': '建造', 'Crafting': '制作', 'Farming': '农场', 'Farming Sim': '农场模拟',
    'Life Sim': '生活模拟', 'Dating Sim': '恋爱模拟', 'Visual Novel': '视觉小说',
    'Anime': '动漫', 'Mature': '成人', 'Dark': '黑暗', 'Dark Fantasy': '暗黑奇幻',
    'Comedy': '喜剧', 'Drama': '剧情', 'Mystery': '悬疑', 'Thriller': '惊悚',
    'Psychological Horror': '心理恐怖', 'Survival Horror': '生存恐怖',
    'Zombies': '丧尸', 'Post-apocalyptic': '后末日', 'Cyberpunk': '赛博朋克',
    'Steampunk': '蒸汽朋克', 'Space': '太空', 'Sci-fi': '科幻',
    'Aliens': '外星人', 'Robots': '机器人', 'Mechs': '机甲',
    'Turn-Based': '回合制', 'Turn-Based Strategy': '回合制策略',
    'Turn-Based Tactics': '回合制战术', 'Real-Time': '实时', 'Real Time Strategy': '实时策略',
    'Tower Defense': '塔防', 'Card Game': '卡牌', 'Board Game': '桌游',
    'Tabletop': '桌面游戏', 'Dungeon Crawler': '地牢探索', 'Metroidvania': '类银河城',
    'Souls-like': '类魂', 'Hack and Slash': '砍杀', "Beat 'em up": '清版动作',
    'Fighting': '格斗', 'Martial Arts': '武术', 'Stealth': '潜行',
    'Parkour': '跑酷', 'Rhythm': '音乐节奏', 'Music': '音乐',
    'Education': '教育', 'Software': '软件', 'Utilities': '工具',
    'Game Development': '游戏开发', 'Design & Illustration': '设计与绘图',
    'Animation & Modeling': '动画与建模', 'Video Production': '视频制作',
    'Audio Production': '音频制作', 'Photo Editing': '图片编辑',
    'Web Publishing': '网页发布', 'Software Training': '软件培训',
    'Physics': '物理', 'Space Sim': '太空模拟', 'Flight': '飞行',
    'Naval': '海战', 'Tanks': '坦克', 'War': '战争',
    'World War II': '二战', 'World War I': '一战', 'Modern': '现代',
    'Historical': '历史', 'Medieval': '中世纪', 'Rome': '古罗马',
    'America': '美国', 'Japan': '日本', 'China': '中国',
    'City Builder': '城市建设', 'Colony Sim': '殖民模拟', 'God Game': '上帝视角',
    'Capitalism': '资本主义', 'Economy': '经济', 'Management': '管理',
    'Inventory Management': '物品管理', 'Resource Management': '资源管理',
    'Base Building': '基地建设', 'Automation': '自动化',
    'Programming': '编程', 'Hacking': '黑客', 'Logic': '逻辑',
    'Clicker': '点击放置', 'Idler': '放置挂机', 'Casual': '休闲',
    'Minigames': '小游戏合集', 'Party Game': '聚会游戏', 'Party': '聚会',
    'Local Multiplayer': '本地多人', 'Local Co-Op': '本地合作',
    'Split Screen': '分屏', 'Cross-Platform Multiplayer': '跨平台多人',
    'MMORPG': '大型多人在线角色扮演', 'MOBA': '多人在线战术竞技',
    'Battle Royale': '大逃杀', 'Hero Shooter': '英雄射击',
    'Class-Based': '职业制', 'Team-Based': '团队制',
    'Competitive': '竞技', 'eSports': '电子竞技',
    'Tactical': '战术', 'Military': '军事', 'Sniper': '狙击',
    'Bullet Hell': '弹幕地狱', 'Twin Stick Shooter': '双摇杆射击',
    'Top-Down Shooter': '俯视射击', 'Side Scroller': '横版卷轴',
    'Runner': '跑酷', 'Precision Platformer': '精准平台',
    '3D Platformer': '3D平台', 'Collectathon': '收集',
    'CRPG': '经典角色扮演', 'JRPG': '日式角色扮演', 'Party-Based RPG': '队伍角色扮演',
    'Action RPG': '动作角色扮演', 'Strategy RPG': '策略角色扮演',
    'Immersive Sim': '沉浸式模拟', 'Walking Simulator': '步行模拟',
    'Interactive Fiction': '互动小说', 'Text-Based': '文字游戏',
    'Choose Your Own Adventure': '选择取向', 'Multiple Endings': '多结局',
    'Choices Matter': '选择重要', 'Nonlinear': '非线性',
    'Open World': '开放世界', 'Semi-Open World': '半开放世界',
    'Linear': '线性', 'Narration': '旁白', 'Cinematic': '电影化',
    'Quick-Time Events': '快速反应事件', 'FMV': '全动态影像',
    'Lore-Rich': '丰富世界观', 'Worldbuilding': '世界观构建',
    'Moddable': '支持MOD', 'Mod': 'MOD支持', 'Level Editor': '关卡编辑器',
    'VR': '虚拟现实', 'Tracked Controller Support': 'VR手柄支持',
    'Seated VR': '坐姿VR', 'Standing VR': '站立VR', 'Room Scale VR': '房间尺度VR',
    'Asymmetric VR': '非对称VR', 'Asynchronous Multiplayer': '异步多人',
    'Perma Death': '永久死亡', 'Procedural Generation': '程序生成',
    'Remake': '重制版', 'Remaster': '高清复刻', 'Classic': '经典',
    'Cult Classic': '邪典经典', 'Masterpiece': '神作', 'Experimental': '实验性',
    'Abstract': '抽象', 'Minimalist': '极简', 'Stylized': '风格化',
    'Hand-drawn': '手绘', 'Cartoon': '卡通', 'Cartoony': '卡通风格',
    'Realistic': '写实', 'Photorealistic': '照片级写实',
    'Cinematic': '电影化', 'Emotional': '情感', 'Relaxing': '放松',
    'Cozy': '温馨', 'Wholesome': '治愈', 'Cute': '可爱',
    'Psychedelic': '迷幻', 'Surreal': '超现实', 'Dystopian': '反乌托邦',
    'Dark Comedy': '黑色幽默', 'Satire': '讽刺', 'Parody': '恶搞',
    'Memes': '梗文化', 'Internet': '互联网', 'Social Media': '社交媒体',
    'Artificial Intelligence': '人工智能', 'Conversation': '对话', 'Diplomacy': '外交',
    'Politics': '政治', 'Philosophical': '哲学', 'Dog': '狗', 'Cats': '猫',
    'Dinosaurs': '恐龙', 'Dragons': '龙', 'Vampire': '吸血鬼',
    'Werewolves': '狼人', 'Magic': '魔法', 'Supernatural': '超自然',
    'Mythology': '神话', 'Greek Mythology': '希腊神话', 'Norse': '北欧',
    'Pirates': '海盗', 'Ninja': '忍者', 'Samurai': '武士',
    'Western': '西部', 'Noir': '黑色电影', 'Lovecraftian': '克苏鲁',
    'Old School': '老派风格', '1990s': '90年代', '1980s': '80年代',
    'Time Travel': '时间旅行', 'Time Manipulation': '时间操控',
    'Gravity': '重力', 'Underwater': '水下', 'Snow': '雪地',
    'Nature': '自然', 'Agriculture': '农业', 'Fishing': '钓鱼',
    'Hunting': '狩猎', 'Golf': '高尔夫', 'Bowling': '保龄球',
    'Pinball': '弹球', 'Gambling': '赌博', 'Casino': '赌场',
    'eBook': '电子书', 'Documentary': '纪录片',
}

FEATURE_TRANSLATION = {
    'HasInAppPurchases': '含内购', 'OriginalPrice': '原价(美元)',
    'NumLanguages': '语言数量', 'PubEncoded': '发行商编码',
    'ReleaseYear': '发行年份', 'NumFeatures': '功能数量',
    'ReleaseMonth': '发行月份', 'HasMultiplayer': '多人游戏',
    'NumTags': '标签数量', 'DevEncoded': '开发商编码',
    'DiscountRate': '折扣率', 'IsFree': '免费游戏',
    'IsOnSale': '促销中', 'HasSinglePlayer': '单人游戏',
    'HasControllerSupport': '手柄支持', 'HasAchievements': 'Steam成就',
    'HasTradingCards': 'Steam交易卡', 'HasWorkshop': '创意工坊',
    'SupportsChinese': '支持中文', 'RecentReviewPct': '近期评分',
    'RecentReviewCount': '近期评论数', 'AllReviewPct': '全量评分',
    'AllReviewCount': '全量评论数', 'AllSentiment': '全量情感',
    'RecentSentiment': '近期情感',
}

ANOMALY_REASON_TRANSLATION = {
    'overpriced_low_score': '高价低分异常',
    'free_game_very_high_score': '免费游戏极高评分',
    'expensive_high_score': '高价高评价',
    'cheap_low_score': '低价低评分',
    'price_score_mismatch': '价格评分不匹配',
    'extreme_popularity': '极高人气',
    'extreme_localization': '极多语言支持',
    'feature_dense': '功能密集',
    'unusual_pattern': '异常模式',
}

def translate_tag(tag_name):
    """翻译标签名，未知标签返回原文"""
    return TAG_TRANSLATION.get(tag_name, tag_name)

def translate_feature(feat_name):
    """翻译特征名，对Tag_前缀做特殊处理"""
    if feat_name.startswith('Tag_'):
        inner = feat_name[4:].replace('_', ' ')
        translated = TAG_TRANSLATION.get(inner, inner)
        return f'标签:{translated}'
    return FEATURE_TRANSLATION.get(feat_name, feat_name)

def translate_anomaly_reason(reason):
    """翻译异常原因"""
    return ANOMALY_REASON_TRANSLATION.get(reason, reason)


# ================================================================
#  Pages
# ================================================================

@app.route('/')
def index():
    """主页面"""
    return app.send_static_file('index.html')


# ================================================================
#  API: 统计概览
# ================================================================

@app.route('/api/stats')
def api_stats():
    """仪表板概览统计数据"""
    db = get_db()

    total = db.execute("SELECT COUNT(*) FROM games").fetchone()[0]

    # 评分分布
    score_avg = db.execute(
        "SELECT AVG(RecentReviewPct) FROM games WHERE RecentReviewPct IS NOT NULL"
    ).fetchone()[0] or 0

    # 免费 vs 付费
    free_count = db.execute("SELECT COUNT(*) FROM games WHERE IsFree=1").fetchone()[0]

    # 支持中文
    cn_count = db.execute("SELECT COUNT(*) FROM games WHERE SupportsChinese=1").fetchone()[0]

    # 平均价格
    avg_price = db.execute(
        "SELECT AVG(OriginalPrice) FROM games WHERE OriginalPrice > 0"
    ).fetchone()[0] or 0

    # 有内购的比例
    iap_count = db.execute("SELECT COUNT(*) FROM games WHERE HasInAppPurchases=1").fetchone()[0]

    # 好评/中评/差评分布
    pos = db.execute(
        "SELECT COUNT(*) FROM games WHERE RecentReviewPct >= 80 AND RecentReviewPct IS NOT NULL"
    ).fetchone()[0]
    mixed = db.execute(
        "SELECT COUNT(*) FROM games WHERE RecentReviewPct >= 50 AND RecentReviewPct < 80 AND RecentReviewPct IS NOT NULL"
    ).fetchone()[0]
    neg = db.execute(
        "SELECT COUNT(*) FROM games WHERE RecentReviewPct < 50 AND RecentReviewPct IS NOT NULL"
    ).fetchone()[0]

    # 年度趋势
    year_rows = db.execute(
        "SELECT ReleaseYear, COUNT(*) as cnt, AVG(AllReviewPct) as avg_score "
        "FROM games WHERE ReleaseYear >= 2010 AND ReleaseYear <= 2026 "
        "GROUP BY ReleaseYear ORDER BY ReleaseYear"
    ).fetchall()
    year_trend = [{'year': r['ReleaseYear'], 'count': r['cnt'],
                   'avg_score': round(r['avg_score'] or 0, 1)} for r in year_rows]

    # 价格分布（桶）
    price_bins = [
        {'label': 'Free', 'min': -0.01, 'max': 0.01},
        {'label': '$0-5', 'min': 0, 'max': 5},
        {'label': '$5-10', 'min': 5, 'max': 10},
        {'label': '$10-20', 'min': 10, 'max': 20},
        {'label': '$20-50', 'min': 20, 'max': 50},
        {'label': '$50+', 'min': 50, 'max': 999999},
    ]
    price_dist = []
    for b in price_bins:
        cnt = db.execute(
            "SELECT COUNT(*) FROM games WHERE OriginalPrice > ? AND OriginalPrice <= ?",
            (b['min'], b['max'])
        ).fetchone()[0]
        price_dist.append({'label': b['label'], 'count': cnt})

    # 热门标签
    tag_rows = db.execute("SELECT \"Popular Tags\" FROM games LIMIT 5000").fetchall()
    tag_counter = {}
    for r in tag_rows:
        try:
            tags = ast.literal_eval(r['Popular Tags'])
            for t in tags:
                tag_counter[t] = tag_counter.get(t, 0) + 1
        except:
            pass
    top_tags = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)[:20]

    return jsonify({
        'total_games': total,
        'avg_score': round(score_avg, 1),
        'free_games': free_count,
        'paid_games': total - free_count,
        'cn_support': cn_count,
        'cn_support_pct': round(cn_count / total * 100, 1),
        'avg_price': round(avg_price, 2),
        'iap_count': iap_count,
        'sentiment': {'positive': pos, 'mixed': mixed, 'negative': neg},
        'year_trend': year_trend,
        'price_distribution': price_dist,
        'top_tags': [{'name': t, 'name_cn': translate_tag(t), 'count': c} for t, c in top_tags],
    })


# ================================================================
#  API: 游戏列表（分页+搜索+筛选+排序）
# ================================================================

@app.route('/api/games')
def api_games():
    """
    游戏列表API
    Query params:
      page:     页码 (默认1)
      per_page: 每页数量 (默认20, 最大100)
      search:   搜索关键词 (搜索标题/开发商/发行商)
      sort:     排序字段 (score/price/year/title)
      order:    排序方向 (asc/desc)
      min_score: 最低评分
      max_price: 最高价格
      is_free:  是否免费 (1/0)
      tag:       标签筛选
      feature:   特征筛选 (singleplayer/multiplayer/achievements/controller/chinese)
    """
    db = get_db()
    page = max(1, request.args.get('page', 1, type=int))
    per_page = min(100, max(1, request.args.get('per_page', 20, type=int)))
    search = request.args.get('search', '').strip()
    sort = request.args.get('sort', 'score')
    order = request.args.get('order', 'desc')
    min_score = request.args.get('min_score', type=float)
    max_price = request.args.get('max_price', type=float)
    is_free = request.args.get('is_free', type=int)
    tag = request.args.get('tag', '').strip()
    feature = request.args.get('feature', '').strip()

    # 构建SQL
    where_clauses = ["RecentReviewPct IS NOT NULL"]
    params = []

    if search:
        where_clauses.append("(Title LIKE ? OR Developer LIKE ? OR Publisher LIKE ?)")
        like = f"%{search}%"
        params.extend([like, like, like])

    if min_score is not None:
        where_clauses.append("RecentReviewPct >= ?")
        params.append(min_score)

    if max_price is not None:
        where_clauses.append("OriginalPrice <= ?")
        params.append(max_price)

    if is_free == 1:
        where_clauses.append("IsFree = 1")
    elif is_free == 0:
        where_clauses.append("IsFree = 0")

    if feature == 'singleplayer':
        where_clauses.append("HasSinglePlayer = 1")
    elif feature == 'multiplayer':
        where_clauses.append("HasMultiplayer = 1")
    elif feature == 'achievements':
        where_clauses.append("HasAchievements = 1")
    elif feature == 'controller':
        where_clauses.append("HasControllerSupport = 1")
    elif feature == 'chinese':
        where_clauses.append("SupportsChinese = 1")

    if tag:
        where_clauses.append("\"Popular Tags\" LIKE ?")
        params.append(f"%'{tag}'%")

    where_sql = " AND ".join(where_clauses)

    # 排序
    sort_map = {
        'score': 'RecentReviewPct',
        'price': 'OriginalPrice',
        'year': 'ReleaseYear',
        'title': 'Title',
    }
    sort_col = sort_map.get(sort, 'RecentReviewPct')
    order_sql = f"{sort_col} {'DESC' if order == 'desc' else 'ASC'}"

    # 查询总数
    count_sql = f"SELECT COUNT(*) FROM games WHERE {where_sql}"
    total = db.execute(count_sql, params).fetchone()[0]

    # 查询数据
    offset = (page - 1) * per_page
    data_sql = f"""
        SELECT id, Title, OriginalPrice, DiscountedPrice, DiscountRate,
               IsFree, IsOnSale,
               RecentReviewPct, RecentReviewCount, RecentSentiment,
               AllReviewPct, AllReviewCount, AllSentiment,
               ReleaseYear, ReleaseMonth,
               Developer, Publisher,
               NumLanguages, SupportsChinese,
               HasSinglePlayer, HasMultiplayer, HasAchievements,
               HasTradingCards, HasInAppPurchases, HasWorkshop,
               NumFeatures, NumTags
        FROM games
        WHERE {where_sql}
        ORDER BY {order_sql}
        LIMIT ? OFFSET ?
    """
    rows = db.execute(data_sql, params + [per_page, offset]).fetchall()

    games = [dict(r) for r in rows]

    return jsonify({
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': max(1, (total + per_page - 1) // per_page),
        'games': games,
    })


# ================================================================
#  API: 游戏详情
# ================================================================

@app.route('/api/games/<int:game_id>')
def api_game_detail(game_id):
    """单个游戏详情"""
    db = get_db()
    row = db.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()

    if not row:
        return jsonify({'error': 'Game not found'}), 404

    game = dict(row)

    # 解析标签和特征为数组（方便前端渲染）
    try:
        game['PopularTags'] = ast.literal_eval(game['Popular Tags'])
    except:
        game['PopularTags'] = []
    try:
        game['GameFeatures'] = ast.literal_eval(game['Game Features'])
    except:
        game['GameFeatures'] = []
    try:
        game['SupportedLanguages'] = ast.literal_eval(game['Supported Languages'])
    except:
        game['SupportedLanguages'] = []

    # 获取同类游戏（同标签、评分相近）
    tag_list = game['PopularTags']
    if tag_list:
        first_tag = tag_list[0]
        similar_rows = db.execute(
            """SELECT id, Title, RecentReviewPct, OriginalPrice, Developer
               FROM games
               WHERE "Popular Tags" LIKE ? AND id != ?
               ORDER BY ABS(RecentReviewPct - ?) ASC
               LIMIT 6""",
            (f"%'{first_tag}'%", game_id, game['RecentReviewPct'] or 0)
        ).fetchall()
        similar = [dict(r) for r in similar_rows]
    else:
        similar = []

    # 预测该游戏的评分
    prediction = None
    try:
        X = pd.DataFrame([{
            c: game.get(c, 0) for c in FEATURE_COLS if c in game
        }])
        for c in FEATURE_COLS:
            if c not in X.columns:
                X[c] = 0
        # 处理Dev/Pub编码
        if 'DevEncoded' in FEATURE_COLS and 'DevEncoded' not in X.columns:
            X['DevEncoded'] = 0
        if 'PubEncoded' in FEATURE_COLS and 'PubEncoded' not in X.columns:
            X['PubEncoded'] = 0

        X = X[FEATURE_COLS].fillna(0)
        pred = model.predict(X)[0]
        prediction = round(float(pred), 1)
    except Exception as e:
        prediction = None

    return jsonify({
        'game': game,
        'similar': similar,
        'prediction': prediction,
        'gap_analysis': analyze_gap(game, prediction, IMPORTANCES) if prediction is not None else None,
    })


# ================================================================
#  API: 实时预测
# ================================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    根据用户输入的特征预测游戏评分

    请求体示例:
    {
        "OriginalPrice": 9.99,
        "IsFree": 0,
        "NumLanguages": 5,
        "HasSinglePlayer": 1,
        "HasMultiplayer": 0,
        "HasAchievements": 1,
        "SupportsChinese": 1,
        "tags": ["Action", "RPG", "Indie"]
    }
    """
    data = request.get_json() or {}

    # 构建特征向量
    X = pd.DataFrame([{
        'OriginalPrice': data.get('OriginalPrice', 0),
        'DiscountRate': data.get('DiscountRate', 0),
        'IsFree': 1 if data.get('IsFree') else 0,
        'IsOnSale': 1 if data.get('IsOnSale') else 0,
        'ReleaseYear': data.get('ReleaseYear', 2024),
        'ReleaseMonth': data.get('ReleaseMonth', 6),
        'NumFeatures': data.get('NumFeatures', 3),
        'NumTags': data.get('NumTags', 10),
        'NumLanguages': data.get('NumLanguages', 3),
        'HasSinglePlayer': 1 if data.get('HasSinglePlayer') else 0,
        'HasMultiplayer': 1 if data.get('HasMultiplayer') else 0,
        'HasControllerSupport': 1 if data.get('HasControllerSupport') else 0,
        'HasAchievements': 1 if data.get('HasAchievements') else 0,
        'HasTradingCards': 1 if data.get('HasTradingCards') else 0,
        'HasInAppPurchases': 1 if data.get('HasInAppPurchases') else 0,
        'HasWorkshop': 1 if data.get('HasWorkshop') else 0,
        'SupportsChinese': 1 if data.get('SupportsChinese') else 0,
    }])

    # 标签特征
    input_tags = data.get('tags', [])
    for c in FEATURE_COLS:
        if c.startswith('Tag_'):
            tag_name = c[4:].replace('_', ' ').replace('-', ' ')
            match = any(
                t.lower().replace(' ', '') == tag_name.lower().replace(' ', '')
                or t.lower() in tag_name.lower()
                or tag_name.lower() in t.lower()
                for t in input_tags
            )
            X[c] = 1 if match else 0

    # Dev/Pub encoded
    X['DevEncoded'] = 0
    X['PubEncoded'] = 0

    # 补齐缺失列
    for c in FEATURE_COLS:
        if c not in X.columns:
            X[c] = 0

    X = X[FEATURE_COLS].fillna(0)

    # 预测
    pred = model.predict(X)[0]
    pred = max(0, min(100, float(pred)))

    # 贡献最大的特征
    pred_series = pd.Series(X.iloc[0].to_dict())
    contributions = {}
    for c in FEATURE_COLS:
        val = pred_series.get(c, 0)
        imp = IMPORTANCES.get(c, 0)
        if abs(val) > 0.001:
            contributions[c] = round(val * imp * 100, 2)

    top_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:8]

    return jsonify({
        'predicted_score': round(pred, 1),
        'interpretation': 'Positive' if pred >= 70 else ('Mixed' if pred >= 50 else 'Negative'),
        'top_factors': [{'feature': c, 'feature_cn': translate_feature(c), 'impact': round(v, 2)} for c, v in top_contrib],
    })


# ================================================================
#  API: 特征重要性
# ================================================================

@app.route('/api/feature-importance')
def api_feature_importance():
    """返回模型的特征重要性"""
    sorted_imp = sorted(IMPORTANCES.items(), key=lambda x: x[1], reverse=True)
    top = sorted_imp[:15]
    return jsonify([{'feature': f, 'feature_cn': translate_feature(f), 'importance': round(i, 4)} for f, i in top])


# ================================================================
#  API: 可用标签列表
# ================================================================

@app.route('/api/tags')
def api_tags():
    """返回所有可用标签"""
    db = get_db()
    rows = db.execute("SELECT \"Popular Tags\" FROM games LIMIT 10000").fetchall()
    counter = {}
    for r in rows:
        try:
            tags = ast.literal_eval(r['Popular Tags'])
            for t in tags:
                counter[t] = counter.get(t, 0) + 1
        except:
            pass
    # 只返回出现>=10次的标签
    result = [{'name': t, 'name_cn': translate_tag(t), 'count': c}
               for t, c in sorted(counter.items(), key=lambda x: -x[1]) if c >= 10]
    return jsonify(result[:100])


# ================================================================
#  API: 评论时序历史（模拟数据）
# ================================================================

@app.route('/api/games/<int:game_id>/review-history')
def api_game_review_history(game_id):
    """返回游戏的模拟月度评论历史时序数据"""
    db = get_db()
    row = db.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    if not row:
        return jsonify({'error': 'Game not found'}), 404

    game = dict(row)
    months_param = min(120, max(6, request.args.get('months', 60, type=int)))

    history = generate_review_history(game, months=months_param,
                                      model=model, feature_cols=FEATURE_COLS)
    return jsonify(history)


# ================================================================
#  API: 异常数据分析
# ================================================================

@app.route('/api/anomalies')
def api_anomalies():
    """返回IsolationForest异常检测结果"""
    global ANOMALY_CACHE

    page = max(1, request.args.get('page', 1, type=int))
    per_page = min(50, max(1, request.args.get('per_page', 20, type=int)))
    sort = request.args.get('sort', 'anomaly_score')
    order = request.args.get('order', 'asc')
    reason_filter = request.args.get('reason', '').strip()

    if ANOMALY_CACHE is None:
        try:
            init_anomaly_detection()
        except Exception:
            return jsonify({'error': 'Anomaly detection not available'}), 500

    anom_df = ANOMALY_CACHE['anomalies'].copy()

    if reason_filter:
        anom_df = anom_df[anom_df['reasons'].apply(lambda r: reason_filter in r)]

    sort_map = {
        'anomaly_score': 'anomaly_score',
        'score': 'RecentReviewPct',
        'price': 'OriginalPrice',
        'title': 'Title',
    }
    sort_col = sort_map.get(sort, 'anomaly_score')
    ascending = order == 'asc'
    anom_df = anom_df.sort_values(sort_col, ascending=ascending)

    total = len(anom_df)
    start = (page - 1) * per_page
    page_df = anom_df.iloc[start:start + per_page]

    anomaly_games = []
    for _, r in page_df.iterrows():
        anomaly_games.append({
            'id': int(r['id']),
            'Title': str(r.get('Title', '')),
            'OriginalPrice': float(r.get('OriginalPrice', 0)),
            'RecentReviewPct': float(r.get('RecentReviewPct', 0)),
            'RecentReviewCount': int(r.get('RecentReviewCount', 0)),
            'ReleaseYear': int(r.get('ReleaseYear', 0)) if pd.notna(r.get('ReleaseYear')) else None,
            'NumFeatures': int(r.get('NumFeatures', 0)),
            'NumTags': int(r.get('NumTags', 0)),
            'anomaly_score': round(float(r['anomaly_score']), 4),
            'reasons': r['reasons'] if isinstance(r['reasons'], list) else [r['reasons']],
        })

    # 散点图数据（采样1000点+所有异常点）
    df = ANOMALY_CACHE['df']
    normal_sample = df[df['is_anomaly'] == 0].sample(min(1000, len(df[df['is_anomaly'] == 0])), random_state=42)
    scatter_data = []
    for _, r in normal_sample.iterrows():
        scatter_data.append({
            'x': float(r['OriginalPrice']),
            'y': float(r['RecentReviewPct']),
            'is_anomaly': False,
            'title': str(r.get('Title', '')),
            'id': int(r['id']),
        })
    for _, r in anom_df.iterrows():
        scatter_data.append({
            'x': float(r['OriginalPrice']),
            'y': float(r['RecentReviewPct']),
            'is_anomaly': True,
            'title': str(r.get('Title', '')),
            'id': int(r['id']),
            'reasons': r['reasons'] if isinstance(r['reasons'], list) else [r['reasons']],
        })

    # 箱线图数据
    def get_quartiles(series):
        s = series.dropna()
        return [float(s.min()), float(s.quantile(0.25)), float(s.median()),
                float(s.quantile(0.75)), float(s.max())]

    normal_df = df[df['is_anomaly'] == 0]
    boxplot_data = {
        'price_normal': get_quartiles(normal_df['OriginalPrice']),
        'price_anomaly': get_quartiles(anom_df['OriginalPrice']),
        'score_normal': get_quartiles(normal_df['RecentReviewPct']),
        'score_anomaly': get_quartiles(anom_df['RecentReviewPct']),
        'score_by_iap': {
            'with_iap': get_quartiles(df[df['HasInAppPurchases'] == 1]['RecentReviewPct']),
            'without_iap': get_quartiles(df[df['HasInAppPurchases'] == 0]['RecentReviewPct']),
        },
        'score_by_free': {
            'free': get_quartiles(df[df['IsFree'] == 1]['RecentReviewPct']),
            'paid': get_quartiles(df[df['IsFree'] == 0]['RecentReviewPct']),
        },
    }

    # 异常原因统计
    reason_counts = {}
    for _, r in anom_df.iterrows():
        for reason in (r['reasons'] if isinstance(r['reasons'], list) else [r['reasons']]):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    top_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])[:8]

    return jsonify({
        'total_anomalies': ANOMALY_CACHE['anomaly_count'],
        'total_games': ANOMALY_CACHE['total_games'],
        'anomaly_rate': ANOMALY_CACHE['anomaly_pct'],
        'anomaly_stats': {
            'avg_score_anomaly': ANOMALY_CACHE['anomaly_avg_score'],
            'avg_score_normal': ANOMALY_CACHE['normal_avg_score'],
            'avg_price_anomaly': ANOMALY_CACHE['anomaly_avg_price'],
            'avg_price_normal': ANOMALY_CACHE['normal_avg_price'],
        },
        'anomaly_games': anomaly_games,
        'scatter_data': scatter_data,
        'boxplot_data': boxplot_data,
        'reason_distribution': [{'reason': r, 'count': c} for r, c in top_reasons],
        'page': page,
        'per_page': per_page,
        'total_pages': max(1, (total + per_page - 1) // per_page),
        'total_filtered': total,
    })


# ================================================================
#  API: 游戏快速信息（用于DeepSeek上下文）
# ================================================================

@app.route('/api/game-quick-info/<int:game_id>')
def api_game_quick_info(game_id):
    """返回游戏的简要信息，用于AI聊天上下文"""
    db = get_db()
    row = db.execute(
        "SELECT id, Title, Developer, Publisher, ReleaseYear, "
        "RecentReviewPct, AllReviewPct, RecentReviewCount, AllReviewCount, "
        "OriginalPrice, IsFree, NumLanguages, SupportsChinese, NumTags, "
        "\"Popular Tags\" "
        "FROM games WHERE id = ?", (game_id,)
    ).fetchone()

    if not row:
        return jsonify({'error': 'Game not found'}), 404

    game = dict(row)
    try:
        tags = ast.literal_eval(game.pop('Popular Tags', '[]'))
    except Exception:
        tags = []

    # 模型预测
    try:
        row_full = db.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
        game_full = dict(row_full)
        X = pd.DataFrame([{c: game_full.get(c, 0) for c in FEATURE_COLS if c in game_full}])
        for c in FEATURE_COLS:
            if c not in X.columns:
                X[c] = 0
        X = X[FEATURE_COLS].fillna(0)
        pred = round(float(model.predict(X)[0]), 1)
    except Exception:
        pred = None

    return jsonify({
        'title': game['Title'],
        'developer': game.get('Developer', ''),
        'publisher': game.get('Publisher', ''),
        'release_year': game.get('ReleaseYear'),
        'all_review_pct': game.get('AllReviewPct'),
        'recent_review_pct': game.get('RecentReviewPct'),
        'recent_review_count': game.get('RecentReviewCount'),
        'all_review_count': game.get('AllReviewCount'),
        'original_price': game.get('OriginalPrice'),
        'is_free': bool(game.get('IsFree')),
        'num_languages': game.get('NumLanguages'),
        'supports_chinese': bool(game.get('SupportsChinese')),
        'num_tags': game.get('NumTags'),
        'top_tags': tags[:8] if tags else [],
        'predicted_score': pred,
    })


# ================================================================
#  Start
# ================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Steam Game Analytics Server")
    print("  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
