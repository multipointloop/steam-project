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
import pickle
import sqlite3
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, g

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
            import ast
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
        'top_tags': [{'name': t, 'count': c} for t, c in top_tags],
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
    import ast
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
        'top_factors': [{'feature': c, 'impact': round(v, 2)} for c, v in top_contrib],
    })


# ================================================================
#  API: 特征重要性
# ================================================================

@app.route('/api/feature-importance')
def api_feature_importance():
    """返回模型的特征重要性"""
    sorted_imp = sorted(IMPORTANCES.items(), key=lambda x: x[1], reverse=True)
    top = sorted_imp[:15]
    return jsonify([{'feature': f, 'importance': round(i, 4)} for f, i in top])


# ================================================================
#  API: 可用标签列表
# ================================================================

@app.route('/api/tags')
def api_tags():
    """返回所有可用标签"""
    db = get_db()
    rows = db.execute("SELECT \"Popular Tags\" FROM games LIMIT 10000").fetchall()
    import ast
    counter = {}
    for r in rows:
        try:
            tags = ast.literal_eval(r['Popular Tags'])
            for t in tags:
                counter[t] = counter.get(t, 0) + 1
        except:
            pass
    # 只返回出现>=10次的标签
    result = [{'name': t, 'count': c} for t, c in sorted(counter.items(), key=lambda x: -x[1]) if c >= 10]
    return jsonify(result[:100])


# ================================================================
#  Start
# ================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Steam Game Analytics Server")
    print("  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
