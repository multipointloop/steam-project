"""
=============================================================================
模块4：特征分析与建模预测 (Feature Analysis & ML Modeling)
=============================================================================
功能：
1. 特征相关性分析（强弱分析，建立明显特征集）
2. 趋势预测（根据特征集预测游戏评分/受欢迎程度）
3. 回测验证（对预测准确率进行评估）
4. 可视化（将预测与真实值进行对比展示）
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互模式，直接保存图片
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, classification_report, confusion_matrix)
from datetime import datetime

CLEAN_CSV = "data/steam_games_clean.csv"
REPORT_DIR = "reports"
FIGURE_DIR = "reports/figures"
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


def load_clean_data():
    """加载清洗后的数据"""
    print(f"[加载] {CLEAN_CSV}")
    df = pd.read_csv(CLEAN_CSV, encoding='utf-8-sig')
    # 解析日期
    df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
    print(f"  记录数: {len(df)}, 字段数: {len(df.columns)}")
    return df


# ================================================================
#  Part 1: 特征相关性分析 (Feature Analysis)
# ================================================================

def feature_analysis(df):
    """特征强弱分析——找出与目标变量最相关的特征"""
    print("\n" + "=" * 60)
    print("Part 1: 特征相关性分析")
    print("=" * 60)

    # 选择要参与分析的数值特征
    feature_cols = [c for c in df.columns if c not in [
        'Title', 'Developer', 'Publisher', 'Link', 'ReleaseDate',
        'RecentSentiment', 'AllSentiment',  # 这些是目标变量
    ]]

    # 以 AllReviewPct 为目标（全量评分更稳定）
    target = 'AllReviewPct'
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # 计算相关系数矩阵
    corr_with_target = df[numeric_cols].corr()[target].drop(target).sort_values(key=abs, ascending=False)

    print(f"\n  与'{target}'(全量评分) 的相关系数 (Top 15):")
    print(f"  {'特征':<35s} {'相关系数':>10s}")
    print(f"  {'-' * 45}")
    for col, val in corr_with_target.head(15).items():
        print(f"  {col:<35s} {val:>10.4f}")

    # 保存Top特征列表（后续建模用）
    top_features = corr_with_target.head(15).index.tolist()

    # 绘制相关性热力图
    fig, ax = plt.subplots(figsize=(14, 10))
    top_corr_cols = [target] + corr_with_target.head(12).index.tolist()
    corr_matrix = df[top_corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap (Top Features vs Target)', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'correlation_heatmap.png'), dpi=120)
    plt.close(fig)
    print(f"\n  [图表已保存] correlation_heatmap.png")

    return top_features


# ================================================================
#  Part 2: 分布可视化
# ================================================================

def distribution_plots(df):
    """关键变量的分布图"""
    print("\n" + "=" * 60)
    print("Part 2: 数据分布可视化")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. 评分分布
    ax = axes[0, 0]
    df['AllReviewPct'].dropna().hist(bins=50, color='steelblue', edgecolor='white', ax=ax)
    ax.axvline(df['AllReviewPct'].median(), color='red', linestyle='--', label=f"Median: {df['AllReviewPct'].median():.0f}%")
    ax.set_title('All Review Score Distribution')
    ax.set_xlabel('Positive Review %')
    ax.legend()

    # 2. 价格分布（排除免费游戏和极端值）
    ax = axes[0, 1]
    paid = df[df['OriginalPrice'] > 0]['OriginalPrice']
    paid = paid[paid <= paid.quantile(0.95)]  # 去95分位以上极端值
    paid.hist(bins=50, color='coral', edgecolor='white', ax=ax)
    ax.set_title(f'Game Price Distribution\n(Paid games, median=${paid.median():.2f})')
    ax.set_xlabel('Price (USD)')

    # 3. 免费 vs 付费
    ax = axes[0, 2]
    counts = df['IsFree'].value_counts()
    colors = ['#ff7f0e', '#1f77b4']
    ax.pie(counts.values, labels=['Paid', 'Free'], autopct='%1.1f%%',
           colors=colors, startangle=90, explode=(0, 0.05))
    ax.set_title('Free vs Paid Games')

    # 4. 发行年份趋势
    ax = axes[1, 0]
    year_stats = df[df['ReleaseYear'] >= 2010].groupby('ReleaseYear').agg(
        count=('Title', 'count'),
        avg_score=('AllReviewPct', 'mean')
    ).reset_index()
    ax2 = ax.twinx()
    ax.bar(year_stats['ReleaseYear'], year_stats['count'], color='lightblue', alpha=0.7)
    ax2.plot(year_stats['ReleaseYear'], year_stats['avg_score'], 'r-o', linewidth=2)
    ax.set_xlabel('Release Year')
    ax.set_ylabel('Game Count', color='blue')
    ax2.set_ylabel('Avg Review Score (%)', color='red')
    ax.set_title('Games & Avg Score by Year')

    # 5. 特征重要性（简单版）
    ax = axes[1, 1]
    feature_impact = {}
    for feat in ['HasSinglePlayer', 'HasMultiplayer', 'HasAchievements',
                 'HasControllerSupport', 'HasTradingCards', 'HasWorkshop',
                 'HasInAppPurchases', 'SupportsChinese']:
        with_feat = df[df[feat] == 1]['AllReviewPct'].mean()
        without_feat = df[df[feat] == 0]['AllReviewPct'].mean()
        feature_impact[feat.replace('Has', '').replace('Supports', '')] = with_feat - without_feat

    colors_bar = ['green' if v > 0 else 'red' for v in feature_impact.values()]
    ax.barh(list(feature_impact.keys()), list(feature_impact.values()), color=colors_bar)
    ax.set_xlabel('Score Difference (with - without)')
    ax.set_title('Feature Impact on Review Score')
    ax.axvline(0, color='black', linewidth=0.5)

    # 6. 折扣分布
    ax = axes[1, 2]
    on_sale = df[df['IsOnSale'] == 1]['DiscountRate']
    on_sale.hist(bins=40, color='orange', edgecolor='white', alpha=0.8, ax=ax)
    ax.set_title(f'Discount Distribution\n({len(on_sale)} games on sale)')
    ax.set_xlabel('Discount Rate (%)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'distribution_dashboard.png'), dpi=120)
    plt.close(fig)
    print("  [图表已保存] distribution_dashboard.png")


# ================================================================
#  Part 3: 趋势预测建模 (ML Prediction)
# ================================================================

def prepare_model_data(df):
    """准备建模数据：特征矩阵X和目标变量y"""
    print("\n" + "=" * 60)
    print("Part 3: 预测模型构建")
    print("=" * 60)

    # 特征列
    feature_cols = [c for c in df.columns if c.startswith('Tag_')]
    feature_cols += [
        'OriginalPrice', 'DiscountRate', 'IsFree', 'IsOnSale',
        'ReleaseYear', 'ReleaseMonth',
        'NumFeatures', 'NumTags', 'NumLanguages',
        'HasSinglePlayer', 'HasMultiplayer', 'HasControllerSupport',
        'HasAchievements', 'HasTradingCards', 'HasInAppPurchases', 'HasWorkshop',
        'SupportsChinese',
    ]

    # 只保留存在的列
    feature_cols = [c for c in feature_cols if c in df.columns]

    # ====== 目标1：评分回归（预测 AllReviewPct） ======
    print("\n  [目标1] 回归任务 - 预测游戏评分百分比")

    reg_df = df.dropna(subset=['AllReviewPct'] + feature_cols).copy()
    print(f"    可用样本: {len(reg_df)}")

    X_reg = reg_df[feature_cols].fillna(0)
    y_reg = reg_df['AllReviewPct']

    # ====== 目标2：评分分类（预测好评/中评/差评） ======
    print("\n  [目标2] 分类任务 - 预测游戏评价等级")

    def classify_score(pct):
        if pd.isna(pct):
            return np.nan
        if pct >= 80:
            return 2  # 好评
        elif pct >= 50:
            return 1  # 中评
        else:
            return 0  # 差评

    df['ScoreClass'] = df['RecentReviewPct'].apply(classify_score)
    cls_df = df.dropna(subset=['ScoreClass'] + feature_cols).copy()
    print(f"    可用样本: {len(cls_df)}")
    print(f"    类别分布: 好评={ (cls_df['ScoreClass']==2).sum()}, "
          f"中评={ (cls_df['ScoreClass']==1).sum()}, "
          f"差评={ (cls_df['ScoreClass']==0).sum()}")

    X_cls = cls_df[feature_cols].fillna(0)
    y_cls = cls_df['ScoreClass']

    return X_reg, y_reg, X_cls, y_cls, feature_cols


def train_regression_models(X, y):
    """回归模型训练与评估"""
    print("\n--- 回归模型 ---")

    # 训练/测试分割 (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 标准化（对线性模型很重要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (L2)': Ridge(alpha=1.0),
        'Lasso (L1)': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    }

    results = []
    best_model = None
    best_score = -np.inf

    for name, model in models.items():
        # 线性模型用标准化数据，树模型用原始数据
        if any(k in name for k in ['Linear', 'Ridge', 'Lasso']):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # 交叉验证R2
        cv_scores = cross_val_score(model, X_train_scaled if any(k in name for k in ['Linear', 'Ridge', 'Lasso']) else X_train, y_train, cv=5, scoring='r2')

        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_R2_mean': cv_scores.mean(),
            'CV_R2_std': cv_scores.std(),
        })

        print(f"  {name:20s} | MAE={mae:6.2f} | RMSE={rmse:6.2f} | R2={r2:.4f} | CV_R2={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

        if r2 > best_score:
            best_score = r2
            best_model = (name, model, scaler if any(k in name for k in ['Linear', 'Ridge', 'Lasso']) else None)

    results_df = pd.DataFrame(results)
    print(f"\n  最佳模型: {best_model[0]} (R2={best_score:.4f})")

    return results_df, best_model, (X_test, y_test)


def train_classification_models(X, y):
    """分类模型训练与评估"""
    print("\n--- 分类模型 ---")

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    }

    results = []

    for name, model in models.items():
        if 'Logistic' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_scaled if 'Logistic' in name else X_train, y_train, cv=5, scoring='accuracy')

        print(f"  {name:25s} | Accuracy={acc:.4f} | CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

        results.append({
            'Model': name,
            'Accuracy': acc,
            'CV_Accuracy_mean': cv_scores.mean(),
            'CV_Accuracy_std': cv_scores.std(),
        })

    results_df = pd.DataFrame(results)
    return results_df


# ================================================================
#  Part 4: 回测验证与可视化
# ================================================================

def backtest_and_visualize(best_model_info, X_test, y_test, feature_names):
    """回测验证 + 预测vs真实值可视化"""
    print("\n" + "=" * 60)
    print("Part 4: 回测验证")
    print("=" * 60)

    name, model, scaler = best_model_info

    # 预测
    if scaler is not None:
        X_test_processed = scaler.transform(X_test)
    else:
        X_test_processed = X_test

    y_pred = model.predict(X_test_processed)

    # === 评估指标 ===
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 自定义准确率：预测误差在10%以内算"准确"
    within_10pct = (np.abs(y_test - y_pred) <= 10).mean() * 100

    print(f"\n  回归评估结果:")
    print(f"    平均绝对误差 (MAE):    {mae:.2f}%")
    print(f"    均方根误差 (RMSE):     {rmse:.2f}%")
    print(f"    R2 决定系数:            {r2:.4f}")
    print(f"    误差<=10%的预测占比:    {within_10pct:.1f}%")

    # === 回测可视化 ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 预测 vs 真实值散点图
    ax = axes[0, 0]
    sample_size = min(2000, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    ax.scatter(y_test.iloc[indices], y_pred[indices], alpha=0.3, s=10, c='steelblue')
    ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('True Score (%)')
    ax.set_ylabel('Predicted Score (%)')
    ax.set_title(f'Predicted vs True Review Score\nR2={r2:.4f}, MAE={mae:.2f}%')
    ax.legend()
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)

    # 2. 残差分布
    ax = axes[0, 1]
    residuals = y_test.values - y_pred
    ax.hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(residuals.mean(), color='orange', linestyle='--', linewidth=1,
               label=f'Mean Residual: {residuals.mean():.2f}')
    ax.set_xlabel('Residual (True - Predicted)')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution')
    ax.legend()

    # 3. 特征重要性（Random Forest或GB）
    ax = axes[1, 0]
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top_importances = importances.sort_values(ascending=False).head(15)
        ax.barh(range(len(top_importances)), top_importances.values[::-1])
        ax.set_yticks(range(len(top_importances)))
        ax.set_yticklabels(top_importances.index[::-1], fontsize=9)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top 15 Feature Importances ({name})')

    # 4. 分价格区间的模型表现
    ax = axes[1, 1]
    if 'OriginalPrice' in feature_names:
        price_idx = feature_names.index('OriginalPrice')
        X_test_df = pd.DataFrame(X_test_processed if scaler is not None else X_test.values, columns=feature_names)
        prices = X_test_df['OriginalPrice']
        # 分桶
        bins = [0, 5, 10, 20, 50, 100, 1000]
        labels = ['Free/$0-5', '$5-10', '$10-20', '$20-50', '$50-100', '$100+']
        X_test_df['PriceBucket'] = pd.cut(prices, bins=bins, labels=labels)
        test_results = X_test_df.copy()
        test_results['error'] = np.abs(y_test.values - y_pred)
        bucket_mae = test_results.groupby('PriceBucket')['error'].mean()
        ax.bar(range(len(bucket_mae)), bucket_mae.values, color='steelblue')
        ax.set_xticks(range(len(bucket_mae)))
        ax.set_xticklabels(bucket_mae.index, rotation=45, fontsize=8)
        ax.set_xlabel('Price Range')
        ax.set_ylabel('MAE (%)')
        ax.set_title('Model Performance by Price Range')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'backtest_validation.png'), dpi=120)
    plt.close(fig)
    print(f"  [图表已保存] backtest_validation.png")

    return mae, rmse, r2


# ================================================================
#  Part 5: 额外加分功能 - 异常检测
# ================================================================

def anomaly_detection(df):
    """
    智能异常检测：
    - 识别评分与价格/特征严重不匹配的游戏
    - 识别可能的"刷评分"或数据异常
    """
    print("\n" + "=" * 60)
    print("Part 5 (Bonus): 智能异常检测")
    print("=" * 60)

    # 简单规则：价格极低但评分极高？或者反之
    # 使用Z-score方法检测多变量异常

    from sklearn.ensemble import IsolationForest

    # 选择用于异常检测的特征
    anomaly_cols = ['OriginalPrice', 'NumFeatures', 'NumTags', 'NumLanguages',
                    'RecentReviewPct', 'ReleaseYear']
    anomaly_cols = [c for c in anomaly_cols if c in df.columns]
    anomaly_df = df[anomaly_cols].dropna().copy()

    # Isolation Forest（无监督异常检测）
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    X_anomaly = anomaly_df.fillna(0).copy()
    anomaly_df['anomaly'] = iso.fit_predict(X_anomaly)

    anomalies = anomaly_df[anomaly_df['anomaly'] == -1]
    normal = anomaly_df[anomaly_df['anomaly'] == 1]

    print(f"  检测到异常样本: {len(anomalies)} / {len(anomaly_df)} ({len(anomalies)/len(anomaly_df)*100:.1f}%)")
    print(f"  异常样本平均评分: {anomalies['RecentReviewPct'].mean():.1f}% (整体: {normal['RecentReviewPct'].mean():.1f}%)")
    print(f"  异常样本平均价格: ${anomalies['OriginalPrice'].mean():.2f} (整体: ${normal['OriginalPrice'].mean():.2f})")

    # 可视化：异常样本在评分-价格空间的分布
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(normal['OriginalPrice'], normal['RecentReviewPct'],
               c='steelblue', alpha=0.3, s=10, label='Normal')
    ax.scatter(anomalies['OriginalPrice'], anomalies['RecentReviewPct'],
               c='red', alpha=0.6, s=25, label='Anomaly', edgecolors='darkred')
    ax.set_xlabel('Original Price (USD)')
    ax.set_ylabel('Recent Review Score (%)')
    ax.set_title('Anomaly Detection: Review Score vs Price')
    ax.legend()
    ax.set_xlim(0, anomaly_df['OriginalPrice'].quantile(0.99))
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'anomaly_detection.png'), dpi=120)
    plt.close(fig)
    print("  [图表已保存] anomaly_detection.png")


# ================================================================
#  Main
# ================================================================

def main():
    print("\n" + "█" * 60)
    print("█  模块4: 特征分析 + 预测建模 + 回测验证 + 可视化")
    print("█" * 60)

    df = load_clean_data()

    # Part 1: 特征分析
    top_features = feature_analysis(df)

    # Part 2: 分布可视化
    distribution_plots(df)

    # Part 3: 建模
    X_reg, y_reg, X_cls, y_cls, feature_names = prepare_model_data(df)

    # 回归预测
    reg_results, best_model_info, test_data = train_regression_models(X_reg, y_reg)
    X_test, y_test = test_data

    # 分类预测
    cls_results = train_classification_models(X_cls, y_cls)

    # Part 4: 回测验证 + 可视化
    mae, rmse, r2 = backtest_and_visualize(best_model_info, X_test, y_test, feature_names)

    # Part 5 (Bonus): 异常检测
    anomaly_detection(df)

    # ====== 保存报告 ======
    report_path = os.path.join(REPORT_DIR, 'model_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Steam Games Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Dataset: {len(df)} games after cleaning\n")
        f.write(f"Features used: {len(feature_names)}\n\n")
        f.write(f"--- Regression Results ---\n")
        f.write(f"Model: {best_model_info[0]}\n")
        f.write(f"MAE: {mae:.2f}%\n")
        f.write(f"RMSE: {rmse:.2f}%\n")
        f.write(f"R2: {r2:.4f}\n")
        f.write(f"\n--- Model Details ---\n")
        f.write(reg_results.to_string())
        f.write(f"\n\n--- Classification Results ---\n")
        f.write(cls_results.to_string())

    print(f"\n{'=' * 60}")
    print(f"[完成] 报告已保存至: {report_path}")
    print(f"[完成] 图表已保存至: {FIGURE_DIR}/")
    print(f"  - correlation_heatmap.png (特征相关性热力图)")
    print(f"  - distribution_dashboard.png (数据分布仪表板)")
    print(f"  - backtest_validation.png (回测验证图)")
    print(f"  - anomaly_detection.png (异常检测图)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
