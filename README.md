# 🎮 Steam Game Analytics

> 2026年实践学期项目 — 基于机器学习的Steam游戏评分预测与可视化分析平台

## 项目简介

本项目从Kaggle获取Steam游戏数据集（7万+款游戏），完成**数据采集→清洗→特征工程→机器学习建模→回测验证→Web可视化→异常检测→偏差分析**的完整数据科学流水线。最终以Web应用形式呈现，支持游戏列表浏览、AI实时预测评分、交互式数据分析看板、异常数据深度分析、DeepSeek AI助手等功能。

## 技术栈

| 层级 | 技术 |
|------|------|
| 数据获取 | KaggleHub (Kaggle API) |
| 数据处理 | Pandas, NumPy |
| 机器学习 | Scikit-learn (Random Forest, Gradient Boosting, IsolationForest) |
| 后端 | Flask (REST API, 10个端点) |
| 前端 | Vanilla JS + Chart.js 4.4.0 |
| 数据库 | SQLite |
| 可视化 | Matplotlib, Seaborn, Chart.js |
| 设计风格 | 玻璃拟态(Glassmorphism) + 深色主题 + CSS动画 |
| AI集成 | DeepSeek 智能助手浮动组件 |

## 项目结构

```
steam_project/
├── data_acquisition.py      # 模块1: 数据获取（从Kaggle下载）
├── data_exploration.py      # 模块2: 数据探索（EDA）
├── data_cleaning.py         # 模块3: 数据清洗与特征提取
├── train_model.py           # 模型训练 + SQLite数据库构建
├── modeling.py              # 模块4: 特征分析+建模+回测+可视化
├── app.py                   # Flask后端（10个API端点）
├── index.html               # 前端页面（单文件完整应用, ~58KB）
├── generate_report.py       # 第一周工作报告生成
├── setup.bat                # 一键环境安装脚本
├── .gitignore
└── README.md
```

## 快速开始

### 环境要求

- Python 3.9+
- Windows / macOS / Linux

### 安装与运行

```bash
# 1. 安装依赖
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn flask python-docx

# 2. 下载数据并训练模型
python data_acquisition.py  # 从Kaggle下载数据
python data_cleaning.py     # 清洗数据
python train_model.py       # 训练模型+建库

# 3. 启动服务器
python app.py

# 4. 浏览器打开
# http://localhost:5000
```

## Web应用功能

### 五大功能标签

| 标签页 | 功能描述 |
|--------|----------|
| 📊 **游戏总览** | 统计卡片 + 可搜索/排序/筛选/分页的游戏列表，点击进入详情 |
| 🔬 **自由预测** | 调节参数（价格/语言/标签/特征），AI实时预测游戏好评率，展示关键影响因素 |
| 📈 **数据分析** | 评分饼图、价格柱状图、年度趋势双轴图、热门标签Top15横条图 |
| ⚠️ **异常分析** | IsolationForest异常检测（1,869个异常样本），含对数散点图、异常原因分布、箱线图、异常游戏列表 |
| 💬 **AI助手** | DeepSeek智能聊天浮动组件（浮动按钮+快捷提问+新标签页打开） |

### 游戏详情页（全新增强）

- **基本信息**：开发商/发行商/发行日期/价格/语言支持
- **评分表盘**：大号评分百分比，颜色随评分等级变化
- **游戏特征**：8种特征开关（单人/多人/成就/手柄/交易卡/创意工坊/内购/中文）
- **AI预测评分**：基于49个特征的模型预测，展示与实际评分的误差
- **📈 评论趋势分析**（新增）：堆叠柱状图(好评/差评) + 评分趋势折线图(含模型预测延展)
- **🔍 预测差异分析**（新增）：当|预测-实际|>15%时，自动分析9种可能原因（经典老游戏效应/质量下滑/内购差评/本地化问题/抢先体验/定价过高/社区氛围/大众市场突破等）
- **标签云**：全部标签展示（支持中英文翻译）
- **同类游戏推荐**：基于共享标签的相似游戏

### API端点（共10个）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 主页面 |
| GET | `/api/stats` | 全局统计概览（含中文标签名） |
| GET | `/api/games?page=&sort=&search=` | 游戏列表（分页+搜索+排序+筛选） |
| GET | `/api/games/<id>` | 游戏详情+AI预测+同类推荐+偏差分析 |
| GET | `/api/games/<id>/review-history` | 🆕 游戏评论历史时序数据（S曲线模拟） |
| POST | `/api/predict` | 实时评分预测（含中文关键影响因素） |
| GET | `/api/feature-importance` | 模型特征重要性（含中文特征名） |
| GET | `/api/tags` | 可用标签列表（含中文翻译） |
| GET | `/api/anomalies` | 🆕 异常数据分析（IsolationForest + 散点图/箱线图数据） |
| GET | `/api/game-quick-info/<id>` | 🆕 游戏快速信息（供AI助手上下文） |

## 模型性能

| 任务 | 模型 | 指标 |
|------|------|------|
| 评分回归 | Random Forest (100棵树, max_depth=15) | MAE=7.48%, R²=0.25, 误差≤10%占比77.5% |
| 评分分类 | Gradient Boosting | Accuracy=62.18% (3分类) |
| 异常检测 | IsolationForest (contamination=0.05) | 检出1,869个异常样本 (5.0%) |

### Top 10 特征重要性

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | 含内购 (HasInAppPurchases) | 8.21% |
| 2 | 原价 (OriginalPrice) | 7.71% |
| 3 | 语言数量 (NumLanguages) | 6.76% |
| 4 | 发行商编码 (PubEncoded) | 6.50% |
| 5 | 发行年份 (ReleaseYear) | 6.26% |
| 6 | 功能数量 (NumFeatures) | 5.62% |
| 7 | 发行月份 (ReleaseMonth) | 5.49% |
| 8 | 多人游戏 (HasMultiplayer) | 5.48% |
| 9 | 标签:2D (Tag_2D) | 4.73% |
| 10 | 标签数量 (NumTags) | 3.85% |

## 预测偏差分析引擎

系统内置9条分析规则，当模型预测与实际评分偏差超过15%时自动触发：

| 规则 | 触发条件 | 置信度 |
|------|----------|--------|
| 经典老游戏效应 | 发行>10年 + 实际>预测 | 0.50-0.85 |
| 近期质量下滑 | 近期评分 < 历史评分 - 10% | 0.40-0.85 |
| 开发商持续改进 | 近期评分 > 历史评分 + 10% | 0.40-0.80 |
| 内购引发差评 | 含内购 + 实际<预测 | 0.55-0.80 |
| 本地化不足 | 支持>10种语言 + 实际<预测 | 0.55 |
| 抢先体验阶段 | 标签含Early Access + 实际<预测 | 0.72 |
| 定价过高 | 价格>$40 + 实际<预测 | 0.40-0.75 |
| 社区氛围问题 | 多人游戏 + 评分<50% | 0.60 |
| 大众市场突破 | 评论>5万条 + 实际>预测 | 0.65 |

## 数据集

- **来源**: [Kaggle - Steam Games Dataset](https://www.kaggle.com/datasets/nikatomashvili/steam-games-dataset)
- **原始记录**: 71,700条
- **清洗后**: 37,481条（含评分数据）
- **特征维度**: 69个字段（价格/评分/标签/特征/语言/开发商编码等）
- **模型特征**: 49个（30个标签 + 17个结构化特征 + 2个编码特征）

## 课程要求对照

| 课程要求 | 本项目实现 |
|----------|-----------|
| 数据获取 | `data_acquisition.py` — KaggleHub程序化下载 |
| 数据保存 | CSV本地存储 + SQLite数据库 |
| 数据清洗 | 价格/评分/日期文本解析，缺失值处理，异常年份过滤 |
| 特征分析 | 相关系数热力图 + 特征重要性排名 + 强弱分析 + 偏差原因分析 |
| 趋势预测 | Random Forest回归 + Gradient Boosting分类 + 时序趋势模拟 |
| 可视化 | 4张静态图表 + Web交互式Chart.js图表（10+种图表类型） |
| 回测验证 | 80/20分割，5折交叉验证，MAE/RMSE/R²评估 |
| 额外亮点 | IsolationForest异常检测 + AI实时预测 + 偏差分析引擎 + 玻璃拟态UI + DeepSeek AI助手集成 |

## 合规声明

- 数据来源为Kaggle公开数据集，符合"免费公开、可下载"要求
- 未爬取任何付费、涉密、商业私有数据
- 数据仅用于教育学习目的
