# 🎮 Steam Game Analytics

> 2026年实践学期项目 — 基于机器学习的Steam游戏评分预测与可视化分析平台

## 项目简介

本项目从Kaggle获取Steam游戏数据集（7万+款游戏），完成**数据采集→清洗→特征工程→机器学习建模→回测验证→Web可视化**的完整数据科学流水线。最终以Web应用形式呈现，支持游戏列表浏览、详情查看、AI实时预测评分、数据分析看板等功能。

## 技术栈

| 层级 | 技术 |
|------|------|
| 数据获取 | KaggleHub (Kaggle API) |
| 数据处理 | Pandas, NumPy |
| 机器学习 | Scikit-learn (Random Forest, Gradient Boosting) |
| 后端 | Flask (REST API) |
| 前端 | Vanilla JS + Chart.js |
| 数据库 | SQLite |
| 可视化 | Matplotlib, Seaborn, Chart.js |

## 项目结构

```
steam_project/
├── data_acquisition.py      # 模块1: 数据获取（从Kaggle下载）
├── data_exploration.py      # 模块2: 数据探索（EDA）
├── data_cleaning.py         # 模块3: 数据清洗与特征提取
├── train_model.py           # 模型训练 + SQLite数据库构建
├── modeling.py              # 模块4: 特征分析+建模+回测+可视化
├── app.py                   # Flask后端（6个API端点）
├── index.html               # 前端页面（单文件完整应用）
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
pip install -q kagglehub pandas numpy matplotlib seaborn scikit-learn flask

# 2. 下载数据并训练模型（二选一）
python setup.bat          # Windows一键脚本
# 或手动执行：
python data_acquisition.py  # 从Kaggle下载数据
python data_cleaning.py     # 清洗数据
python train_model.py       # 训练模型+建库

# 3. 启动服务器
python app.py

# 4. 浏览器打开
# http://localhost:5000
```

## Web应用功能

| 页面 | 功能描述 |
|------|----------|
| 📊 **游戏总览** | 统计卡片 + 可搜索/排序/筛选的游戏列表，点击进入详情 |
| 🔬 **自由预测** | 调节参数（价格/语言/标签/特征），AI实时预测游戏好评率 |
| 📈 **数据分析** | 评分饼图、价格分布、年度趋势、热门标签等图表 |

### API端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/stats` | 全局统计概览 |
| GET | `/api/games?page=&sort=&search=` | 游戏列表（分页+搜索+排序） |
| GET | `/api/games/<id>` | 游戏详情+AI预测+同类推荐 |
| POST | `/api/predict` | 实时评分预测 |
| GET | `/api/feature-importance` | 模型特征重要性 |
| GET | `/api/tags` | 可用标签列表 |

## 模型性能

| 任务 | 模型 | 指标 |
|------|------|------|
| 评分回归 | Random Forest | MAE=7.48%, R²=0.25, 误差≤10%占比77.5% |
| 评分分类 | Gradient Boosting | Accuracy=62.18% (3分类) |

## 数据集

- **来源**: [Kaggle - Steam Games Dataset](https://www.kaggle.com/datasets/nikatomashvili/steam-games-dataset)
- **原始记录**: 71,700条
- **清洗后**: 37,481条（含评分数据）
- **特征维度**: 60个字段（价格/评分/标签/特征/语言等）

## 课程要求对照

| 课程要求 | 本项目实现 |
|----------|-----------|
| 数据获取 | `data_acquisition.py` — KaggleHub程序化下载 |
| 数据保存 | CSV本地存储 + SQLite数据库 |
| 数据清洗 | 价格/评分/日期文本解析，缺失值处理，异常年份过滤 |
| 特征分析 | 相关系数热力图 + 特征重要性排名 + 强弱分析 |
| 趋势预测 | Random Forest回归 + Gradient Boosting分类 |
| 可视化 | 4张静态图表 + Web交互式Chart.js图表 |
| 回测验证 | 80/20分割，5折交叉验证，MAE/RMSE/R²评估 |
| 额外亮点 | IsolationForest异常检测 + AI实时预测工具 |

## 合规声明

- 数据来源为Kaggle公开数据集，符合"免费公开、可下载"要求
- 未爬取任何付费、涉密、商业私有数据
- 数据仅用于教育学习目的
