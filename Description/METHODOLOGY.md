# 统一实验方法论（METHODOLOGY）

> 本文档定义 30 个不平衡学习模型（5 个迁移旧模型 + 25 个综述复现模型）共用的
> 实验协议、统一参数、指标体系与重型优化器预算封顶政策。
> 本文档**不含实验结果**——完整基准由用户在算力更优的设备上运行 `run_all.py --full` 产出。

---

## 1. 统一实验协议（保证 30 模型直接可比）

| 维度 | 设置 | 说明 |
|---|---|---|
| **数据集** | 115 个 KEEL 5 折 + 32 个构造 Bearing IR | KEEL 用其自然不平衡；Bearing 由原始 CSV 人工构造 IR{5,10,20,30} |
| **数据划分** | 5 折交叉验证 | KEEL 直接读预划分的 5 折；Bearing 用确定性 `StratifiedKFold(5, shuffle=True, random_state=42)`，保证 30 模型同折 |
| **预处理** | StandardScaler（仅训练折 fit）+ LabelEncoder | 由统一 runner `common.base` 在唯一一处执行，杜绝泄漏与重复 |
| **统一分类器** | RandomForest(n_estimators=30) | 重采样方法（14）下游统一头；bagging 型集成基学习器；DBCF/SPE/DEAHS 等保留其本体分类器 |
| **boosting 弱学习器** | DecisionTreeClassifier(max_depth=3) | boosting 型集成（AdaBoost 系）的弱学习器 |
| **k 近邻** | k=5（默认） | 各重采样方法内部 k-NN / SMOTE 统一；部分方法论文指定 k=7（DES-MI） |
| **集成规模** | N=30 | bagging/boosting 基分类器数统一为 30（与旧 5 模型一致） |
| **随机性** | random_state 逐折透传 | 每折种子 = base_seed + fold_id，可复现 |
| **路径** | 全部相对路径（PROJECT_ROOT 定位） | 可迁移到任意设备运行 |

## 2. Bearing IR 数据集构造

对 8 个原始 Bearing 数据集（CWRU/Gearbox/HUST/JNU/MFPT/Ottawa/SEU/XJTU）各构造 IR∈{5,10,20,30}：
- **多数类**（样本最多的类）保持原样；
- **所有少数类**统一下采样到 `floor(N_majority / IR)`，floor=1（保证多类结构不退化）；
- 构造脚本：`build_bearing_ir.py`（一次性，幂等），产物 `Dataset/Bearing_IR/IR{5,10,20,30}/<name>.csv` + `_manifest.json`。
- IR = 多数类样本数 / 少数类样本数，定义无歧义。

## 3. 指标体系（每模型×数据集，5 折 mean±std）

10 项指标 + 运行时间：

| 指标 | 说明 | 二分类路径 |
|---|---|---|
| Accuracy | 准确率 | — |
| Precision | macro 精确率 | — |
| Recall | macro 召回率 | — |
| F1 | macro F1 | — |
| GMean | 各类召回几何均值 | — |
| AUC | ovr macro ROC-AUC | n_classes==2 走二分类 |
| MCC | Matthews 相关系数 | — |
| AUPRC | macro 平均精确率（PR-AUC，失衡关键） | n_classes==2 走二分类 |
| IBA | 平衡准确率指数，T=0.5 | — |
| Kappa | Cohen's Kappa | — |
| Runtime | fit 时长（秒，不含预测） | — |

**混淆矩阵**：按（模型×数据集）将 5 折测试集**求和**成一张汇总矩阵，行归一化（对少数类公平），dpi=600，Times New Roman，英文标签。导出至 `Result/Figures/Confusion/<Model>__<Dataset>.png`。

## 4. 统一模型契约

每个模型模块导出 sklearn 风格类（`fit(X,y)` / `predict(X)` / `predict_proba(X)` / `classes_`）+ `MODEL_KEY` + `build(random_state, smoke, **unified)` 工厂。统一 runner `common.base.run_model_on_dataset` 驱动全部 30 模型，零分支：
- 重采样方法（14）：`fit()` 内部重采样→训练 RF(30)→`predict_proba` 委托之；
- 集成方法（11）：`fit()` 内部建 N 基学习器并聚合概率。
- `preencoded=True` 旗标：runner 已完成 scale+encode，模型跳过自身 scaler/encoder。

## 5. 重型优化器预算封顶政策（GLOBAL BUDGET CAP）

为在单线程、有限算力下完成，对依赖元启发式优化器的方法，**机制（MECHANISM）原样保留，仅预算（BUDGET）缩减**。每个受影响方法的 `code_review.md` 列出"论文预算 vs 使用预算"对照表。

| 方法 | 优化器 | 机制（保留） | 使用预算（封顶） |
|---|---|---|---|
| OCSV-US | 单目标 GA (DEAP) | 二进制掩码选 SV，目标=平衡准确率 | pop=12, gen=8 |
| EVINCI | NSGA-II (DEAP selNSGA2) | 双目标（准确率↑, 重叠区多数类保留惩罚↓）选实例子集 | pop=12, gen=8 |
| S-SMOTE | 两阶段动态过采样 + 模态 GA + 岭 LS 精修 | 阶段 1 扩全局最小类；进化期扩 minimum-sensitivity 类；GA 调 RBFNN 中心数/宽度/岭，LS 解权重 | pop=12, gen=8 |
| AMCS | BPSO | 路由内基分类器的特征子集最大化 CV AUCarea；可切 FCBF / BPSO | particles=20, iters=10; w=0.729; c1=c2=1.49445 |
| MC-EVHS / E-EVRS | Dempster-Shafer 证据合成 | 质量积+冲突归一化闭式（手写） | 无预算（闭式） |
| GLOS | MMO+AMBO+RRO 三子算法 | 确定性几何，无优化器 | 无封顶 |

**近似标注原则**：任何与论文的实现差异（如 sklearn CART 代替 C4.5/J48、CFSFDP→safe-level 分组、聚类→连续阈值、OvO 简化为多类直接、模态内循环省略等）都在该方法的 `code_review.md` 中显式列出，**不静默简化**。

## 6. 本报告系列不含的内容

- **实验结果报告**：本系列只含技术说明（算法/公式/超参/偏差）与代码审查（论文↔代码保真表+近似清单）。完整基准结果（30 模型 × 147 数据集）由 `run_all.py --full` 产出 `Result/All_Models_Results.csv`，结果分析报告不在本项目交付范围（用户自行运行后撰写）。
