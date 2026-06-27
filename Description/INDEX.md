# 30 模型技术报告索引（INDEX）

> 本项目复现并统一接入 **30 个**多类不平衡学习模型：5 个迁移旧模型 + 25 个综述
> （Fachrie et al. 2025）复现模型（14 重采样 + 11 集成）。全部模型经全量冒烟测试
> （KEEL ecoli1/glass + Bearing IR5_CWRU，各 1 折）通过，0 错误。

## 统一协议

见 [METHODOLOGY.md](METHODOLOGY.md)：115 KEEL 5 折 + 32 构造 Bearing IR{5,10,20,30} 数据集；
5 折 CV；StandardScaler 仅训练折 fit；统一下游 RF(30)（boosting 用 DT(max_depth=3)）；
k=5，集成 N=30；10 指标 + Runtime + dpi600 Times New Roman 混淆矩阵。

## 模型清单

### A. 迁移旧模型（原 5 模型，统一 harness 迁移；技术说明见 `Description/`）

| 模型 | 机制 | 分类器 | 报告 |
|---|---|---|---|
| SPE | 自步硬度分箱欠采样集成 | RF(30) 集成 | 五模型技术说明.md |
| DeepSMOTE | 编解码器(重构+惩罚损失)+潜空间 SMOTE | RF(30) | 五模型技术说明.md |
| DDPM | 类条件扩散模型数据增广 | 论文 CNN 头 / 可选 RF(30) 评测适配 | 五模型技术说明.md |
| DBCF | Up-down 采样 + Bent 平衡熵级联森林 | 平衡级联森林(Embedded) | 五模型技术说明.md |
| DEAHS | SAC 欠采样 + 加权过采样 + 动态集成 | RF(30) 集成 | 五模型技术说明.md |

### B. 重采样方法（14，下游统一 RF(30)）— 详见 [重采样方法_技术说明与代码审查.md](重采样方法_技术说明与代码审查.md)

| 模型 | 年份 | 过采样机理 | 欠采样机理 | 类处理 | heavy |
|---|---|---|---|---|---|
| soup (SOUP) | 2019 | 相似度感知复制 | 相似度感知优先级 | 逐类 | — |
| mdo (MDO) | 2016 | 等马氏轮廓 PC 生成 | — | 逐类 | — |
| smom (SMOM) | 2017 | 方向加权 SMOTE(聚类标签驱动) | — | 全类 | — |
| mc_rbo (MC-RBO) | 2020 | mutual-potential 势域搜索 | — | 多类专用 | — |
| orem_m (OREM-M) | 2023 | CMR+clean-subregion 可靠扩展 | — | 全类 | — |
| mc_ccr (MC-CCR) | 2020 | 能量驱动球形边界 | 数据平移 | 多类专用 | — |
| shsampler (SHSampler) | 2022 | 球形(跨类k-NN半径) | difficulty-aware 轮盘赌 | 相对全类 | — |
| mc_nro (MC-NRO) | 2024 | mutual-potential 径向势域 | 平移+邻域重划分删除 | 多类专用 | — |
| mc_evhs (MC-EVHS) | 2022 | 证据SMOTE(低belief) | 优先级 | 全类 | DS |
| nromm (NROMM) | 2023 | adaptive-embedding 三组生成 | overlap/非 inland 清洗 | 全类(OvE) | — |
| scut (SCUT) | 2015 | SMOTE | EM-BIC 聚类分层 RUS | 逐类 | — |
| glos (GLOS) | 2023 | MMO+AMBO+RRO(全局+局部) | — | 全类 | compound |
| s_smote (S-SMOTE) | 2011 | 两阶段动态灵敏度SMOTE | — | 最小 sensitivity 类驱动 | memetic-GA |
| ocsv_us (OCSV-US) | 2021 | — | OC-SVM SV + 类平衡 GA 选择 | 逐类 | GA |

### C. 集成学习方法（11）— 详见 [集成学习方法_技术说明与代码审查.md](集成学习方法_技术说明与代码审查.md)

| 模型 | 年份 | 机制 | 基分类器 | 聚合 | heavy |
|---|---|---|---|---|---|
| pt_bagging (PT-Bagging) | 2018 | Bagging+plug-in threshold | DT(J48≈CART) | 先验阈值移动 | — |
| multirandbal (MultiRandBal) | 2020 | 随机平衡+SMOTE | DT | 多数投票 / alpha 加权投票 | — |
| adaboost_ad (AdaBoost.AD) | 2024 | 分布与自适应权重 AdaBoost | Decision Tree | beta 加权投票 | — |
| oremboost (OREMBoost) | 2023 | AdaBoost+OREM-M每轮 | DT(max_depth=3) | alpha 加权投票 | — |
| des_mi (DES-MI) | 2018 | 随机平衡 + 加权邻域动态选择 | CART | 动态硬投票 / 概率平均 | — |
| drcw_aseg (DRCW-ASEG) | 2018 | OvO + 测试期 ASEG | CART | 距离相对能力加权 | — |
| easy_bpnn (Easy-BPNN) | 2016 | OvO + EasyEnsemble | BPNN(Embedded) | OvO 后验融合 | — |
| evinci (EVINCI) | 2019 | 随机平衡+样本级 NSGA-II | CART | 概率平均 | NSGA-II |
| amcs (AMCS) | 2016 | 8 类数据自适应路由 + FCBF/BPSO + AdaBoost.M1 | C4.5 / SVM / KNN / RBF-NN | AUCarea 加权多数投票 | BPSO |
| e_evrs (E-EVRS) | 2023 | Bagging+MC-EVHS | DT / SVM | Dempster 证据合成 | DS |
| dpse (DPSE) | 2021 | OvA差分分区+s-RUS/br-SMOTE | DT | 置信度加权 | — |

## 运行

```bash
conda activate chuantaoli
python build_bearing_ir.py                 # 构造 32 个 Bearing IR 数据集（一次性）
python run_all.py --smoke                  # 冒烟（3 数据集 × 1 折）
python run_all.py --full                   # 完整基准（147 数据集 × 5 折，慢）
python run_all.py --models soup,mdo        # 仅部分模型
python run_all.py --datasets ecoli,glass   # 仅部分 KEEL
python run_all.py --list-models            # 查看注册表
```
