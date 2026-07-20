# 运行指南

查看可用模型：python run_all.py --list-models

冒烟测试：只对 ecoli 数据集进行五折交叉验证：python run_all.py

断点续跑，跳过已经完成的实验：python run_all.py --resume

完整实验，开始运行就用这个指令：python run_all.py --full

指定模型：python run_all.py --models soup,DBCF,spe

指定数据集：python run_all.py --datasets ecoli1,glass0,yeast1

指定故障诊断数据集及不平衡比：python run_all.py --bearing-names CWRU --irs 5,20

使用自定义的配置文件运行：python run_all.py --config my_config.yaml

```python
usage: run_all.py [-h] [--config CONFIG] [--full] [--models MODELS]
                  [--datasets DATASETS] [--bearing-names BEARING_NAMES]
                  [--irs IRS] [--n-folds N_FOLDS] [--out OUT]
                  [--resume] [--list-models]

参数:
  --config CONFIG         YAML 配置文件路径（默认: config.yaml）
  --full                  Full 模式（覆盖配置文件的 mode）
  --models MODELS         模型列表，逗号分隔
  --datasets DATASETS     KEEL 数据集列表，逗号分隔
  --bearing-names NAMES   轴承名称列表，逗号分隔
  --irs IRS               不平衡率列表，逗号分隔，如 5,20
  --n-folds N             每数据集折数
  --out PATH              输出 xlsx 路径
  --resume                断点续跑（跳过已完成的结果）
  --list-models           列出所有发现的模型及对应论文名
```
