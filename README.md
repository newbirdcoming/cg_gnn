## 项目说明：风险因果链最小实验

本项目实现了一个面向**新诉求节点**的最小知识图谱推断实验，围绕“风险因果链”完成数据预处理与两类基线模型（TransE / R-GCN）。

### 目录结构总览

- `data/`
  - `combinnation.xlsx`：原始人工整理 Excel
  - `cleaned_combinnation.xlsx`：清洗后的宽表（阶段 1 中间产物）
  - `run_full_pipeline.py`：从原始 Excel 一键生成清洗宽表 + triples + processed 数据（推荐入口）
  - `raw/`
    - `triples.csv`：最终用于实验的三元组表
      - 列：`complaint_id, head, relation, tail`
      - 只保留 6 类节点与 9 类关系
  - `processed/`
    - `train.txt`：训练图三元组（id 格式，`head_id\trelation_id\ttail_id`）
    - `valid.txt`：验证图三元组（已移除目标边）
    - `test.txt`：测试图三元组（已移除目标边）
    - `entity2id.json`：实体字符串 → id 映射
    - `relation2id.json`：关系字符串 → id 映射
    - `complaint_split.json`：按 complaint_id 的 7:1:2 归纳式切分
    - `valid_queries.json`：验证集查询与标准答案（仅 `{诉求, 包含风险/后果, ?}`）
    - `test_queries.json`：测试集查询与标准答案
  - `data_preprocess/`
    - `preview_clean_excel.py`：从 `combinnation.xlsx` 生成 `cleaned_combinnation.xlsx`
    - `build_triples_and_splits.py`：从清洗后 Excel 生成 `triples.csv` + processed 数据
    - `README.md`：阶段 1 预处理流程说明
- `data/preprocess.py`：从 `data/raw/triples.csv` 重新构造 processed 数据（保留作备选入口/复现实验）
- `baseline/`
  - `data.py`：统一加载 processed 数据（train/valid/test + queries + 映射）
  - `metrics.py`：实现 filtered setting 下的 Hits@K / MRR 评估
  - `transe_baseline.py`：TransE 链路预测基线
  - `rgcn_baseline.py`：R-GCN 图模型基线
- `results/`
  - `transe_metrics.json`：TransE 在 valid/test 上的指标
  - `rgcn_metrics.json`：R-GCN 在 valid/test 上的指标
- 其他：
  - `background.txt`：实验范围与任务背景说明
  - `task.txt`：整体实验设定与阶段划分
  - `inspect_graph.py`：简单查看训练图中部分诉求节点及其出边结构（只打印 id 和关系名，避免中文乱码）

---

## 阶段 1：数据预处理与数据集构造

目标：从原始 Excel 构造一个**面向新诉求的风险因果链最小实验数据集**，并保证：

- 只保留 6 类节点：诉求、实体、隐患、风险、事件、后果
- 只保留 9 类关系：
  - `诉求-包含实体-实体`
  - `诉求-包含隐患-隐患`
  - `诉求-包含事件-事件`
  - `诉求-包含风险-风险`
  - `诉求-包含后果-后果`
  - `实体-易感于-隐患`
  - `隐患-导致-风险`
  - `事件-触发风险-风险`
  - `风险-导致-后果`

### 1.1 一键从原始 Excel 构造数据集（推荐入口）

```bash
python -m data.run_full_pipeline
```

该脚本只依赖：`data/combinnation.xlsx`，并完成：

1. 调用 `data.data_preprocess.preview_clean_excel`：
   - 从原始 Excel 读取数据
   - 清洗空值/多值，生成 `data/cleaned_combinnation.xlsx`
2. 调用 `data.data_preprocess.build_triples_and_splits`：
   - 从 `cleaned_combinnation.xlsx` 构造 `data/raw/triples.csv`
   - 同时生成 `data/processed/` 下的 train/valid/test 与 queries

### 1.2 从已存在的 triples.csv 重新构造数据集（备选入口）

```bash
python -m data.preprocess
```

该脚本依赖：`data/raw/triples.csv`，并完成：

1. 读取三元组：列为 `complaint_id, head, relation, tail`
2. 清洗：
   - 丢弃关键字段为空的行
   - 丢弃不在 9 类关系集合中的行
   - 删除重复三元组
3. 构建映射：
   - `entity2id.json`：所有节点字符串 → 连续整数 id
   - `relation2id.json`：所有关系字符串 → 连续整数 id
4. 按 `complaint_id` 做归纳式切分：
   - 按 7:1:2 划分为 train / valid / test
   - 同一个 complaint_id 只会出现在一个集合中
   - 输出 `complaint_split.json`
5. 构造图与查询（逻辑与 1.1 中的构造保持一致）：
   - `train.txt`：保留该投诉下的所有三元组（id 格式）
   - `valid.txt` / `test.txt`：
     - 从图中移除目标边：
       - `{诉求, 包含风险, ?}`
       - `{诉求, 包含后果, ?}`
     - 只保留：
       - `诉求-包含实体-实体`
       - `诉求-包含隐患-隐患`
       - `诉求-包含事件-事件`
       - 以及因果链内部边（实体-隐患-风险-后果）
   - `valid_queries.json` / `test_queries.json`：
     - 针对每个 (complaint_id, relation ∈ {包含风险, 包含后果}) 聚合所有 tail：
       - `complaint_id`
       - `head_id`（诉求节点 id）
       - `relation_id`
       - `answers`（所有正确 tail_id 列表）
       - 可读字段：`head, relation, answer_texts`

---

## 阶段 2：最小基线实验

目标：在构造好的数据集上，实现两个最小基线模型，完成新诉求的：

- `{诉求, 包含风险, ?}`
- `{诉求, 包含后果, ?}`

尾实体预测任务。

### 2.1 依赖安装

```bash
pip install pandas openpyxl
pip install torch  # 或根据机器选择合适的 PyTorch 安装命令
```

### 2.2 统一数据与评估接口

- `baseline/data.py`
  - 从 `data/processed/` 读取：
    - `entity2id.json` / `relation2id.json`
    - `train.txt` / `valid.txt` / `test.txt`
    - `valid_queries.json` / `test_queries.json`
  - 返回 `KGData`，包含：
    - `num_entities, num_relations`
    - `train_triples, valid_triples, test_triples`（id 三元组）
    - `valid_queries, test_queries`
    - `all_triples_set`：所有三元组集合，用于 filtered setting

- `baseline/metrics.py`
  - `compute_ranking_metrics(ranks)`：给定 rank（1=最好），计算 Hits@1/3/10 和 MRR
  - `evaluate_tail_predictions(queries, score_fn, num_entities, all_triples_set)`：
    - 使用 filtered setting 做尾实体预测评估：
      - 对每个 query，只在候选集中保留不与其他真尾冲突的实体
      - 对每个真 tail 计算 rank，然后汇总指标

### 2.3 TransE 链路预测基线

运行命令：

```bash
python -m baseline.transe_baseline
```

- 模型：
  - 标准 TransE：打分为 \(-\|h + r - t\|\)，分数越高越好
- 训练：
  - 从 `train.txt` 随机采样负样本（腐化 tail）
  - 使用 `MarginRankingLoss` 进行优化
  - 默认参数：
    - 维度 dim=100
    - 学习率 lr=1e-3
    - 轮数 epochs=50
    - batch_size=512
- 评估：
  - 在 `valid_queries.json` / `test_queries.json` 上：
    - 只针对 `{诉求, 包含风险, ?}` 与 `{诉求, 包含后果, ?}` 做尾实体预测
    - 使用 filtered setting 计算 Hits@1/3/10 和 MRR
- 输出：
  - `results/transe_metrics.json`
    - 结构形如：
      - `{"valid": {...}, "test": {...}}`

### 2.4 R-GCN 图模型基线（R-GCN + DistMult）

运行命令：

```bash
python -m baseline.rgcn_baseline
```

  - 图构建：
    - 使用 `train.txt` 三元组构造有向图，并将每条边视为无向（h→t 与 t→h），以便信息双向流动。
- 模型结构：
  - R-GCN 编码器：
    - 每种关系 r 拥有一个变换矩阵 W_r
    - 分层聚合邻居信息，叠加自环权重
    - 最终得到每个实体的图嵌入
  - DistMult 解码器：
    - 为每个关系 r 学习一个关系向量 `r`
    - 使用 DistMult 风格得分函数：`score(h, r, t) = <h, r, t>`
- 训练（带类型约束的负采样）：
  - 在训练三元组上进行二分类（正样本 vs 负样本），使用 `BCEWithLogitsLoss`
  - 负样本生成策略：
    - 只腐化 tail（保持 (h, r) 不变）
    - 对 `包含风险` 关系：只从 `risk:*` 节点集合中采样负尾实体
    - 对 `包含后果` 关系：只从 `outcome:*` 节点集合中采样负尾实体
    - 对其他关系：从全体实体中随机采样
    - 避免采到当前正例 tail，同时尽量过滤掉该 (h, r) 在训练集中已出现过的其他真 tail，减少“假负样本”
- 评估：
  - 与 TransE 相同，使用 `evaluate_tail_predictions` 在 valid/test queries 上做 filtered tail prediction
- 输出：
  - `results/rgcn_metrics.json`

---

## 备注与后续扩展

- 当前实现只覆盖：
  - 阶段 1：数据预处理与实验数据集构造
  - 阶段 2：最小基线（TransE / R-GCN）
- 阶段 3 及以后（局部子图模型、元路径增强等）可以在 `baseline/` 目录继续扩展：
  - 新建子目录或脚本（如 `subgraph_model.py`、`metapath_model.py`）
  - 继续复用 `baseline/data.py` 与 `baseline/metrics.py` 以保证接口和评估的一致性。

