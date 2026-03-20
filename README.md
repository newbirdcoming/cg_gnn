## 项目说明：风险因果链最小实验

本项目实现了一个面向**新诉求节点**的最小知识图谱推断实验，围绕"风险因果链"完成数据预处理、基线模型（TransE / R-GCN）以及规则引导的动态子图模型。

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
    - `background.txt`：背景图（train + valid/test 结构边，用于子图提取时的全图邻接查询）
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
  - `grail_style.py`：GraIL-style 对比基线（query-specific 封闭子图 + 结构特征 + 消息传递编码器）
- `subgraph_model/`
  - `subgraph.py`：邻接表构建工具（`build_adjacency`）
  - `subgraph_dynamic.py`：动态子图提取核心逻辑
    - `extract_dynamic_subgraph`：针对 (h, r, t) 三元组，按规则路径（P1–P5）构建面向查询的最小子图
    - `dynamic_path_support_mapping`：预计算 (h, r) 下哪些候选 tail 有路径支撑
  - `encoder.py`：`LocalRGCNEncoder`（局部子图 GNN 编码器）
  - `decoder.py`：`SubgraphDistMultDecoder`（DistMult 打分解码器）
  - `minimal_dynamic_subgraph_model.py`：`MinimalSubgraphModel` 类 + 动态子图版完整训练/评估 runner
  - `dynamic_keypath_fusion_model.py`：`DynamicKeypathFusionModel` 类 + 融合版完整训练/评估 runner
- `results/`
  - `transe_metrics.json`：TransE 在 valid/test 上的指标
  - `rgcn_metrics.json`：R-GCN 在 valid/test 上的指标
  - `subgraph_minimal_dynamic_metrics.json`：动态子图模型在 valid/test 上的指标
  - `subgraph_dynamic_fusion_metrics.json`：动态子图 + 关键路径融合模型在 valid/test 上的指标
  - `grail_style_metrics.json`：GraIL-style baseline 在 valid/test 上的指标
- 其他：
  - `background.txt`：实验范围与任务背景说明
  - `task.txt`：整体实验设定与阶段划分
  - `inspect_graph.py`：简单查看训练图中部分诉求节点及其出边结构

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
5. 构造图与查询：
   - `train.txt`：保留该投诉下的所有三元组（id 格式）
   - `valid.txt` / `test.txt`：
     - 从图中移除目标边：`{诉求, 包含风险, ?}` 和 `{诉求, 包含后果, ?}`
     - 只保留结构边（诉求-包含实体/隐患/事件）与因果链内部边（实体-隐患-风险-后果）
   - `valid_queries.json` / `test_queries.json`：
     - 针对每个 (complaint_id, relation ∈ {包含风险, 包含后果}) 聚合所有 tail

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
    - 使用 filtered setting 做尾实体预测评估

### 2.3 TransE 链路预测基线

```bash
python -m baseline.transe_baseline
```

- 标准 TransE，打分为 `-||h + r - t||`
- 训练：MarginRankingLoss，负样本腐化 tail
- 输出：`results/transe_metrics.json`

### 2.4 R-GCN 图模型基线（R-GCN + DistMult）

```bash
python -m baseline.rgcn_baseline
```

- R-GCN 编码器（每关系独立变换矩阵）+ DistMult 解码器
- 训练：BCEWithLogitsLoss，带实体类型约束的负采样
- 输出：`results/rgcn_metrics.json`

---

## 阶段 3：规则引导的动态子图模型

目标：利用领域规则（风险因果链路径 P1–P5）为每个查询动态构建最小支撑子图，替代全图 GNN，提升归纳性和可解释性。

### 3.1 核心设计

**动态子图提取**（`subgraph_model/subgraph_dynamic.py`）：

针对查询三元组 `(head, relation, tail_candidate)`，只保留能支撑以下规则路径的节点与边：

| 任务 | 路径类型 | 路径模式 |
|---|---|---|
| 包含风险 | P1 | 诉求 -(包含隐患)→ 隐患 -(导致)→ 风险 |
| 包含风险 | P2 | 诉求 -(包含事件)→ 事件 -(触发风险)→ 风险 |
| 包含风险 | P3 | 诉求 -(包含实体)→ 实体 -(易感于)→ 隐患 -(导致)→ 风险 |
| 包含后果 | P4 | 诉求 -(包含隐患)→ 隐患 -(导致)→ 风险 -(导致)→ 后果 |
| 包含后果 | P5 | 诉求 -(包含事件)→ 事件 -(触发风险)→ 风险 -(导致)→ 后果 |

子图 meta 信息包含 `has_valid_path`（是否有有效路径）与 `matched_path_types`（命中的路径类型列表）。

**模型结构**（`subgraph_model/minimal_dynamic_subgraph_model.py`）：

- `MinimalSubgraphModel`：`LocalRGCNEncoder`（self-loop 版）+ `SubgraphDistMultDecoder`
- 训练时只对 `has_valid_path=True` 的样本计算 loss
- 负样本从 `dynamic_path_support_mapping` 返回的可达 tail 集合中采样（同类型约束）
- 评估时用 `reachable_map` 门控：无路径支撑的候选 tail 直接给 `-1e9` 分

### 3.2 运行命令

```bash
python -m subgraph_model.minimal_dynamic_subgraph_model
```

- 输出：`results/subgraph_minimal_dynamic_metrics.json`

---

## 阶段 4：动态子图 + 规则关键路径融合

目标：在动态子图 graph branch 的基础上，引入规则关键路径 path branch，在**分数层**进行线性融合：

```
score = lambda_r * score_graph + (1 - lambda_r) * score_path
```

### 4.1 核心设计

**路径特征**（5 维，固定维度顺序）：

| 维度 | 路径类型 | 路径模式 |
|---|---|---|
| P1 | 风险-直接隐患 | 诉求-(包含隐患)→隐患-(导致)→风险 |
| P2 | 风险-直接事件 | 诉求-(包含事件)→事件-(触发风险)→风险 |
| P3 | 风险-实体链 | 诉求-(包含实体)→实体-(易感于)→隐患-(导致)→风险 |
| P4 | 后果-隐患链 | 诉求-(包含隐患)→隐患-(导致)→风险-(导致)→后果 |
| P5 | 后果-事件链 | 诉求-(包含事件)→事件-(触发风险)→风险-(导致)→后果 |

路径特征直接从 `extract_dynamic_subgraph` 返回的 `meta.matched_path_types` 构造，无需重新枚举全图。

**score_path 计算**：

```
final_weight = rule_init_weight + delta_weight
score_path = path_feat @ final_weight[rel_task_idx]
```

- `rule_init_weight`：固定规则先验（risk=[1,1,0.5,0,0]，outcome=[0,0,0,1,1]），不参与梯度
- `delta_weight`：可学习修正项，初始化为零，从纯规则先验出发由数据修正

**lambda_r**：仅对两个目标关系各维护一个 logit，sigmoid 后得到 (0,1) 融合权重

**graph branch**：直接复用 `MinimalSubgraphModel`（LocalRGCNEncoder + DistMult），不修改内部结构

### 4.2 训练日志

每个 epoch 输出：
```
[Fusion] Epoch N/5  loss=X  used=N  skip_pos=M  skip_neg=K  lambda_risk=X
```
- `skip_pos`：无有效路径的正样本跳过数
- `skip_neg`：找不到合格负样本的跳过数

### 4.3 运行命令

```bash
python -m subgraph_model.dynamic_keypath_fusion_model
```

- 实现文件：`subgraph_model/dynamic_keypath_fusion_model.py`
- 输出：`results/subgraph_dynamic_fusion_metrics.json`

---

## 阶段 5：GraIL-style 对比基线

本实现是一个**最小可运行、思想对齐的 GraIL-style baseline**，用于第四章对比实验。不是对任何外部原始工程的严格逐项复现，仅在以下四个核心维度上与 GraIL 思想对齐：

| 维度 | GraIL-style Baseline | 主模型（动态子图融合版）|
|---|---|---|
| 子图构建 | k-hop BFS 封闭子图（双侧展开，无规则过滤） | 规则路径过滤子图（P1–P5） |
| 节点特征 | 结构距离特征（dist_h/dist_t/is_head/is_tail） | 全局实体 embedding（id-based） |
| 图编码器 | GraIL 风格最小消息传递 R-GCN | self-loop 映射 |
| 打分方式 | MLP(z_h, z_t, r_emb) | λ·DistMult + (1-λ)·path_score |

### 5.1 核心设计

**子图提取**（`extract_grail_subgraph`）：
- 从 head 和 candidate tail 各做 k=2 跳 BFS（补逆边，无向图）
- 取两侧邻域并集作为子图节点
- 保留子图内所有有向边，排除目标关系（包含风险/包含后果）
- head 和 tail 强制包含

**节点结构特征**（4 维，不含实体 id）：
```
[dist_h_norm, dist_t_norm, is_head, is_tail]
```
BFS 不可达节点距离设为 k+1，归一化后截断至 1.0。

**编码器**（`GraILEncoder`）：
- `input_proj`：Linear(4, dim) 将结构特征投影到隐层
- 多层 `GraILConvLayer`：对每个节点按入边关系类型做加权聚合，入度归一化

**打分**：
```
score(h, r, t) = MLP( concat(z_h, z_t, r_emb) )
```
relation_id 通过 r_emb 显式参与打分。

### 5.2 运行命令

```bash
python -m baseline.grail_style
```

运行后会先打印风险/后果任务各一个样本的 query-specific 子图（节点数、边数、角色标记、距离特征），随后开始训练并输出评估指标。

- 输出：`results/grail_style_metrics.json`
- 输出：`outputs/grail_style/metrics.json`

---

## 备注

- 所有模型均复用 `baseline/data.py` 与 `baseline/metrics.py` 接口，评估方式一致（filtered setting，Hits@1/3/10 + MRR）。
- 邻接表构建使用 `background.txt`（若存在）作为子图提取的背景图，否则退回 `train.txt`。
