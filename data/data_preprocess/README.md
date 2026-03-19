## 阶段 1：数据预处理与数据集构造

本目录只做 **数据预处理与实验数据集构造**，不实现模型。

### 原始数据

- `data/combinnation.xlsx`：人工整理的原始 Excel，表头至少包含：
  - `编号`（诉求唯一标识，作为 complaint_id）
  - `实体`
  - `隐患`
  - `风险`
  - `事件`
  - `后果`

### 输出数据总览

运行完两个脚本后，会得到：

- `data/cleaned_combinnation.xlsx`：清洗后的宽表，用于人工检查
- `data/raw/triples.csv`：包含 complaint_id 的三元组表
  - 列：`complaint_id, head, relation, tail`
- `data/processed/entity2id.json`
- `data/processed/relation2id.json`
- `data/processed/train.txt`
- `data/processed/valid.txt`
- `data/processed/test.txt`
  - 行格式：`head_id<TAB>relation_id<TAB>tail_id`
- `data/processed/valid_queries.json`
- `data/processed/test_queries.json`

### 第 0 步：环境准备

在项目根目录安装依赖（至少需要）：

```bash
pip install pandas openpyxl
```

### 第 1 步：Excel 清洗预览

脚本：`data/data_preprocess/preview_clean_excel.py`

从项目根目录运行：

```bash
python -m data.data_preprocess.preview_clean_excel
```

该脚本会：

1. 读取 `data/combinnation.xlsx`
2. 只保留列：`编号, 实体, 隐患, 风险, 事件, 后果`
3. 对每个单元格：
   - 去掉首尾空格
   - 按多种分隔符拆分：中文逗号、英文逗号、分号、斜杠、换行
   - 只取拆分后的 **第一个值** 作为主值
4. 行级规则：
   - `编号` 为空 ⇒ 丢弃整行
   - `实体/隐患/风险/后果` 任一为空 ⇒ 丢弃整行
   - `风险 == '空白'` ⇒ 丢弃整行
   - `事件` 为空 ⇒ 保留该行，并将 `事件` 填为 `'空白'`
5. 去重：完全相同的行只保留一条
6. 输出：`data/cleaned_combinnation.xlsx`

你可以打开 `data/cleaned_combinnation.xlsx` 检查是否符合预期，再进行下一步。

### 第 2 步：构造三元组与数据集切分

脚本：`data/data_preprocess/build_triples_and_splits.py`

从项目根目录运行：

```bash
python -m data.data_preprocess.build_triples_and_splits
```

该脚本会基于 `data/cleaned_combinnation.xlsx`：

1. 构造三元组表 `data/raw/triples.csv`：
   - 对每一行（一个诉求编号 + 一组实体/隐患/风险/事件/后果），生成以下三元组：
     - `complaint:编号  --包含实体-->  entity:实体`
     - `complaint:编号  --包含隐患-->  hidden:隐患`
     - `complaint:编号  --包含事件-->  event:事件`
     - `complaint:编号  --包含风险-->  risk:风险`
     - `complaint:编号  --包含后果-->  outcome:后果`
     - `entity:实体   --易感于-->   hidden:隐患`
     - `hidden:隐患  --导致-->     risk:风险`
     - `event:事件   --触发风险--> risk:风险`
     - `risk:风险    --导致-->     outcome:后果`
   - `complaint_id` 列记录原始 `编号`
   - `head`/`tail` 为带类型前缀的节点字符串（如 `entity:xxx`），方便避免跨类型同名冲突

2. 构建映射：
   - `entity2id.json`：所有出现在 `head` 或 `tail` 中的节点字符串映射到整数 id
   - `relation2id.json`：所有关系字符串映射到整数 id

3. 按 `complaint_id` 做 **归纳式切分**：
   - 先打乱所有 `complaint_id`（固定随机种子 42）
   - 按 7:1:2 割成 `train/valid/test` 三个诉求集合
   - 同一个 `complaint_id` 只会落在一个集合中

4. 写出初始图三元组：
   - 把每个集合中的三元组全部映射为 id，写入：
     - `data/processed/train.txt`
     - `data/processed/valid.txt`
     - `data/processed/test.txt`
   - 行格式：`head_id<TAB>relation_id<TAB>tail_id`

5. 为 valid/test 构造查询与标准答案（不生成负样本）：
   - 对 `valid` / `test` 中的三元组：
     - 把以下两种关系视为 **待预测目标边**，不保留在图中：
       - `诉求-包含风险-?`（关系字符串：`包含风险`）
       - `诉求-包含后果-?`（关系字符串：`包含后果`）
     - 对每个 `(complaint_id, relation)` 聚合所有 tail，生成一条 query：
       - `complaint_id`
       - `head_id`：`complaint:编号` 对应的实体 id
       - `relation_id`
       - `answers`：所有正确 tail_id（可能多个）
       - `head`：`complaint:编号`（便于调试）
       - `relation`：关系原文字符串
       - `answer_texts`：tail 的节点字符串列表（便于人工查看）
   - 输出到：
     - `data/processed/valid_queries.json`
     - `data/processed/test_queries.json`

6. 更新 valid/test 图：
   - 在构造 queries 过程中，脚本会从 `valid` / `test` 的图中移除所有目标边（`包含风险`、`包含后果`）
   - 然后重新覆盖写回 `valid.txt` / `test.txt`，保证：
     - 测试时模型的输入只包含：
       - `诉求-包含实体-实体`
       - `诉求-包含隐患-隐患`
       - `诉求-包含事件-事件`
       - 以及风险因果链内部边（实体-隐患-风险-后果）
     - 目标边只以 query+answers 形式出现，不在图结构中泄露答案

### 与论文任务要求的对应关系

- 节点类型：只保留 6 类节点（诉求、实体、隐患、风险、事件、后果），通过类型前缀区分
- 关系类型：严格限定为任务说明中的 9 类关系
- 切分方式：按 `complaint_id` 归纳式划分，保证测试集中诉求为训练未见的新节点
- 评估输入：valid/test 的图只提供“包含实体/隐患/事件”边作为已观测信息
- 评估目标：通过 queries 仅针对 `{诉求, 包含风险, ?}` 和 `{诉求, 包含后果, ?}` 两类关系进行预测

