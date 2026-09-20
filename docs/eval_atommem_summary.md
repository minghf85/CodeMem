# AtomMem 评测实现总结

## 实现内容

### 1. 核心文件

- **`scripts/eval_atommem.py`**: 原子记忆评测脚本
  - 同时评测 msgmem 和 atommem 的检索召回和答案准确率
  - 支持 `--resume` 断点续跑
  - 支持 `--limit` 限制样本数（用于快速测试）
  - 实时显示 judge accuracy, evidence recall, ETA 等指标

- **`configs/atommem.yaml`**: 评测配置文件
  - embedding 模型配置（Qwen3-Embedding-4B）
  - 生成模型配置（Qwen3-8B）
  - judge 模型配置
  - 并发控制参数

- **`scripts/analyze_atommem_results.py`**: 结果分析脚本
  - 对比 msgmem 和 atommem 的召回率和答案准确率
  - 统计改进、持平、下降的问题数量
  - 输出典型案例

### 2. 关键功能

#### Evidence 召回计算
- **msgmem**: 直接使用 msgmem.jsonl 中的 `metadata.id`
- **atommem**: 使用 atommem 的 `metadata.source` 字段追溯到原始 msgmem ID

#### 双路评测
每个问题同时评测两种记忆：
1. 用 embedding 检索 msgmem（top-k=30）
2. 用 embedding 检索 atommem（top-k=30）
3. 分别生成答案
4. 分别 judge 答案
5. 计算各自的 evidence recall, exact match, token F1

## 测试结果（50个样本）

### 总体指标

| 指标 | msgmem | atommem | 提升 |
|------|--------|---------|------|
| **Evidence Recall** | 28.0% | **85.1%** | **+204%** |
| **Judge Accuracy** | 16.0% | **48.0%** | **+200%** |
| **Token F1** | 0.089 | **0.169** | **+90%** |

### 召回改进分布

- **68%** 的问题召回率提升（34/50）
- **32%** 的问题召回率持平（16/50）
- **0%** 的问题召回率下降

### 答案准确率改进

- **净提升 +32%**（+16 个问题）
- 只有 atommem 答对：**38%**（19/50）
- 只有 msgmem 答对：**6%**（3/50）
- 两者都答对：10%（5/50）
- 两者都答错：46%（23/50）

### 典型案例

#### Case 1: 召回率从 0% → 100%
- **问题**: "What did Caroline research?"
- **msgmem 召回**: 0% → 答案 "Unknown"
- **atommem 召回**: 100% → 答案 "Adoption agencies and counseling career options"
- **分析**: atommem 的原子化抽取使得"research adoption agencies"这个事实更容易被检索到

#### Case 2: 时间信息更准确
- **问题**: "When did Caroline go to the LGBTQ support group?"
- **参考答案**: "7 May 2023"
- **msgmem**: "2023-05-08" ❌（日期错误）
- **atommem**: "2023-05-07" ✅（日期正确）
- **分析**: 虽然两者召回率都是 100%，但 atommem 的时间字段标准化帮助模型更准确地定位时间

#### Case 3: 零召回下仍能答对
- **问题**: "What is Caroline's identity?"
- **两者召回**: 0%
- **msgmem**: "Caroline is Melanie's friend and a supportive individual" ❌
- **atommem**: "Caroline's identity is as a trans woman" ✅
- **分析**: atommem 将身份信息抽取为独立的原子记忆，即使 evidence 没有直接命中，相关原子记忆仍能提供正确信息

## 为什么 AtomMem 更好？

### 1. 粒度优势
- **msgmem**: 一条消息可能包含多个事实，语义混杂
- **atommem**: 每个原子记忆只包含单一事实，语义更纯粹

### 2. 结构优势
- **时间标准化**: ISO-8601 格式，易于比较和检索
- **实体显式化**: tag 中明确标注 speaker、entity、relation 等
- **类型区分**: inner（核心事实）vs outer（临时状态）

### 3. 溯源优势
- **source 字段**: 明确记录每个原子记忆来自哪些原始消息
- **evidence 映射**: 通过 source 字段可以追溯到 evidence

## 使用方法

### 快速测试（50个样本）
```bash
python scripts/eval_atommem.py --experiment atommem_test --limit 50
```

### 完整评测（1986个样本，约2小时）
```bash
python scripts/eval_atommem.py --experiment atommem_full
```

### 断点续跑
```bash
python scripts/eval_atommem.py --experiment atommem_full --resume
```

### 分析结果
```bash
python scripts/analyze_atommem_results.py data/eval_runs/atommem_test_*/eval_atommem.jsonl
```

## 输出文件

```
data/eval_runs/{experiment}_{timestamp}/
├── eval_atommem.jsonl    # 每行一个 QA 结果
└── summary.json          # 汇总指标
```

每个结果包含：
- msgmem 和 atommem 的召回率、答案、judge 结果
- 按 category 统计的指标
- 完整的 retrieved_ids 用于分析

## 下一步

- [ ] 运行完整评测（1986个样本）
- [ ] 对比 baseline（完整上下文）、RAG（msgmem）、AtomMem（atommem）三种方法
- [ ] 分析不同 category（multi_hop, temporal, open_domain, single_hop）的表现差异
- [ ] 研究 atommem 失败的案例，找出改进方向
