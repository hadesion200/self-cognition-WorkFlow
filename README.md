# QA 评测工作流（三段式，按 jsonl 知识领域独立评测）

基于本地 jsonl 问答数据的 LLM 问答质量评测流水线：

1. **预处理**：扫描 `datasets/` 下所有 jsonl，每个文件视为一个独立知识领域；去重、去掉 assistant 内容里的思维链，生成精简数据集；
2. **被测模型问答**：按配置并发调用被测模型 API，每个领域单独输出回答文件，回答与数据集原答案一起实时落盘；
3. **评分 + 报告**：评分模型逐个领域打分，实时保存分数与理由；每个领域独立生成一份 Markdown 报告，满分 **1 分**。

所有超参数集中在 YAML，`sh run.sh` 一键运行；每条结果实时追加写入，中断后重新运行自动从断点续跑。

---

## 1. 快速开始（不需要真实 API）

```bash
# WSL / Linux / macOS / Git Bash
sh run.sh --mock

# Windows CMD / PowerShell
run.bat --mock
```

`config.mock.yaml` 使用内置模拟模型，读取 `sample_data/` 里的三个示例领域，跑通全部阶段。

每个 jsonl 最终只产出 **三个文件**：

```
outputs/
├── self_cognition_model_response.jsonl    # 被测模型回答（含原答案）
├── self_cognition_model_judge.jsonl       # 评分模型打分 + 理由
└── self_cognition_report.md               # 该知识领域的最终报告
```

文件名规则：

| 产物 | 文件名 |
| --- | --- |
| 被测模型回答 | `<数据集文件名>_model_response.jsonl` |
| 评分模型打分 | `<数据集文件名>_model_judge.jsonl` |
| 最终报告 | `<数据集文件名>_report.md` |

例如 `datasets/TX-CW-self-cognition.jsonl` 会生成：

```
outputs/TX-CW-self-cognition_model_response.jsonl
outputs/TX-CW-self-cognition_model_judge.jsonl
outputs/TX-CW-self-cognition_report.md
```

> 模拟模式的分数由内置 mock 按文本重合度 + 稳定扰动生成，只用于验证流程、断点和报告，不代表真实模型能力。

### 进度显示

阶段 2、3 在终端中使用**单行动态进度条**，原地刷新，不会逐条刷屏：

```
  self_cognition [########----------------] 8/17  47% | 成功 8 失败 0 | 4.2 条/秒 | ETA 2s
```

- 只有 stdout 是 TTY 时才启用动态刷新；重定向到文件或管道时自动关闭，只保留每个领域的开始/结束日志。
- 失败样本、重试信息会临时打断进度条并重新绘制，历史日志仍可见。
- 每个领域开始、结束各一行日志，方便回看。

### 在 WSL（推荐）下运行

```bash
wsl
cd /mnt/c/Users/18227/Desktop/中继数据/self-cognition-WorkFlow

python3 --version
python3 -c "import yaml, requests"       # 若报错，按下面安装依赖

sh run.sh --mock
```

如果 WSL 里缺少依赖（Ubuntu 24.04 默认禁止 `pip install` 到系统 Python，建议用虚拟环境）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
sh run.sh --mock
```

`run.sh` 会优先使用 `python`，没有时自动回退到 `python3`；也可以用 `PYTHON=/path/to/python sh run.sh` 指定解释器。数据集很大时，建议把项目复制到 WSL 的 Linux 文件系统（如 `~/qa-eval`）再跑，`/mnt/c` 的磁盘 I/O 会明显更慢。

---

## 2. 接入真实模型

编辑 `config.yaml`：

```yaml
model_under_test:
  mode: api
  name: deepseek-v4.1-flash        # 仅用于报告展示；文件名使用数据集名
  base_url: http://your-host:8000/v1
  api_key: ""                       # 推荐留空，改用环境变量
  api_key_env: MODEL_API_KEY
  model: deepseek-v4.1-flash-1017
  temperature: 1
  max_tokens: 8192

judge:
  mode: api
  name: judge-model
  base_url: https://api.moonshot.cn/v1
  api_key_env: MOONSHOT_API_KEY
  model: kimi-k2.6
  temperature: 1                    # kimi-k2.6 只接受 temperature=1
  max_tokens: 4096
```

设置密钥：

```bash
export MODEL_API_KEY="..."
export MOONSHOT_API_KEY="..."
```

也可以直接填 `api_key: "sk-..."`，但不要把含密钥的配置提交到公开仓库。

然后：

```bash
sh run.sh
```

### 常见请求差异

- 某些模型只接受 `temperature=1`：把该模型配置改成 1，或用 `drop_keys: ["temperature"]` 删除该字段。
- 某些服务要求 `max_completion_tokens`：用 `extra_body` 增加字段，并用 `drop_keys` 删除 `max_tokens`。
- 关闭思考模式（如 Qwen/vLLM）：`extra_body: {"chat_template_kwargs": {"enable_thinking": false}}`。

---

## 3. 数据格式

每个 `.jsonl` 一行一条记录，最少包含 `messages`：

```json
{"id": "sc_001", "category": "self_cognition", "messages": [
  {"role": "user", "content": "你是谁？"},
  {"role": "assistant", "content": "<think>内部推理……</think>\n\n我是……"}
]}
```

预处理规则：

- 最后一条 `assistant` 消息 → 数据集原答案（参考答案）；
- 最后一条 `user` 消息 → 问题；
- 如果数据以 `user` 结尾（没有最终 assistant），保留为“无参考答案”的待答题；
- `<think>`、`<thinking>`、`<analysis>`、`<reasoning>`、`[think]…[/think]`、`<|begin_of_thought|>…` 等思维链会被删除；
- `assistant` 消息里的 `reasoning_content` / `reasoning` 字段不会进入答案；
- 默认在每个 jsonl 内部按问题去重，可通过 `dedupe_scope: global` 改为跨文件去重；
- 没有 `user` 的记录会被跳过，并在预处理统计中计入“无效”。

预处理后会打印每个文件的原始行数、保留数、去重数、无效数和坏行数，并生成 `work/prepared/_manifest.json`。

---

## 4. 目录与脚本

```
self-cognition-WorkFlow/
├── config.yaml              # 真实 API 配置
├── config.mock.yaml         # 模拟测试配置
├── run.sh / run.bat         # 一键运行
├── qa_common.py             # 配置、CoT 清洗、断点存储、API/mock、进度条
├── qa_prepare.py            # 阶段 1：预处理（每个 jsonl 一个领域）
├── qa_generate.py           # 阶段 2：按领域并发问答
├── qa_score.py              # 阶段 3：按领域评分
├── qa_report.py             # 阶段 4：按领域生成 Markdown 报告
├── sample_data/             # 模拟测试数据
├── datasets/                # 真实数据
├── work/prepared/           # 预处理中间结果
└── outputs/                 # 每个领域三份最终产物
```

常用命令：

```bash
sh run.sh                          # 全流程（config.yaml）
sh run.sh --config my.yaml         # 指定配置
sh run.sh --mock                   # 模拟测试
sh run.sh --fresh                  # 清空旧结果后重跑
python qa_generate.py --config config.yaml   # 单独重跑某一阶段
```

`run.sh` 分阶段调用四个 Python 文件。某个领域未完成时，阶段脚本返回 3，但 `run.sh` 会继续处理其他领域并生成已完成领域的报告，最后返回非 0，提示可以续跑。

---

## 5. 断点续跑

- 阶段 2 / 3 每完成一条就 `append + flush + fsync`，进程被 Ctrl+C 或杀掉后，已完成结果都在文件里；
- 重新执行 `sh run.sh`（或对应阶段脚本）时：
  - 已成功且输入未变化的样本直接跳过；
  - 上次失败（`ok: false`）的样本在 `retry_failed: true` 时重试；
  - 只有在 `--fresh` 时才会清空重来。
- 每条回答/评分都带 `input_hash`。如果改了模型、问题、参考答案、评分权重或最大上下文，对应样本会自动重新生成，而不会误用旧结果。
- 进程被强杀可能留下没有换行符的半行；下一次写入前会自动补换行并忽略坏行，不会污染后续记录。
- 每个领域独立判断完成度：领域 A 失败不会阻塞领域 B 的评分和报告。

> 并发写入由主线程统一落盘，worker 只负责 API 调用，不会出现多线程同时写同一文件导致的半行损坏。

如果某个领域的部分样本因内容审核、超长等原因始终失败，确认无法恢复后可以对该领域已完成部分评分：

```bash
python qa_score.py --config config.yaml --allow-partial
python qa_report.py --config config.yaml
```

此时该领域报告会标记为“部分完成”，并在汇总表中显示未答/未评数量。

---

## 6. 输出字段

### `<数据集>_model_response.jsonl`

| 字段 | 说明 |
| --- | --- |
| `key` | 领域 + 数据 id（或问题哈希），全局唯一 |
| `domain` | 来源 jsonl 文件对应的知识领域 |
| `question` | 问题 |
| `reference_answer` | 数据集原答案（已去思维链） |
| `answer` | 被测模型回答（已去思维链） |
| `ok` / `error` | 是否成功 / 失败原因 |
| `input_hash` | 输入指纹，用于断点校验 |
| `model` / `model_id` | 被测模型 / 报告展示 ID |
| `latency_ms` / `usage` | 耗时与 token 用量 |
| `source` | 来源文件与行号 |
| `created_at` | 生成时间 |

### `<数据集>_model_judge.jsonl`

在回答字段基础上增加：

| 字段 | 说明 |
| --- | --- |
| `scores` | `correctness/completeness/relevance/faithfulness/clarity`，均为 0~1 |
| `overall` | 按权重加权的综合分，0~1 |
| `reason` | 评分理由 |
| `judge_model` | 评分模型 |
| `input_hash` | 问题 + 原答案 + 回答 + 权重的指纹 |

### `<数据集>_report.md`

只针对当前领域，包含：

- 综合得分（满分 1 分）、中位数、最低/最高、标准差、通过率、覆盖率；
- 各维度均分与权重；
- 综合分分布；
- 可选的 `category` 字段汇总；
- 低分案例（问题、维度分、理由）；
- 报告状态：完整 / 部分完成。

---

## 7. 超参数一览（`config.yaml`）

| 参数 | 作用 |
| --- | --- |
| `paths.data_dir` | 原始数据集目录 |
| `paths.prepared_dir` | 预处理输出目录 |
| `paths.output_dir` | 回答/评分/报告输出目录 |
| `dataset.recursive` | 是否递归扫描子目录 |
| `dataset.dedupe` / `dedupe_scope` / `dedupe_key` | 去重开关、范围、依据 |
| `dataset.strip_cot` | 是否去除思维链 |
| `dataset.multi_turn` | 是否把多轮上下文一并发给被测模型 |
| `dataset.max_records_per_file` | 每个文件最多处理条数（0 不限，便于试跑） |
| `model_under_test.*` | 被测模型 API、temperature、max_tokens、超时、重试 |
| `judge.*` | 评分模型 API 与参数 |
| `runtime.workers` | 每个领域的并发数 |
| `runtime.max_context_chars` | 最大上下文字符数（超出会截断较早内容） |
| `runtime.progress_interval_sec` | 进度条刷新间隔，默认 0.2 秒 |
| `runtime.resume` / `retry_failed` | 断点续跑 / 重试失败样本 |
| `runtime.request_interval_sec` | 请求最小间隔，限流时使用 |
| `runtime.fsync_every` | 每 N 条执行一次 fsync |
| `scoring.pass_score` | 通过线（0~1） |
| `scoring.weights` | 各维度权重（自动归一化） |
| `report.bins` / `grade_bands` | 分布区间 / 等级划分 |
| `report.top_n_bad_cases` | 报告中展示的低分案例数量 |

---

## 8. 注意事项

- **评分模型建议不要与被测模型相同**，否则容易出现自评偏高；报告中保留 `judge_model` 以便追溯。
- LLM 评分存在波动，生产使用建议：固定评分模型版本、对低分案例人工抽检、对关键领域重复评分取均值。
- 请确认 `max_tokens` 足够大。部分推理模型会把 token 预算耗在思维链上，导致 `content` 为空；流水线会给出明确报错并建议调大 `max_tokens`。
- 评分请求已按 `max_context_chars` 对问题、参考答案、回答分别截断，避免超长样本把整个 prompt 撑爆。
- 本流程不依赖旧版 `qa_eval_pipeline.py`；如果目录里还保留旧脚本，可以按需删除，不要与新的 `outputs/` 混用。

---

## 9. 依赖

```bash
pip install -r requirements.txt
```

Python 3.9+；依赖 `requests`、`PyYAML`。
