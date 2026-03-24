# Scored Output 集成测试指南

## 1. 背景

为 `mineru-vl-utils` 的 VlmClient 新增了 logprobs / PPL 支持，用于难例挖掘和标注质量校验。

### 新增 API

| 方法 | 用途 | 引擎 |
|------|------|------|
| `predict_scored()` / `batch_predict_scored()` | 生成文本 + 置信度指标 | vllm-engine (sync) |
| `score()` / `batch_score()` | 对已有标注做 teacher forcing 评估 PPL | vllm-engine (sync) |
| `aio_predict_scored()` / `aio_batch_predict_scored()` | 异步生成 + 置信度 | vllm-async-engine |
| `aio_score()` / `aio_batch_score()` | 异步 teacher forcing | vllm-async-engine |

### 返回类型 `ScoredOutput`

```python
@dataclass
class ScoredOutput:
    text: str                    # 文本内容
    token_ids: list[int]         # token ID 序列
    logprobs: list[float]        # 每个 token 的 logprob（自然对数），范围 (-∞, 0]
    perplexity: float            # exp(-mean(logprobs))，范围 [1, +∞)，越低越确定
    min_logprob: float           # min(logprobs)，最不确定的 token
    low_confidence_ratio: float  # logprob < threshold 的 token 占比，范围 [0, 1]
```

### 修改的文件

| 文件 | 改动 |
|------|------|
| `vlm_client/base_client.py` | +ScoredOutput, +compute_confidence_metrics(), +8个基类stub方法 |
| `vlm_client/vllm_engine_client.py` | 实现 predict_scored, batch_predict_scored, score, batch_score |
| `vlm_client/vllm_async_engine_client.py` | 实现 aio_predict_scored, aio_batch_predict_scored, aio_score, aio_batch_score |
| `vlm_client/__init__.py` | 导出 ScoredOutput, compute_confidence_metrics |
| `tests/test_scored.py` | Mock 单元测试（已全部通过，11/11） |

## 2. 环境准备

需要 GPU 机器，安装 vllm >= 0.10.1.1。

```bash
# 安装项目 + vllm 依赖
pip install -e ".[vllm]"

# 或者如果已有 vllm 环境，只装项目本身
pip install -e .
```

## 3. 集成测试脚本

### 3.1 测试 predict_scored（生成 PPL）

```python
from vllm import LLM
from mineru_vl_utils.vlm_client import new_vlm_client

# 初始化
llm = LLM(model="Qwen/Qwen2-VL-2B-Instruct", max_model_len=4096)
client = new_vlm_client("vllm-engine", vllm_llm=llm)

# 准备一张测试图片
from PIL import Image
image = Image.open("test_image.png")  # 替换为实际图片路径

# --- 生成 PPL ---
result = client.predict_scored(image=image)
print(f"text: {result.text}")
print(f"token count: {len(result.token_ids)}")
print(f"perplexity: {result.perplexity:.2f}")
print(f"min_logprob: {result.min_logprob:.4f}")
print(f"low_confidence_ratio: {result.low_confidence_ratio:.2%}")

# 验证点：
# 1. result.text 应与 client.predict(image=image) 返回一致（相同采样参数下）
# 2. len(result.token_ids) == len(result.logprobs) == 生成 token 数
# 3. result.perplexity > 0 且有限
# 4. 所有 logprob <= 0
assert len(result.token_ids) == len(result.logprobs)
assert result.perplexity > 0
assert all(lp <= 0 for lp in result.logprobs)
print("predict_scored: PASSED")
```

### 3.2 测试 predict_scored 与 predict 一致性

```python
# 固定温度为 0 确保确定性输出
from mineru_vl_utils.vlm_client import SamplingParams
sp = SamplingParams(temperature=0.0)

text_only = client.predict(image=image, sampling_params=sp)
scored = client.predict_scored(image=image, sampling_params=sp)

assert text_only == scored.text, f"Mismatch:\n  predict: {text_only}\n  scored:  {scored.text}"
print(f"一致性检查 PASSED, text: {scored.text[:80]}...")
```

### 3.3 测试 score（teacher forcing 评估 PPL）

```python
# 先用模型生成一个"正确"答案
correct_answer = client.predict(image=image, sampling_params=SamplingParams(temperature=0.0))

# 用 score 评估正确答案的 PPL
result_correct = client.score(image=image, scored_text=correct_answer)
print(f"正确答案 PPL: {result_correct.perplexity:.2f}")
print(f"正确答案 low_confidence_ratio: {result_correct.low_confidence_ratio:.2%}")

# 用 score 评估一段随机文本的 PPL
result_random = client.score(image=image, scored_text="asdf qwer zxcv 1234 随机文本")
print(f"随机文本 PPL: {result_random.perplexity:.2f}")
print(f"随机文本 low_confidence_ratio: {result_random.low_confidence_ratio:.2%}")

# 验证点：正确答案的 PPL 应显著低于随机文本
assert result_correct.perplexity < result_random.perplexity, (
    f"Expected correct PPL ({result_correct.perplexity:.2f}) < random PPL ({result_random.perplexity:.2f})"
)
print("score 区分度检查: PASSED")
```

### 3.4 测试 batch 方法

```python
images = [image] * 4
scored_texts = [correct_answer, "wrong answer 1", "错误答案2", "asdf"]

results = client.batch_score(images=images, scored_texts=scored_texts)
for i, r in enumerate(results):
    print(f"  [{i}] PPL={r.perplexity:8.2f}  low_conf={r.low_confidence_ratio:.2%}  text={r.text[:40]}...")

# 第一个（正确答案）PPL 应最低
assert results[0].perplexity == min(r.perplexity for r in results)
print("batch_score: PASSED")
```

### 3.5 测试 vllm-async-engine（如需要）

```python
import asyncio
from vllm.v1.engine.async_llm import AsyncLLM

async def test_async():
    async_llm = ... # 按实际方式初始化 AsyncLLM
    client = new_vlm_client("vllm-async-engine", vllm_async_llm=async_llm)

    result = await client.aio_predict_scored(image=image)
    print(f"async predict_scored PPL: {result.perplexity:.2f}")

    result = await client.aio_score(image=image, scored_text="test text")
    print(f"async score PPL: {result.perplexity:.2f}")

asyncio.run(test_async())
```

## 4. 预期指标范围参考

基于 Qwen2-VL-2B-Instruct（vocab_size=151,643）：

| 场景 | perplexity | low_confidence_ratio |
|------|-----------|---------------------|
| 模型对正确 OCR 结果做 score | 1 ~ 10 | < 5% |
| 模型对相似但有错误的标注做 score | 10 ~ 100 | 5% ~ 30% |
| 模型对完全不相关的文本做 score | 100 ~ 数千 | > 50% |
| predict_scored 简单清晰图片 | 1 ~ 5 | < 2% |
| predict_scored 模糊/复杂图片 | 5 ~ 50+ | 5% ~ 20%+ |

## 5. 排查清单

- [ ] `predict_scored` 返回的 text 与 `predict` 一致（temperature=0）
- [ ] `token_ids` 和 `logprobs` 长度相同
- [ ] 所有 `logprobs` 值 <= 0
- [ ] `perplexity` > 0 且有限
- [ ] `score` 对正确标注的 PPL < 对随机文本的 PPL
- [ ] `batch_predict_scored` / `batch_score` 结果数量与输入一致
- [ ] `low_confidence_threshold` 参数生效（改变 low_confidence_ratio）
