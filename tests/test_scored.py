"""Mock 单元测试：测试 predict_scored / score 的 logprobs 提取和指标计算逻辑。

策略：
- 真实 Qwen2-VL-2B-Instruct tokenizer（纯 CPU，无需模型权重）
- mock vllm.LLM.generate() 返回值，构造符合 vLLM RequestOutput 结构的对象
- 通过 object.__new__() 绕过 __init__ 中的 vllm import
"""

import math
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from transformers import AutoTokenizer

from mineru_vl_utils.vlm_client.base_client import (
    SamplingParams,
    ScoredOutput,
    compute_confidence_metrics,
)
from mineru_vl_utils.vlm_client.vllm_engine_client import VllmEngineVlmClient


# ---------------------------------------------------------------------------
# Mock vLLM types
# ---------------------------------------------------------------------------


@dataclass
class MockLogprob:
    logprob: float


@dataclass
class MockCompletionOutput:
    text: str
    token_ids: list[int]
    logprobs: list[dict[int, MockLogprob]]
    finish_reason: str = "stop"


@dataclass
class MockRequestOutput:
    outputs: list[MockCompletionOutput]
    finished: bool = True
    # for score (prompt_logprobs)
    prompt_token_ids: list[int] = field(default_factory=list)
    prompt_logprobs: list[dict[int, MockLogprob] | None] = field(default_factory=list)


class MockVllmSamplingParams:
    """可以像 vllm.SamplingParams 一样接受任意 kwargs 并允许后续赋值。"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


@pytest.fixture
def client(tokenizer):
    """构造 VllmEngineVlmClient 实例，绕过 __init__，使用真实 tokenizer。"""
    c = object.__new__(VllmEngineVlmClient)
    c.prompt = "What is the text in the illustrate?"
    c.system_prompt = "You are a helpful assistant."
    c.sampling_params = None
    c.text_before_image = False
    c.allow_truncated_content = False
    c.tokenizer = tokenizer
    c.model_max_length = 4096
    c.VllmSamplingParams = MockVllmSamplingParams
    c.batch_size = 0
    c.use_tqdm = False
    c.debug = False
    c.vllm_llm = MagicMock()
    return c


# ---------------------------------------------------------------------------
# compute_confidence_metrics 纯函数测试
# ---------------------------------------------------------------------------


class TestComputeConfidenceMetrics:
    def test_empty(self):
        ppl, min_lp, ratio = compute_confidence_metrics([])
        assert ppl == float("inf")
        assert min_lp == float("-inf")
        assert ratio == 1.0

    def test_single_token(self):
        ppl, min_lp, ratio = compute_confidence_metrics([-1.0])
        assert ppl == pytest.approx(math.exp(1.0))
        assert min_lp == -1.0
        assert ratio == 0.0  # -1.0 > -2.0

    def test_all_low_confidence(self):
        ppl, min_lp, ratio = compute_confidence_metrics([-3.0, -4.0, -5.0])
        assert ratio == 1.0
        assert min_lp == -5.0

    def test_mixed(self):
        logprobs = [-0.5, -1.0, -3.0, -0.2]
        ppl, min_lp, ratio = compute_confidence_metrics(logprobs)
        expected_ppl = math.exp(-sum(logprobs) / len(logprobs))
        assert ppl == pytest.approx(expected_ppl)
        assert min_lp == -3.0
        assert ratio == 0.25  # only -3.0 < -2.0

    def test_custom_threshold(self):
        _, _, ratio = compute_confidence_metrics([-0.5, -1.0, -3.0, -0.2], threshold=-0.8)
        assert ratio == 0.5  # -1.0 and -3.0 < -0.8


# ---------------------------------------------------------------------------
# predict_scored 测试
# ---------------------------------------------------------------------------


class TestPredictScored:
    def test_basic(self, client):
        """验证 predict_scored 正确提取 logprobs 并计算指标。"""
        # 用 tokenizer 编码一段文本，模拟模型输出
        generated_text = "Hello, world!"
        token_ids = client.tokenizer.encode(generated_text, add_special_tokens=False)
        # 为每个 token 分配 logprob
        fake_logprobs = [-0.1 * (i + 1) for i in range(len(token_ids))]

        mock_output = MockRequestOutput(
            outputs=[
                MockCompletionOutput(
                    text=generated_text,
                    token_ids=token_ids,
                    logprobs=[
                        {tid: MockLogprob(lp)} for tid, lp in zip(token_ids, fake_logprobs)
                    ],
                )
            ]
        )
        client.vllm_llm.generate.return_value = [mock_output]

        result = client.predict_scored(image=None, prompt="Describe this.")
        assert isinstance(result, ScoredOutput)
        assert result.text == generated_text
        assert result.token_ids == token_ids
        assert result.logprobs == pytest.approx(fake_logprobs)

        expected_ppl = math.exp(-sum(fake_logprobs) / len(fake_logprobs))
        assert result.perplexity == pytest.approx(expected_ppl)
        assert result.min_logprob == min(fake_logprobs)

    def test_logprobs_enabled_in_sampling_params(self, client):
        """验证 batch_predict_scored 在调 generate 前设置了 logprobs=0。"""
        mock_output = MockRequestOutput(
            outputs=[
                MockCompletionOutput(
                    text="x",
                    token_ids=[100],
                    logprobs=[{100: MockLogprob(-0.5)}],
                )
            ]
        )
        client.vllm_llm.generate.return_value = [mock_output]

        client.predict_scored(image=None, prompt="test")

        # 检查传给 generate 的 sampling_params 有 logprobs=0
        call_kwargs = client.vllm_llm.generate.call_args
        sp_list = call_kwargs.kwargs.get("sampling_params") or call_kwargs[1].get("sampling_params")
        for sp in sp_list:
            assert hasattr(sp, "logprobs") and sp.logprobs == 0

    def test_batch(self, client):
        """batch_predict_scored 多个样本。"""
        outputs = []
        for i in range(3):
            text = f"text_{i}"
            tids = client.tokenizer.encode(text, add_special_tokens=False)
            lps = [-0.2 * (j + 1) for j in range(len(tids))]
            outputs.append(
                MockRequestOutput(
                    outputs=[
                        MockCompletionOutput(
                            text=text,
                            token_ids=tids,
                            logprobs=[{t: MockLogprob(lp)} for t, lp in zip(tids, lps)],
                        )
                    ]
                )
            )
        client.vllm_llm.generate.return_value = outputs

        results = client.batch_predict_scored(
            images=[None, None, None],
            prompts=["p1", "p2", "p3"],
        )
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.text == f"text_{i}"
            assert r.perplexity > 0


# ---------------------------------------------------------------------------
# score (teacher forcing) 测试
# ---------------------------------------------------------------------------


class TestScore:
    def test_basic(self, client, tokenizer):
        """验证 score 使用 prompt_logprobs 正确提取 scored_text 部分的 logprobs。"""
        scored_text = "This is a test answer."

        # 用真实 tokenizer 计算 token 数量差
        messages = client.build_messages("Describe this.", 0)
        messages_with_assistant = messages + [{"role": "assistant", "content": scored_text}]

        full_prompt = tokenizer.apply_chat_template(
            messages_with_assistant, tokenize=False, add_generation_prompt=False
        )
        base_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        full_token_ids = tokenizer.encode(full_prompt)
        base_token_ids = tokenizer.encode(base_prompt)
        scored_token_count = len(full_token_ids) - len(base_token_ids)
        assert scored_token_count > 0, "scored_text should produce at least 1 token"

        # 为整个 prompt 构造 prompt_logprobs
        # prompt_logprobs[0] = None (BOS), 其余为 {token_id: Logprob}
        prompt_logprobs: list[dict[int, MockLogprob] | None] = [None]  # BOS
        for i in range(1, len(full_token_ids)):
            tid = full_token_ids[i]
            # scored_text 部分给低一点的 logprob，前面部分给高的
            if i >= len(base_token_ids):
                lp = -0.5  # scored part
            else:
                lp = -0.01  # prompt part (high confidence)
            prompt_logprobs.append({tid: MockLogprob(lp)})

        mock_output = MockRequestOutput(
            outputs=[
                MockCompletionOutput(
                    text="",  # score 模式下 generate 的输出文本不重要
                    token_ids=[full_token_ids[-1]],  # max_tokens=1 产生的无关 token
                    logprobs=[{full_token_ids[-1]: MockLogprob(-0.1)}],
                )
            ],
            prompt_token_ids=full_token_ids,
            prompt_logprobs=prompt_logprobs,
        )
        client.vllm_llm.generate.return_value = [mock_output]

        result = client.score(image=None, scored_text=scored_text, prompt="Describe this.")

        assert isinstance(result, ScoredOutput)
        assert result.text == scored_text
        assert len(result.token_ids) == scored_token_count
        assert len(result.logprobs) == scored_token_count
        # 所有 scored part 的 logprob 都是 -0.5
        assert all(lp == pytest.approx(-0.5) for lp in result.logprobs)
        assert result.perplexity == pytest.approx(math.exp(0.5))

    def test_prompt_logprobs_enabled(self, client):
        """验证 score 在调 generate 前设置了 prompt_logprobs=0 和 max_tokens=1。"""
        scored_text = "answer"
        messages = client.build_messages("q", 0)
        messages_with = messages + [{"role": "assistant", "content": scored_text}]
        full_prompt = client.tokenizer.apply_chat_template(
            messages_with, tokenize=False, add_generation_prompt=False
        )
        full_ids = client.tokenizer.encode(full_prompt)

        prompt_logprobs: list[dict[int, MockLogprob] | None] = [None]
        for i in range(1, len(full_ids)):
            prompt_logprobs.append({full_ids[i]: MockLogprob(-0.3)})

        mock_output = MockRequestOutput(
            outputs=[
                MockCompletionOutput(
                    text="", token_ids=[full_ids[-1]],
                    logprobs=[{full_ids[-1]: MockLogprob(-0.1)}],
                )
            ],
            prompt_token_ids=full_ids,
            prompt_logprobs=prompt_logprobs,
        )
        client.vllm_llm.generate.return_value = [mock_output]

        client.score(image=None, scored_text=scored_text, prompt="q")

        call_kwargs = client.vllm_llm.generate.call_args
        sp_list = call_kwargs.kwargs.get("sampling_params") or call_kwargs[1].get("sampling_params")
        for sp in sp_list:
            assert sp.prompt_logprobs == 0
            assert sp.max_tokens == 1

    def test_correct_label_lower_ppl_than_random(self, client, tokenizer):
        """模拟：正确标注的 PPL 应低于随机文本的 PPL。"""
        correct_text = "The quick brown fox"
        random_text = "asdf qwer zxcv bnm"

        def mock_score_one(scored_text, logprob_value):
            messages = client.build_messages("q", 0)
            messages_with = messages + [{"role": "assistant", "content": scored_text}]
            full_prompt = tokenizer.apply_chat_template(
                messages_with, tokenize=False, add_generation_prompt=False
            )
            full_ids = tokenizer.encode(full_prompt)

            prompt_logprobs: list[dict[int, MockLogprob] | None] = [None]
            for i in range(1, len(full_ids)):
                prompt_logprobs.append({full_ids[i]: MockLogprob(logprob_value)})

            return MockRequestOutput(
                outputs=[
                    MockCompletionOutput(
                        text="", token_ids=[full_ids[-1]],
                        logprobs=[{full_ids[-1]: MockLogprob(-0.1)}],
                    )
                ],
                prompt_token_ids=full_ids,
                prompt_logprobs=prompt_logprobs,
            )

        # 正确标注 → 高置信度 (logprob 接近 0)
        client.vllm_llm.generate.return_value = [mock_score_one(correct_text, -0.3)]
        result_correct = client.score(image=None, scored_text=correct_text, prompt="q")

        # 随机文本 → 低置信度 (logprob 很负)
        client.vllm_llm.generate.return_value = [mock_score_one(random_text, -4.0)]
        result_random = client.score(image=None, scored_text=random_text, prompt="q")

        assert result_correct.perplexity < result_random.perplexity
        assert result_correct.low_confidence_ratio < result_random.low_confidence_ratio
