import asyncio

from mineru_vl_utils import MinerUClient
from PIL import Image

print("Importing transformers ...")
from transformers import AutoProcessor

print("Importing vllm...")
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

model_path = (
    "/share/jinzhenjiang/hetianyao_MinerU-LVLM-0821_checkpoints_Qwen2VL_2048_16384_thumb-1036_Stage2_100k-500k-400k-300k_6.5ep"
)

print("Loading processor ...")
processor = AutoProcessor.from_pretrained(
    model_path,
    use_fast=True,
)

print("Creating async_llm ...")
async_llm = AsyncLLM.from_engine_args(AsyncEngineArgs(model_path))

print("Creating client ...")
client = MinerUClient(
  backend="vllm-async-engine",
  vllm_async_llm=async_llm,
  processor=processor,
)

print("Loading image ...")
image_path = "/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_00039896.1983.10545823.pdf_1.jpg"
image = Image.open(image_path).convert("RGB")

print("Extracting ...")

async def main():
    output = await client.aio_two_step_extract(image)
    print(output)

asyncio.run(main())

async_llm.shutdown()
