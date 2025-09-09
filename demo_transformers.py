from mineru_vl_utils import MinerUClient
from PIL import Image

print("Importing transformers ...")
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

model_path = (
    "/share/jinzhenjiang/hetianyao_MinerU-LVLM-0821_checkpoints_Qwen2VL_2048_16384_thumb-1036_Stage2_100k-500k-400k-300k_6.5ep"
)

print("Loading model ...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)

print("Loading processor ...")
processor = AutoProcessor.from_pretrained(
    model_path,
    use_fast=True,
)

print("Creating client ...")
client = MinerUClient(
    backend="transformers",
    model=model,
    processor=processor,
)

print("Loading image ...")
image_path = "/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_00039896.1983.10545823.pdf_1.jpg"
image = Image.open(image_path).convert("RGB")

print("Extracting ...")
output = client.two_step_extract(image)
print(output)
