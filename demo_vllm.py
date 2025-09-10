from mineru_vl_utils import MinerUClient
from PIL import Image

print("Importing vllm...")
from vllm import LLM

model_path = (
    "/share/jinzhenjiang/hetianyao_MinerU-LVLM-0821_checkpoints_Qwen2VL_2048_16384_thumb-1036_Stage2_100k-500k-400k-300k_6.5ep"
)


def get_client_example1():
    """
    In this example, user can control every
    parameter of the llm instance.
    """
    print("Loading vllm LLM ...")
    llm = LLM(model=model_path)

    print("Creating client ...")
    client = MinerUClient(
        backend="vllm-engine",
        vllm_llm=llm,
    )
    return client


def get_client_example2():
    """
    In this example, user only need to provide model_path.
    The model and processor will be automatically initialized
    with default parameters.
    """
    print("Creating client ...")
    client = MinerUClient(
        backend="vllm-engine",
        model_path=model_path,
    )
    return client


client = get_client_example1()

print("Loading image ...")
image_path = "/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_00039896.1983.10545823.pdf_1.jpg"
image = Image.open(image_path).convert("RGB")

print("Extracting ...")
output = client.two_step_extract(image)
print(output)
