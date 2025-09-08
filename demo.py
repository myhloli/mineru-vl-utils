from mineru_vl_utils import MinerUClient
from PIL import Image

if __name__ == "__main__":
    client = MinerUClient(model_name="mineru_dev_250903_2step", server_url="http://llm.bigdata.shlab.tech")
    image = Image.open("/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_00039896.1983.10545823.pdf_1.jpg")
    print(client.two_step_extract(image))
