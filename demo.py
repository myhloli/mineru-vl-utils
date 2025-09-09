import time

from mineru_vl_utils import MinerUClient
from PIL import Image

test_image_paths = """
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_00039896.1983.10545823.pdf_1.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-60368448.pdf_343.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-60403612.pdf_179.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-60482015.pdf_56.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-60599898.pdf_30.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-60832903.pdf_88.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61323717.pdf_203.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61465568.pdf_186.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61467079.pdf_40.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61510621.pdf_161.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61510863.pdf_73.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61511646.pdf_188.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61513005.pdf_179.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61518266.pdf_149.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61520553.pdf_328.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61520612.pdf_140.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61520779.pdf_76.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61520788.pdf_391.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61520814.pdf_185.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61521185.pdf_167.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61521384.pdf_750.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61521429.pdf_358.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61521600.pdf_210.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61522126.pdf_206.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61522235.pdf_170.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61523162.pdf_276.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61524043.pdf_156.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61562126.pdf_80.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61564860.pdf_81.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61566094.pdf_55.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61569294.pdf_128.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61569751.pdf_155.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61570078.pdf_315.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61570552.pdf_129.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61571237.pdf_81.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61571259.pdf_239.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-61571884.pdf_94.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63674848.pdf_101.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63675767.pdf_66.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63677793.pdf_332.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63678132.pdf_161.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63684042.pdf_172.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63684502.pdf_263.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63685157.pdf_105.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63685611.pdf_222.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63686103.pdf_55.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63686436.pdf_57.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63687685.pdf_106.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63688043.pdf_110.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63688445.pdf_489.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63690689.pdf_134.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63708140.pdf_170.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63709763.pdf_72.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63710191.pdf_336.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63710614.pdf_149.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63711094.pdf_34.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_dianzishu_zhongwenzaixian-o.O-63711094.pdf_56.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_enbook-zlib-o.O-15322190.pdf_138.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_enbook-zlib-o.O-17208435.pdf_105.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_enbook-zlib-o.O-17208435.pdf_57.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_enbook-zlib-o.O-17342542.pdf_25.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_enbook-zlib-o.O-17761417.pdf_894.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_enbook-zlib-o.O-19091739.pdf_181.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_enbook-zlib-o.O-19221575.pdf_1173.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_enbook-zlib-o.O-21353024.pdf_39.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_enbook-zlib-o.O-21882649.pdf_3.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_enbook-zlib-o.O-22206811.pdf_5.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-adsc.201190003.pdf_6.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-ajhb.10190.pdf_5.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-ana.20363.pdf_5.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-bf00151572.pdf_3.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-bf00326833.pdf_3.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-bf03175144.pdf_4.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-ceat.200407001.pdf_3.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-ceat.200600266.pdf_2.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-ceat.200600410.pdf_5.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-ceat.200600410.pdf_8.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-chem.200700118.pdf_7.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-chem.200700133.pdf_6.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-chem.200700285.pdf_4.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-chem.200701412.pdf_10.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-chem.200701981.pdf_9.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-chin.200305072.pdf_1.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-chin.200427104.pdf_1.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-chin.201023119.pdf_1.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-chin.201025015.pdf_1.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-dneu.20401.pdf_2.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-dneu.20833.pdf_13.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-dvdy.10165.pdf_7.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-hup.777.pdf_7.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-hup.777.pdf_8.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-hup.908.pdf_5.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-hyp.1040.pdf_9.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-hyp.1092.pdf_1.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-ijc.22820.pdf_4.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-ijc.22994.pdf_3.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-ijc.22999.pdf_3.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-j.1365-2117.1999.00104.x.pdf_11.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-j.1365-2117.2002.00179.x.pdf_10.jpg
/share/jinzhenjiang/OmniDocBench/v1_0/docstructbench_llm-raw-scihub-o.O-j.1365-2818.2000.00739.x.pdf_4.jpg
""".strip().split("\n")

if __name__ == "__main__":
    client = MinerUClient(model_name="mineru_dev_250903_2step", server_url="http://llm.bigdata.shlab.tech")

    # Test1
    image = Image.open(test_image_paths[0])
    print("Processing 1 image...")
    begin = time.time()
    output = client.two_step_extract(image)
    elapsed = time.time() - begin
    print(output)
    print(f"Time for extracting 1 image: {elapsed:.2f} seconds")

    # Test2
    images = []
    for image_path in test_image_paths[:10]:
        images.append(Image.open(image_path))

    print(f"Processing {len(images)} images...")
    begin = time.time()
    outputs = client.batch_two_step_extract(images)
    elapsed = time.time() - begin
    for idx, output in enumerate(outputs):
        print(f"Output for image {idx}:")
        print(output)
    print(f"Time for extracting {len(images)} images: {elapsed:.2f} seconds")
    print("Images per second:", len(images) / elapsed)

    # Test3
    images = []
    for image_path in test_image_paths:
        images.append(Image.open(image_path))

    print(f"Processing {len(images)} images...")
    begin = time.time()
    outputs = client.batch_two_step_extract(images)
    elapsed = time.time() - begin
    for idx, output in enumerate(outputs):
        print(f"Output for image {idx}:")
        print(output)
    print(f"Time for extracting {len(images)} images: {elapsed:.2f} seconds")
    print("Images per second:", len(images) / elapsed)
