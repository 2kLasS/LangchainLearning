import os
from unstructured.partition.pdf import partition_pdf

cache_dir = "D:/unstructured_cache"
os.environ['UNSTRUCTURED_CACHE_DIR'] = cache_dir
os.environ['HF_HOME'] = cache_dir

pdf_elements = partition_pdf(
    filename='../../文件/978-7-5170-2271-8_1.pdf',
    strategy='hi_res',
    languages=['chi_sim'],

    extract_images_in_pdf=True,
    extract_image_block_types=['Image'],
    extract_image_block_output_dir="./extracted_images",

    max_characters=400,  # 留点余量，防止超 512 Token
    new_after_n_chars=300,  # 达到 300 字符后尝试分块
    combine_text_under_n_chars=100,  # 小于 100 字符的块尝试合并
    chunking_strategy="by_title",
)

for page in pdf_elements:
    print(page)
