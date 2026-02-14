from langchain_unstructured import UnstructuredLoader


# 实际效果不太好，最好把文档手动清洗
loader = UnstructuredLoader(
    file_path="../文件/2020-03-17__厦门灿坤实业股份有限公司__200512__闽灿坤__2019年__年度报告.pdf",
    languages=["chi_sim"],
    strategy="hi_res",
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=2048,
    new_after_n_chars=1536,
    combine_text_under_n_chars=1024
)
chunks = loader.load()

print(f"Chunks长度: {len(chunks)}")
for chunk in chunks[10:20]:
    print(chunk)
