# 切分文档 存入向量数据库 写入prompt 找大模型申请回复
import os
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI


class MyDB:
    def __init__(self, collection_name):
        # 存储到内存
        db = chromadb.Client(Settings(allow_reset=True))
        self.collection = db.get_or_create_collection(name=collection_name)
        self.modelClient = OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


    def __getEmbWithBatch(self, texts, batch_size=10):
        all_emb = []
        for i in range (0, len(texts), batch_size):
            cur_text = texts[i:i + batch_size]
            res = self.modelClient.embeddings.create(model="text-embedding-v4", input=cur_text).data
            res = [x.embedding for x in res]
            all_emb.extend(res)
        return all_emb


    def AddDoc(self, texts):
        emb_result = self.__getEmbWithBatch(texts)
        self.collection.add(ids=[f'id{i}' for i in range(len(emb_result))], embeddings=emb_result, documents=texts)


    def search(self, query, top_n=5):
        query_emb = self.__getEmbWithBatch([query])
        docs = self.collection.query(query_emb, n_results=top_n)
        return docs


# 读取文档
with open('../Data/deepseek百度百科.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 切割文档
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=['\n\n', '\n', '。', '？', '！', ',', '、', ''])
texts = splitter.split_text(content)

# 向量化与加入文档
db = MyDB(collection_name='test_ds')
db.AddDoc(texts)

# 搜索答案与提示词嵌入
query = 'deepseek的性能和chatgpt比谁更强？'
search_result = db.search(query=query, top_n=5)
print(search_result)
docs = '\n\n'.join(search_result['documents'][0])

prompt = f'''
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。
----
已知信息:
{docs}
----
用户问：
{query}

请用中文回答用户问题。
'''

answer = db.modelClient.chat.completions.create(
    model = 'qwen3-max',
    messages = [{'role': 'user', 'content': prompt}],
    temperature = 0.5
)

print(answer.choices[0].message.content)