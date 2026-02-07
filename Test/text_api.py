import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

# splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=5)
# docs = """
# 夜色像一块被水洗过的丝绒，从屋檐一路垂到街角。路灯亮起时，光晕在地面摊开，像一圈圈安静的涟漪。便利店的玻璃门“叮”地响了一声，有人拎着热咖啡走出来，白汽在空气里慢慢散开。风不大，却足够把树叶吹得彼此碰撞，发出细碎又温柔的声响。
#
# 清晨的雾气贴着河面缓缓流动，像一条不愿醒来的白色丝带。桥上的行人不多，脚步声被水汽吸走，只剩下轻轻的回响。卖豆浆的小摊刚点起炉火，甜香顺着风飘散开来，一只麻雀跳上栏杆，歪着头看着升起的热气，仿佛在判断今天会不会是个好天气。
#
# 午后的阳光透过百叶窗，在地板上切成整齐的金色条纹。书页被翻动时发出细微的沙沙声，空气里混着纸张与木头的味道。窗外有人练吉他，旋律断断续续，却意外地好听，像是世界在进行一场不太正式、但格外真诚的排练。
#
# 傍晚的天空被涂成橘红和紫蓝的渐变色，云层像慢慢散开的颜料。高楼的玻璃反射出最后一抹日光，街道开始热闹起来，小吃摊升起白烟，油锅滋啦作响。人群在霓虹灯下交错，每个人都带着自己的目的地，却又在同一条路上短暂相遇。
#
# 深夜的城市换了一副面孔。广告牌依旧明亮，但窗户里多半只剩零星的灯光。清洁车缓缓驶过，刷子与地面摩擦出规律的节奏，像一首低声哼唱的摇篮曲。月亮挂在高处，冷冷地照着空旷的十字路口，让整个世界显得既遥远又安静，仿佛下一秒就会重新翻到新的一页。
#
# 远处传来公交车低沉的引擎声，一只猫蹲在路边，尾巴绕着爪子，认真地观察这个世界仿佛刚刚被重启。时间在这一刻变得很慢，慢到可以数清楚呼吸之间的空隙，也慢到让人突然意识到——原来平凡本身，就已经很美了。
# """
#
# chunks = splitter.split_text(docs)
# print(type(chunks))
#
# loader = UnstructuredWordDocumentLoader("")

his = InMemoryChatMessageHistory()

llm = ChatOpenAI(model="qwen3-max-2026-01-23", api_key=os.environ.get("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
resp = llm.invoke("你好你是谁")
print("原始返回：")
print(resp)
print(type(resp))

print("聊天记录")
his.add_message(resp)
print(his.messages)
