import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


llm = ChatOpenAI(model="qwen3-max-2026-01-23", api_key=os.environ.get("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# 情感分析 问题归类 紧急程度分析
analysis_emotion_prompt = ChatPromptTemplate.from_template("""
    根据以下内容，分析情感类型（积极/中性/消极）：
    {text}
    
    要求：
    按照JSON格式返回，注意仅返回JSON，内容为：
    {{
        "emotion" : 用户情感状态，从[POSITIVE, NEUTRAL, NEGATIVE]中选择,
        "confidence" : 对情感字段的置信度，一个0-1之间的数
    }}
""")

# 经过JsonOutputParser后会返回一个字典
analyze_emotion_chain = analysis_emotion_prompt | llm | JsonOutputParser()

problem_classify_prompt = ChatPromptTemplate.from_template("""
    根据以下内容，分析里面的内容属于什么分类:
    分类选项：
    - 物流问题：配送延迟、物流损坏等
    - 产品质量：商品瑕疵、功能故障等
    - 客户服务：客服态度、响应速度等
    - 支付问题：扣款异常、退款延迟等
    - 退货退款：退货流程、退款金额等
    - 其他：无法归类的反馈
    
    内容为：
    {text}
    
    要求：
    按照JSON格式返回，注意仅返回JSON，内容为：
    {{
        "class" : 分类选项
    }}
""")

problem_classify_chain = problem_classify_prompt | llm | JsonOutputParser()

final_response_prompt = ChatPromptTemplate.from_template("""
    你是一个电商客服专家，接下来需要你根据用户的反馈进行回复。
    
    用户反馈:
    {text}
    
    初步分析：
    内容属于{feedback_class}分类
    用户情感为: {emotion}, 置信度为:{confidence}
    
    要求：
    根据用户的反馈进行回复，在50-150字之间
    若用户处于负面情绪，请安抚并给出解决方案
    若用户为积极反馈，请对用户的支持表示感谢
""")

final_response_chain = RunnablePassthrough.assign(
    emotion_analysis = analyze_emotion_chain,
    question_class = problem_classify_chain
) | RunnableLambda(
    lambda x: {
        "text": x["text"],
        "emotion": x["emotion_analysis"]["emotion"],
        "confidence": x["emotion_analysis"]["confidence"],
        "feedback_class": x["question_class"]["class"],
    }
) | final_response_prompt | llm | StrOutputParser()

result = final_response_chain.stream({"text" : "挺好吃的但是吃完我就拉肚子，我又吃得停不下来"})
for chunk in result:
    print(chunk, end="", flush=True)
