import requests
import time
import json
from tqdm import *
import pandas as pd
import os
import sys
from requests.exceptions import Timeout
from safety import normalize_json_output
from openai import OpenAI

class LLMClient:
    """
    LLM 统一调用封装。

    真实项目中，你可以在这里接入：
    - OpenAI
    - Claude
    - DeepSeek
    - Qwen
    - 本地模型
    """
    # 初始化客户端
    client = OpenAI(
        base_url="https://www.autodl.art/api/v1",
        api_key="AP8t1uYJ1JbF7YFGbb88beEZ8Wbpkc45FL2fHceGNtLt3NdU",
    )
    def fetch_response(self, query_prompt=[]):
        query, doc, prompt = query_prompt
        # 调用接口（这里使用了stream=True进行流式响应）
        try:
            response = client.chat.completions.create(
                model="DeepSeek-R1-0528",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            content = response.json()
        except:
            pass
        nums = 0 
        while (len(content) == 0 or 'error' in response.json()) and nums <= 3:
            time.sleep(3)
            nums += 1
            try:
                response = self.client.chat.completions.create(
                        model="DeepSeek-R1-0528",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                content = response.json()
            except:
                content = ""
        print("llm_zzc", content)
        output_json = {
            "query": query,
            "doc":doc,
            "request_data": response
        }
        return output_json

    def generate_text(self, query: str, doc: str, prompt: str) -> str:
        """
        文本生成接口。
        """
        rqp = [query, doc, prompt]

        result = self.fetch_response(rqp)
        content = ""
        try:
            content = response.choices[0].message.content
        except:
            pass
        return content


if __name__ == "__main__":
    query = "春节祝福语"
    doc = "短文"
    prompt = """# 角色
你是一个中文内容理解与标注专家。对输入文本进行意图分类、关键信息抽取、检索改写查询生成。

# 任务
1、意图识别： 识别用户的核心创作意图，从以下类别中选择最匹配的一项：
    网名：用户寻求创建或推荐昵称、用户名、ID等；
    签名：寻求个人简介、个性签名、状态语录等；
    对联：寻求上下联对仗工整的对联或者春联；
    祝福语：寻求节日祝福、生日祝福、贺词等祝福意图；
    短句：寻求简短的句子、金句、语录、哲理语句等；
    文案：寻求社交媒体帖子、广告语、宣传文案、朋友圈内容等长文本的内容；
    猜谜：寻找猜谜数据；
    造句：用户造句意图；
    笑话：创作幽默、搞笑的内容；
2、query改写：从用户输入中提取最核心的关键词/主题词，基于每个核心词生成完整语义的一个检索查询语句，避免生成重复或相似度过高的查询
3、字数要求：判断用户输入中是否包含明确的字数限制要求，解析出具体的字符长度范围或上下限，如果无，则不进行标注
    number：具体字数要求，如8字、10字、20字；
    op:描述number的关系，gt：大于； lt：小于；gte：不小于； lte：不超过； range：字数之间； eq：等于
4、时间/节假日要求：识别用户输入中与时间或节假日相关的需求，
    type：类型，取值为 "time"（时间年月日）或 "holiday"（节假日）
    value: 具体内容
# 输出格式
{
    "thought": "详细的理解用户query，并给出合理的解释"
    "intent": "识别用户query的意图",
    "rewrite_query": ["改写query"],
    "number": "用户query的字数要求",
    "time":"具体年月日时间"
    "holiday":"具体的节假日，比如：春节、端午节、清明节、劳动节等"
}

#输入query:
"""+ query
    llm_client = LLMClient()
    result = llm_client.generate_text(query, doc, prompt)
    print("result", result)
    print(normalize_json_output(result))