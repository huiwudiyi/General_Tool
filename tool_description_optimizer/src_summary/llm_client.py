import os
import sys
import requests
import time
import json
from tqdm import tqdm
import pandas as pd
from json_repair import repair_json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


class LLMClient:
    """OpenAI-compatible chat-completions client."""
    def __init__(self, base_url, api_key, timeout=60, max_retry=3, host=None, appid=None):
        self.base_url = base_url
        self.api_key = api_key
        self.max_retry = max_retry
        self.sleep_seconds = 3
        self.timeout = timeout
        
        self.headers = {"Content-Type": "application/json", 'X-Tc-Timeout': str(timeout)}
        default_headers = {}
        if api_key:
            if self.api_key.startswith("Bearer "):
                default_headers["Authorization"] = api_key
            else:
                default_headers["Authorization"] = f"Bearer {api_key}"

        if host:
            default_headers["Host"] = host
        if appid:
            default_headers["appid"] = appid

        self.headers.update(default_headers)
            

    def parse_chat_content(self, response: Dict[str, Any]) -> str:
        """Extract assistant text from an OpenAI-compatible response."""
        """
        content = ""
        try:
            content = result["request_data"]["choices"][0]["message"]["content"]
        except:
            pass
        return content
        """
        content = ""
        try:
            content = response["choices"][0]["message"]["content"]
        except:
            pass
        cleanStr = content.replace("```json", "").replace("```", "")
        if len(cleanStr.strip()) == 0:
            return "", "", False
        
        if not isinstance(cleanStr, str):
            return cleanStr, type(cleanStr).__name__, False

        raw_data = None
        is_repaired = False

        # 第一步：原生标准JSON解析
        try:
            raw_data = json.loads(cleanStr)
            if isinstance(raw_data, (dict, list)):
                return raw_data, type(raw_data).__name__, False
        except json.JSONDecodeError:
            pass

        # 第二步：使用repair_json修复损坏JSON后再解析
        try:
            fixed_json_str = repair_json(cleanStr)
            raw_data = json.loads(fixed_json_str)
            if isinstance(raw_data, (dict, list)):
                return raw_data, type(raw_data).__name__, True
        except Exception:
            pass

        # 第三步：无法解析为dict/list，判定为原始字符串
        return cleanStr, "raw_str", False
            
    def generate_text(
            self,
            prompt: Optional[str] = None,
            messages: Optional[List[Dict[str, str]]] = None,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            extra_body: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """使用 requests 直接调用 OpenAI 兼容的 API（备用方法）"""

            if not self.base_url:
                raise RuntimeError("OPENAI_API_BASE is empty. Please set it first.")

            url = self.base_url.rstrip("/")
            if not url.endswith("/chat/completions"):
                url = url + "/chat/completions"


            if messages is None and len(prompt) > 10:
                messages = [{"role": "user", "content": prompt }]

            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.8 if temperature is None else temperature,
            }


            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if extra_body:
                payload.update(extra_body)

            last_error: Optional[Exception] = None
            for attempt in range(1, self.max_retry + 1):
                try:
                    resp = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
                    resp.raise_for_status()
                    return resp.json()
                except Exception as exc:
                    last_error = exc
                    print(f"[Attempt {attempt}/{self.max_retry}] Request failed: {exc}")
                    if attempt < self.max_retry:
                        time.sleep(self.sleep_seconds)

            raise RuntimeError(f"LLM request failed after {self.max_retry} retries: {last_error}")


if __name__ == "__main__":

    # 业务参数
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
        "thought": "详细的理解用户query，并给出合理的解释",
        "intent": "识别用户query的意图",
        "rewrite_query": ["改写query"],
        "number": "用户query的字数要求",
        "time":"具体年月日时间",
        "holiday":"具体的节假日，比如：春节、端午节、清明节、劳动节等"
    }

    #输入query:
    """+ query
    
    llmclient = LLMClient(
        base_url="http://10.11.175.3/tianchi/chat/completions",
        api_key='Bearer 2cff86c63848008ab7982b5c63c9',
        timeout=60,
        host="tianchi-proxy.baidu-int.com",
        max_retry=3,
        appid='app-CdjpA4YQ'
    )

    response = llmclient.generate_text(prompt=prompt,
                    model="deepseek-v3.2",
                    temperature=0.2,
                    max_tokens=1024,
                    extra_body={"top_p": 0.9},
                )
    print("response", response)
    print(llmclient.parse_chat_content(response))