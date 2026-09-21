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
#                     resp = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
                    resp = requests.request("POST", url, headers=self.headers, data=json.dumps(payload), timeout=self.timeout)
                    resp.raise_for_status()
                    return resp.json()
                except Exception as exc:
                    last_error = exc
                    print(f"[Attempt {attempt}/{self.max_retry}] Request failed: {exc}")
                    if attempt < self.max_retry:
                        time.sleep(self.sleep_seconds)

            raise RuntimeError(f"LLM request failed after {self.max_retry} retries: {last_error}")

llmclient = LLMClient(
        base_url="http://10.11.175.3/tianchi/chat/completions",
        api_key='Bearer 13af0a5f5048000a72509152a644',
        timeout=60,
        host="tianchi-proxy.baidu-int.com",
        max_retry=3,
        appid='app-RNgOjXzL'
    )

if __name__ == "__main__":

    # 业务参数
    query = "春节祝福语"
    doc = "短文"
    prompt = """# Role
你是工具描述生成器，负责为工具重写一段同时服务于「向量召回」与「LLM 选择」的结构化描述。输出必须包含「功能区」与「边界区」两部分。

# Task
- 功能区：说明工具能解决什么问题、覆盖哪些查询意图、返回什么内容。要覆盖[正例查询]体现的场景，语义聚焦、无冗余修饰。
- 边界区：明确列出不适用的场景，重点把[负例查询]这类"相似但不该命中"的需求排除掉。要写成可判别的条件，而不是笼统的否定。
- 两个区内容不得重复；不得编造工具不具备的能力，只能表述[当前描述]中真实存在的能力。

# 执行过程
1. 读[当前描述]，确认工具真实具备的能力与其自然边界；
2. 归纳[正例查询]的共性意图，提炼进功能区；
3. 对比[负例查询]与正例的差别，找出最容易混淆的区分维度，写进边界区；
4. 压缩表达，去掉口语化与营销话术，保证向量检索时语义集中。

# 工具标题
wise重要url保护人工干预mis

# 当前描述
工具描述：该工具组件用于展示 Instagram 平台账号注册与登录入口及产品功能介绍
工具返回内容：Instagram 账号注册 / 登录指引、平台核心功能介绍
插件满足样式：
（1）官方 Instagram 标识
（2）账号注册、登录相关提示文案
（3）平台核心功能说明：拍摄、编辑、分享照片、视频以及和亲友收发消息，主打简单有趣的创意社交
插件交互样式：
（1）可进入 Instagram 账号注册或登录流程
（2）可跳转进入 Instagram 官方平台使用图片视频社交功能

# 正例查询（应命中本工具）
专利业务办理系统、德高防水、东本贴吧、拼多多拼多多、国家药品监督管理局、中国驻英国大使馆、吉林省前卫医院、mc.、yige、外婆味道、股市通、claude、原麦山丘、小狐狸钱包、武炼顶峰、chatgpt网址、金域医学、www.24、百度百家账号注册入口、土流网、携程租车、百度文心助手、诡秘之主第二季、文心一言4.5、格林豪泰酒店、id3、华润集团、四川银行招聘官网、中国旅游文化资源开发促进会、广东工商学院、蛋仔 派对、宿命之环、tensorflow、robotaxi、大连外国语大学、马来西亚电子入境卡、郑州交通技师学院、两江新区人民政府、元宝官网、中国联通、即梦ai官网、华东交通大学、极越汽车、山中大学、欧易下载、极客湾、神秘复苏。、度加、山西工程技术学院、芝商所、.浙江体彩网、广州慈惠医院、中国保密在线、hugging face、虎牙直播、民法典、抖音网页端、山东省教育招生考试院、爱奇艺国际版、上海交大巴黎卓越工程师学院、大众途观价格、国家电网、有戏ai官网、婚姻法、福建事业编2026岗位表、下载文小言官方版、tp钱包官网下载、youtibe、ems、隐秘死角、学习强国、深圳阳光采购平台、中国金币网、百度贴吧。、江西管理职业学院、苟在妖武乱世修仙、湘西党建网、学信网个人学历查询、北京车展、威然、metamask、facebook、欧交易所官网、广州工商学院、文小言、菜鸟app、bibili、中共中央社会工作部、锦州医科大学医疗学院、御兽之王、好看 视频、deepseek苹果手机版、甘肃省教育厅基础教育课程教材中心、网页版抖音、赫兹租车、deepseek官方免费下载最新版、网飞官网、辽宁警察学院、twitter、火山云、中囯竟彩网、国南电网、biligle、上海中信大厦、中建深圳装饰工程有限公司、蛋仔paida、爱艺奇国际版官网、重亲大学、抖音有网页版、pindudu、抖音网络版、通易万象、两安建筑科技大学、北京养老服务网、tiwitter、顾家开发银行、玄间仙族、燕景理工学院、北京理工倪赫、阿坝州疾病预防控制中心、国家政务平台是什么、四大学川、bilipul、南京大屠杀是真的吗?、ehviewet、黑龙江农业职业技术院。、北京智慧中小学教育平台、‌deepseek下载、拷貝漫書下载、囧次元 下载、观陵山墓园、颐和园浩雷、tesala、西视频、"twitter、南陵工业大学、chtgpt、湖北文化和旅游厅、ajent、nertflix、殡葬网站、中国连通、biljibabili、文心一言如何下载、纽约时报官方网站、上诲中心大厦、chaegtp、metamast官网、大众烕然、只乎、西瓜视、佰度、遣忘之海、阳光融合医院官网、豆bao、社会工作部工作、instergam、metamask官网、全国服务政务平台、bililbil、山东省教育考试、西压视频、悦纹、给神话悟空、西礻瓜视频、维叶纳酒店、twwiter、qq 影音、langyi、liliilup、chat gpe、大众途昻、文心一言官方app、tieba、沈阳 大学、阿里星校招、chatglt、百度希壤、翁牛特旗法院、chagpt、ghatgpt、tokenp0cket、计能人才评价工作网、- facebook、中国在线保密、营销科学学报、国家政务服务平台\、#中山大学、石问子大学、重生我要冲浪、大众vlioran、kimi!、国家教育读书平台、颐和园,、哔哩哔哩唧唧、文心叻手、美团外卖订餐平、香港科技大学。、北京教育统一认证平台、bilibl

# 负例查询（不应命中本工具）
手机号码查寻、查询电话号码、查电话号码归属地。、的的客服24小时热线、手机号查件、手机号码查泃、查电话号码、24小时人工服务电话、电信手机号码查询、查电活号码、付机号码、骚扰电活、电话号码归属地查询。、联系客服热线、查号平台、电话所在地、企业电话号码查询大全、号码规属地查询、电话查询。、手机号归属地号码查询

# 输出格式
```json
{
"功能区": "工具能力、覆盖的查询意图、返回内容",
"边界区": "不适用的场景，以及需要排除的相似需求及其区分依据"
}
```
输出:
"""
    
    
    llmclient = LLMClient(
        base_url="http://10.11.175.3/tianchi/chat/completions",
        api_key='Bearer 13fe002f2048000a464097e9d048',
        timeout=60,
        host="tianchi-proxy.baidu-int.com",
        max_retry=3,
        appid='app-Kk6z2xYE'
    )

    response = llmclient.generate_text(prompt=prompt,
                    model="deepseek-v4-flash",
                    temperature=0.2,
                    max_tokens=1024,
                    extra_body={"top_p": 0.9},
                )
    print("response", response)
    print(llmclient.parse_chat_content(response))