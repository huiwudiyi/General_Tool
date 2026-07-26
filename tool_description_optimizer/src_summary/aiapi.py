import requests
import json
import ssl
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        # 设置支持的TLS版本
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

def request_aiapi(query):
    url = "https://m.baidu.com/search/api/v2"
    
    params = {
        "word": query,
        "qf": "monster",
    }
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Linux; Android 13; Mi 10 Build/TKQ1.221114.001; wv) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 "
            "Chrome/97.0.4692.98 Mobile Safari/537.36 "
            "T7/13.58 SP-engine/2.97.0 baiduboxapp/13.69.0.10 "
            "(Baidu; P1 13) NABar/1.0"
        ),
        "api-key": "100000",
    }
    
    data = {
        "size": {
            "main": 10,
            "video": 0,
            "aladdin": 10,
        }
    }
    
    # 创建带SSL适配器的session
    session = requests.Session()
    session.mount('https://', SSLAdapter())
    
    try:
        response = session.post(
            url,
            params=params,
            headers=headers,
            json=data,
            timeout=30
        )
        if response.status_code == 200:
            return response.text
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text[:200]}")
            return None
    except requests.exceptions.SSLError as e:
        print(f"SSL错误: {e}")
        # 如果SSL失败，尝试使用verify=False
        try:
            print("尝试使用verify=False重试...")
            response = requests.post(
                url,
                params=params,
                headers=headers,
                json=data,
                verify=False,
                timeout=30
            )
            if response.status_code == 200:
                return response.text
        except Exception as retry_e:
            print(f"重试也失败: {retry_e}")
            return None
    except Exception as e:
        print(f"请求异常: {e}")
        return None

def parse_aiapi(response):
    if not response:
        print("响应为空")
        return {}, []
    
    aladdin_aiapi_dict = {}
    natural_result = []
    
    try:
        # 解析JSON
        data = json.loads(response)
        
        # 检查是否存在results和result
        if "results" in data and "result" in data["results"]:
            for aladdin_info in data["results"]["result"]:
                if aladdin_info["type"] == 1 or aladdin_info["type"] == 2:
                    title = aladdin_info.get("title", "")
                    snippet = aladdin_info.get("snippet", "")
                    sentence = aladdin_info.get("sentence", "")
                    natural_result.append({
                        "title": title, 
                        "snippet": snippet, 
                        "sentence": sentence
                    })
                elif aladdin_info["type"] == 4:
                    if "aladdin_info" in aladdin_info:
                        srcid = aladdin_info["aladdin_info"].get("srcid", "")
                        if "structure_data" in aladdin_info["aladdin_info"]:
                            label_data = aladdin_info["aladdin_info"]["structure_data"].get("label_data", {})
                            aladdin_aiapi_dict[srcid] = label_data
                elif aladdin_info["type"] == 3:
                    pass
        else:
            print("响应中没有找到results数据")
            print(f"响应结构: {list(data.keys())}")
            
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"原始响应前200字符: {response[:200]}")
    except KeyError as e:
        print(f"键错误: {e}")
        print(f"响应结构可能不完整")
    except Exception as e:
        print(f"解析过程中出现未知错误: {e}")
    
    return aladdin_aiapi_dict, natural_result

if __name__ == "__main__":
    print("开始请求...")
    response = request_aiapi("旅游")
    
    if response:
        print(f"响应长度: {len(response)} 字符")
        print(f"响应前200字符: {response[:200]}")
        
        aladdin_aiapi_dict, natural_result = parse_aiapi(response)
        
        print(f"\n解析结果:")
        print(f"aladdin_aiapi_dict 条目数: {len(aladdin_aiapi_dict)}")
        print(f"natural_result 条目数: {len(natural_result)}")
        
        if aladdin_aiapi_dict:
            print(f"aladdin_aiapi_dict 示例: {list(aladdin_aiapi_dict.keys())[:3]}")
        if natural_result:
            print(f"natural_result 示例: {natural_result[:2]}")
    else:
        print("请求失败，无法解析")


# import requests

# def request_aiapi(query):
#     url = "https://m.baidu.com/search/api/v2"

#     params = {
#         "word": query,
#         "qf": "monster",
#     }

#     headers = {
#         "Content-Type": "application/json",
#         "User-Agent": (
#             "Mozilla/5.0 (Linux; Android 13; Mi 10 Build/TKQ1.221114.001; wv) "
#             "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 "
#             "Chrome/97.0.4692.98 Mobile Safari/537.36 "
#             "T7/13.58 SP-engine/2.97.0 baiduboxapp/13.69.0.10 "
#             "(Baidu; P1 13) NABar/1.0"
#         ),
#         "api-key": "100000",
#     }

#     data = {
#         "size": {
#             "main": 10,
#             "video": 0,
#             "aladdin": 10,
#         }
#     }

#     response = requests.post(
#         url,
#         params=params,
#         headers=headers,
#         json=data,
#         verify=False
#     )
#     if response.status_code == 200:
#         return response.text

# def parse_aiapi(response):
#     aladdin_aiapi_dict = {}
#     natural_result = []
#     try:
#         for aladdin_info in json.loads(response)["results"]["result"]:
#             if aladdin_info["type"] == 1 or aladdin_info["type"] == 2:
#                 title = aladdin_info["title"]
#                 snippet = aladdin_info["snippet"]
#                 sentence = aladdin_info["sentence"]
#                 natural_result.append({"title": title, "snippet": snippet, "sentence": sentence})
#             elif aladdin_info["type"] == 4:
#                 srcid = aladdin_info["aladdin_info"]["srcid"]
#                 label_data = aladdin_info["aladdin_info"]["structure_data"]["label_data"]
#                 aladdin_aiapi_dict[srcid] = label_data
#             elif aladdin_info["type"] == 3:
#                 pass
#     except:
#         pass
#     return aladdin_aiapi_dict, natural_result
# if __name__ == "__main__":
#     response = request_aiapi("旅游")
#     aladdin_aiapi_dict, natural_result = parse_aiapi(response)