#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : test_baise.py.py
@Author  : zhuzerun
@Date    : 2026-08-04 14:55
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""
import os
import json
import datetime
import pandas as pd
from json_repair import repair_json
import json
from time import sleep
from unittest import registerResult
import requests
from datetime import datetime
import sys
import pandas as pd


# from bns import get_random_ipport

def request_aurora(query):
    ip = '10.141.194.106:2396'

    product_name = "AgentsEcologyAgentsEcologyNew"
    handler_name = "rank_mcp_main"
    identifier = datetime.now().strftime('%Y%m%d%H%M%S')
    req_params = {
        "product_name": product_name,
        "service_name": "VSRank",
        "handler_name": handler_name,
        "identifier": identifier,
        "version": "default",
        "exp_id_list": ["123456"],
        "param_list": [
            """params={\"page_size\" : 5}""",
            """is_debug=0""",
            """threshold=0.01""",
            """channel_control=ala_baikan-all"""
        ],
        "query_request_info": {
            "query": query
            # "query" : f"Instruct: 根据用户的查询意图，检索最相关的工具或服务\nQuery:{query}"
        }
    }
    response = requests.post(f'http://{ip}/VsRankService/Search', data=json.dumps(req_params))
    return response.text


def main():
    """主函数"""
    print(request_aurora("施舍的含义"))

    pass


if __name__ == "__main__":
    main()
