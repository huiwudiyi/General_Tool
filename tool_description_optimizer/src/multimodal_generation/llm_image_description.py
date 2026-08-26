from __future__ import annotations

import json
import argparse
import base64
import json
import mimetypes
import os
import sys
from typing import Any, Dict, Tuple
from common.utils.utils import *

class LLMImageDescription:
    """基于阿拉丁组件截图生成界面布局描述与功能服务总结（prompts.json 的 image_description）。"""
    @staticmethod
    def _gen_prompt(prompt: str, image_path: str = "") -> str:
        mime_type = guess_mime_type(image_path)
        base64_image = encode_image_to_base64(image_path)
        data_url = "data:{};base64,{}".format(mime_type, base64_image)

        # headers = {
        #     "Content-Type": "application/json",
        #     "Authorization": f"Bearer {API_KEY}",
        #     "Host": API_HOST,
        # }
        # payload = json.dumps({
        #     "model": model,
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
        #                 {"type": "image_url", "image_url": {"url": data_url}},
        #             ],
        #         },
        #     ],
        #     "stream": True,
        # })
        return prompt, data_url

    @staticmethod
    def _vertify_result(response: str):
        # check response result
        if len(set(response.keys()) - set(['layout_description', 'function_summary'])) > 0:
            return {}, False, "key error"
        # check 两部分内容
        layout_description = response["layout_description"]
        function_summary = response["function_summary"]

        return response


if __name__ == "__main__":
    # 自检：合法结果通过，缺 key / 过短 / 超长均被拦截
    good = {"layout_description": "顶部为搜索输入框，中部以卡片式展示核心信息，底部有操作栏和扩展属性区。" * 8,
            "function_summary": "支持拼音查询、释义、近反义词、典故溯源等核心查询能力。" * 8}
    res, flag, err = LLMImageDescription._vertify_result(good)
    assert flag and err == "" and set(res.keys()) == {'layout_description', 'function_summary'}

    for bad in [{"layout_description": "短", "function_summary": "短"},
                {"summary": "缺字段"}, good | {"extra": "x"}]:
        _, flag, _ = LLMImageDescription._vertify_result(bad)
        assert not flag
    print("ok")
