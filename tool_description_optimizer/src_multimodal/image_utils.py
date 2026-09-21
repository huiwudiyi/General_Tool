#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : image_utils.py
@Desc    : 截图加载与多模态消息拼装

截图在 Excel 里可能以三种形态出现，这里统一收敛成 OpenAI 兼容的 image_url：
1. http(s) 链接      —— 原样透传，由模型服务自己去取
2. 本地文件路径      —— 读成 base64 data URL（服务端通常访问不到我们的本地盘）
3. 已经是 data: URL  —— 原样透传
另外支持 xlsx 内嵌图片：openpyxl 能读出图片及其锚定行，据此和数据行对齐。
"""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

# 常见截图格式，扩展名识别不出来时按 png 兜底
DEFAULT_MIME = "image/png"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


def guess_mime(path: str) -> str:
    """根据扩展名猜 MIME，猜不出来按 png 处理。"""
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("image/"):
        return mime
    return DEFAULT_MIME


def to_image_url(source: Any, max_bytes: int = 8 * 1024 * 1024) -> Optional[str]:
    """把一个截图来源转成可直接放进 image_url 的字符串。

    Args:
        source: http(s) 链接 / 本地路径 / data URL。
        max_bytes: 单图体积上限，超过则跳过——多模态请求体过大容易被网关拒掉，
                   与其发出去拿 413/400，不如这里先拦住并告警。

    Returns:
        可用的 url 字符串；无法处理时返回 None（调用方跳过该图）。
    """
    if source is None:
        return None
    text = str(source).strip()
    if not text:
        return None

    if text.startswith("data:"):
        return text
    if text.startswith("http://") or text.startswith("https://"):
        return text

    if not os.path.isfile(text):
        print(f"[image] 截图文件不存在，跳过: {text}")
        return None
    try:
        size = os.path.getsize(text)
        if size > max_bytes:
            print(f"[image] 截图过大({size}B > {max_bytes}B)，跳过: {text}")
            return None
        with open(text, "rb") as reader:
            payload = base64.b64encode(reader.read()).decode("ascii")
        return "data:%s;base64,%s" % (guess_mime(text), payload)
    except Exception as exc:
        print(f"[image] 截图读取失败，跳过 {text}: {exc}")
        return None


def build_multimodal_messages(
    prompt: str,
    screenshots: Optional[Sequence[Any]] = None,
    max_images: int = 4,
    detail: str = "high",
) -> List[Dict[str, Any]]:
    """拼装 OpenAI 兼容的多模态 messages。

    Args:
        prompt: 文本部分。
        screenshots: 截图来源列表。
        max_images: 单次请求最多带几张图；截图多时只取前 N 张，
                    控制 token 消耗，也避免请求体过大。
        detail: 图片精细度，布局分析需要看清 UI 元素，默认 high。

    Returns:
        messages 列表。没有任何可用截图时退化为纯文本消息，
        由上层决定是否仍要继续（纯文本也能生成描述，只是缺视觉证据）。
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    used = 0
    for source in list(screenshots or []):
        if used >= max_images:
            break
        url = to_image_url(source)
        if not url:
            continue
        content.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
        used += 1

    if used == 0:
        return [{"role": "user", "content": prompt}]
    return [{"role": "user", "content": content}]


def count_usable(screenshots: Optional[Sequence[Any]]) -> int:
    """统计有多少张截图真的可用，用于日志与降级判断。"""
    return sum(1 for item in (screenshots or []) if to_image_url(item))
