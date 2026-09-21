#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : excel_loader.py
@Desc    : 读入 Excel 输入数据并按 srcid 聚合

Excel 列（列名做了别名兼容，大小写与空格不敏感）：
- query      用户输入的 query
- srcid      工具 id
- label_data 该 query 调用多模态后返回的文本数据
- 截图        该 query 调用多模态后返回的截图（路径 / 链接，或 xlsx 内嵌图片）

一个 srcid 通常对应多条 query，这里按 srcid 聚合：文本拼成带 query 标注的段落，
截图去重后汇总——描述 agent 面对的是"这个工具整体长什么样"，而不是单条 query。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from image_utils import IMAGE_EXTS

COLUMN_ALIASES = {
    "query": ("query", "用户query", "用户问题", "问题"),
    "srcid": ("srcid", "src_id", "工具id", "资源号", "id"),
    "label_data": ("label_data", "labeldata", "文本数据", "召回数据", "返回数据"),
    "screenshot": ("截图", "截图数据", "screenshot", "screenshots", "图片", "image", "图片路径"),
}


@dataclass
class MultiModalSample:
    """一个 srcid 聚合后的输入。"""

    srcid: str
    queries: List[str] = field(default_factory=list)
    label_items: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)

    @property
    def label_data(self) -> str:
        """拼成带 query 标注的文本，便于模型区分不同 query 的召回内容。"""
        return "\n\n".join(self.label_items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "srcid": self.srcid,
            "queries": list(self.queries),
            "label_data": self.label_data,
            "screenshots": list(self.screenshots),
        }


def _norm(name: Any) -> str:
    return str(name or "").strip().lower().replace(" ", "").replace("_", "")


def resolve_columns(columns: Sequence[Any]) -> Dict[str, Optional[str]]:
    """把实际列名映射到标准字段名，找不到的返回 None 由上层报错。"""
    normalized = {_norm(col): col for col in columns}
    resolved: Dict[str, Optional[str]] = {}
    for field_name, aliases in COLUMN_ALIASES.items():
        hit = None
        for alias in aliases:
            if _norm(alias) in normalized:
                hit = normalized[_norm(alias)]
                break
        resolved[field_name] = hit
    return resolved


def extract_embedded_images(excel_path: str, sheet_name: Optional[str], out_dir: str) -> Dict[int, List[str]]:
    """导出 xlsx 内嵌图片，并按锚定行号归位。

    截图直接贴在单元格里时，pandas 读不到，只能用 openpyxl 取 ws._images。
    图片锚点的 _from.row 是 0-based，这里统一转成 1-based 的 Excel 行号，
    方便和 pandas 的数据行对齐（pandas 第 0 行数据 = Excel 第 2 行）。
    """
    images: Dict[int, List[str]] = {}
    try:
        from openpyxl import load_workbook
    except ImportError:
        print("[excel] 未安装 openpyxl，跳过内嵌图片提取")
        return images

    try:
        workbook = load_workbook(excel_path)
        worksheet = workbook[sheet_name] if sheet_name else workbook.worksheets[0]
    except Exception as exc:
        print(f"[excel] 打开工作簿失败，跳过内嵌图片提取: {exc}")
        return images

    embedded = list(getattr(worksheet, "_images", []) or [])
    if not embedded:
        return images

    os.makedirs(out_dir, exist_ok=True)
    for index, image in enumerate(embedded):
        try:
            anchor = getattr(image, "anchor", None)
            row = getattr(getattr(anchor, "_from", None), "row", None)
            if row is None:
                continue
            excel_row = int(row) + 1
            data = image._data() if callable(getattr(image, "_data", None)) else None
            if not data:
                continue
            ext = str(getattr(image, "format", "") or "png").lower()
            if not ext.startswith("."):
                ext = "." + ext
            if ext not in IMAGE_EXTS:
                ext = ".png"
            path = os.path.join(out_dir, f"row{excel_row}_{index}{ext}")
            with open(path, "wb") as writer:
                writer.write(data)
            images.setdefault(excel_row, []).append(path)
        except Exception as exc:
            print(f"[excel] 第 {index} 张内嵌图片导出失败: {exc}")

    print(f"[excel] 导出内嵌图片 {sum(len(v) for v in images.values())} 张 -> {out_dir}")
    return images


def _split_screenshot_cell(value: Any) -> List[str]:
    """单元格里可能写了多张图（换行 / 逗号 / 分号分隔），拆开。"""
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return []
    for sep in ("\n", ";", "；", ","):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def load_samples(
    excel_path: str,
    sheet_name: Optional[str] = None,
    image_dir: Optional[str] = None,
) -> Dict[str, MultiModalSample]:
    """读 Excel 并按 srcid 聚合。

    Returns:
        {srcid: MultiModalSample}
    """
    frame = pd.read_excel(excel_path, sheet_name=sheet_name or 0)
    if isinstance(frame, dict):  # sheet_name=None 时 pandas 返回 dict
        frame = list(frame.values())[0]

    columns = resolve_columns(frame.columns)
    missing = [name for name in ("srcid", "label_data") if not columns.get(name)]
    if missing:
        raise ValueError(f"Excel 缺少必要列 {missing}，实际列={list(frame.columns)}")

    embedded = extract_embedded_images(
        excel_path, sheet_name, image_dir or os.path.join(os.path.dirname(excel_path) or ".", "embedded_images")
    )

    samples: Dict[str, MultiModalSample] = {}
    for position, (_, row) in enumerate(frame.iterrows()):
        srcid = str(row.get(columns["srcid"], "") or "").strip()
        if not srcid or srcid.lower() in ("nan", "none"):
            continue
        sample = samples.setdefault(srcid, MultiModalSample(srcid=srcid))

        query = ""
        if columns.get("query"):
            query = str(row.get(columns["query"], "") or "").strip()
        if query and query.lower() != "nan" and query not in sample.queries:
            sample.queries.append(query)

        label = str(row.get(columns["label_data"], "") or "").strip()
        if label and label.lower() != "nan":
            sample.label_items.append(f"[query] {query or '-'}\n[召回数据] {label}")

        shots: List[str] = []
        if columns.get("screenshot"):
            shots.extend(_split_screenshot_cell(row.get(columns["screenshot"])))
        # pandas 第 position 行 == Excel 第 position+2 行（第 1 行是表头）
        shots.extend(embedded.get(position + 2, []))
        for shot in shots:
            if shot not in sample.screenshots:
                sample.screenshots.append(shot)

    print(f"[excel] 载入 {len(frame)} 行，聚合成 {len(samples)} 个 srcid")
    return samples
