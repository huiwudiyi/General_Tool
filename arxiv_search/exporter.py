#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : exporter.py
@Author  : zhuzerun
@Date    : 2026-08-04 18:09
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""
import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class Exporter:
    def to_dict(self, query: str, papers: list[Any]) -> dict:
        rows = [asdict(paper) if is_dataclass(paper) else paper for paper in papers]
        return {"query": query, "papers": rows}

    def to_json(self, query: str, papers: list[Any], output: str | Path | None = None) -> str:
        payload = self.to_dict(query, papers)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if output:
            Path(output).write_text(text, encoding="utf-8")
        return text

    def to_csv(self, papers: list[Any], output: str | Path) -> None:
        rows = [asdict(paper) if is_dataclass(paper) else paper for paper in papers]
        fieldnames = [
            "arxiv_id",
            "title",
            "authors",
            "abstract",
            "categories",
            "submit_date",
            "paper_url",
            "pdf_url",
            "affiliations",
            "is_big_company",
            "company",
        ]
        with Path(output).open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                serialized = row.copy()
                for key in ("authors", "categories", "affiliations", "company"):
                    serialized[key] = "; ".join(serialized.get(key, []))
                writer.writerow({key: serialized.get(key, "") for key in fieldnames})