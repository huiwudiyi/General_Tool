#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : result_store.py
@Desc    : 结果落库（sqlite）

为什么用 sqlite 而不是直接写 csv/json：这条流水线是按 srcid 批量跑的，
需要支持「跑到一半停掉、下次只跑没跑完的」以及「按 srcid 查历史轮次」，
用一张表 + 唯一键就能同时满足，避免每次重跑都全量覆盖产物文件。

与 langgraph 的 checkpoint 库是两回事：那个存流程执行状态用于续跑，
这个存业务产物用于查询与导出。
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS multimodal_description (
    srcid            TEXT PRIMARY KEY,
    queries          TEXT,
    label_data       TEXT,
    screenshots      TEXT,
    description      TEXT,
    layout_text      TEXT,
    function_text    TEXT,
    judge_score      INTEGER,
    judge_passed     INTEGER,
    evidence         TEXT,
    suggestions      TEXT,
    rounds           INTEGER,
    accepted         INTEGER,
    updated_at       TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS multimodal_stage_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    srcid            TEXT,
    round_id         INTEGER,
    stage            TEXT,
    info             TEXT,
    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_stage_srcid ON multimodal_stage_log(srcid, round_id);
"""


class ResultStore:
    """产物存储，按 srcid 幂等 upsert。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @staticmethod
    def _dump(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def save_result(self, state: Mapping[str, Any]) -> None:
        """写入/覆盖某个 srcid 的最终产物。"""
        row = (
            str(state.get("srcid", "")),
            self._dump(state.get("queries", [])),
            self._dump(state.get("label_data", "")),
            self._dump(state.get("screenshots", [])),
            self._dump(state.get("final_description") or state.get("description", "")),
            self._dump(state.get("layout_text", "")),
            self._dump(state.get("function_text", "")),
            state.get("judge_score", None),
            1 if state.get("judge_passed") else 0,
            self._dump(state.get("evidence", [])),
            self._dump(state.get("suggestions", [])),
            int(state.get("round_id", 0) or 0),
            1 if state.get("accepted") else 0,
        )
        self.conn.execute(
            """
            INSERT INTO multimodal_description
                (srcid, queries, label_data, screenshots, description, layout_text,
                 function_text, judge_score, judge_passed, evidence, suggestions,
                 rounds, accepted, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, CURRENT_TIMESTAMP)
            ON CONFLICT(srcid) DO UPDATE SET
                queries=excluded.queries, label_data=excluded.label_data,
                screenshots=excluded.screenshots, description=excluded.description,
                layout_text=excluded.layout_text, function_text=excluded.function_text,
                judge_score=excluded.judge_score, judge_passed=excluded.judge_passed,
                evidence=excluded.evidence, suggestions=excluded.suggestions,
                rounds=excluded.rounds, accepted=excluded.accepted,
                updated_at=CURRENT_TIMESTAMP
            """,
            row,
        )
        self.conn.commit()

    def log_stage(self, srcid: str, round_id: int, stage: str, info: Any) -> None:
        """记录一次阶段输出，用于事后追溯每轮改了什么。"""
        self.conn.execute(
            "INSERT INTO multimodal_stage_log (srcid, round_id, stage, info) VALUES (?,?,?,?)",
            (str(srcid), int(round_id), str(stage), self._dump(info)),
        )
        self.conn.commit()

    def finished_srcids(self, require_accepted: bool = False) -> List[str]:
        """已产出结果的 srcid，用于批量跑时跳过。"""
        sql = "SELECT srcid FROM multimodal_description WHERE description != ''"
        if require_accepted:
            sql += " AND accepted = 1"
        return [row[0] for row in self.conn.execute(sql).fetchall()]

    def export_rows(self) -> List[Dict[str, Any]]:
        """导出全部产物，便于转 Excel/CSV 交付。"""
        cursor = self.conn.execute("SELECT * FROM multimodal_description ORDER BY srcid")
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
