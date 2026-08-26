#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : models.py
@Author  : zhuzerun
@Date    : 2026-08-04 18:10
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""
import os
import json
import datetime
import pandas as pd
from json_repair import repair_json
from dataclasses import dataclass, field


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    submit_date: str
    paper_url: str
    pdf_url: str
    affiliations: list[str] = field(default_factory=list)
    is_big_company: bool = False
    company: list[str] = field(default_factory=list)