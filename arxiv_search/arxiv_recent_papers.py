#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : arxiv_recent_papers.py
@Author  : zhuzerun
@Date    : 2026-08-04 18:08
@Version : 1.0
@Desc    : 
@Contact : zachary6chu@gmail.com
"""
import argparse
from pathlib import Path

from arxiv_searcher import ArxivSearcher
from exporter import Exporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Search recent arXiv papers and label major-company affiliations.")
    parser.add_argument("query", help="Search query, for example: 'Query Rewriting'")
    parser.add_argument("--days", type=int, default=90, help="Only keep papers submitted in the last N days.")
    parser.add_argument("--max-pages", type=int, default=3,help="Maximum arXiv result pages to parse; omit for automatic pagination.")
    parser.add_argument("--page-size", type=int, default=25, choices=(25, 50, 100, 200), help="arXiv page size.")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent workers for detail/PDF enrichment.")
    parser.add_argument("--cache-dir", default=".arxiv_pdf_cache", help="PDF cache directory.")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF download/text extraction.")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR fallback when PDF text extraction is empty.")
    parser.add_argument("--json-output", help="Optional JSON output path.")
    parser.add_argument("--csv-output", help="Optional CSV output path.")
    args = parser.parse_args()
    print("*" * 60)
    searcher = ArxivSearcher(
        page_size=args.page_size,
        max_workers=args.workers,
        cache_dir=args.cache_dir,
    )
    print("*" * 60 )
    papers = searcher.search(
        args.query,
        days=args.days,
        max_pages=args.max_pages,
        use_pdf=not args.no_pdf,
        use_ocr=not args.no_ocr,
    )

    exporter = Exporter()
    json_text = exporter.to_json(args.query, papers, output=args.json_output)
    if args.csv_output:
        exporter.to_csv(papers, Path(args.csv_output))
    print(json_text)


if __name__ == "__main__":
    main()