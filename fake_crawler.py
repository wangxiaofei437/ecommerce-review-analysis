# -*- coding: utf-8 -*-
"""
电商评论爬虫 (Comment Crawler)
================================

完整的爬虫框架结构，包含：
    - requests.Session 会话管理
    - User-Agent / Cookie / Referer 请求头池
    - 代理 IP 池 (可选)
    - 请求重试 + 指数退避
    - 请求频率控制 (限速 + 随机抖动)
    - robots.txt 检查
    - HTML/JSON 解析层
    - 增量去重 (基于 comment_id)
    - 断点续爬 (state 文件)
    - 多线程并发抓取
    - 日志系统
    - 数据持久化 (CSV / JSONL)

注意：
    出于演示与合规考虑，真实的 HTTP 抓取层 (`_http_get`) 在最终返回
    数据时会回退到本地数据集 `ecommerce_comments_100k.csv` 中随机抽样，
    模拟服务端返回的 JSON 响应。所有上层流程 (Session/重试/解析/去重/
    存储) 均按真实爬虫实现，可直接替换 `_http_get` 为真实接口调用。

用法：
    python fake_crawler.py --platform 淘宝 --pages 10 --per-page 20 \
        --workers 4 --output crawled_comments.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Iterable, Optional
from urllib.parse import urlencode
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    from requests.packages.urllib3.util.retry import Retry  # type: ignore


# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "ecommerce_comments_100k.csv")
STATE_PATH = os.path.join(BASE_DIR, ".crawler_state.json")
LOG_PATH = os.path.join(BASE_DIR, "crawler.log")

PLATFORM_HOSTS = {
    "淘宝": "rate.taobao.com",
    "天猫": "rate.tmall.com",
    "京东": "club.jd.com",
    "拼多多": "mobile.yangkeduo.com",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Mobile/15E148",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

ACCEPT_LANGUAGES = ["zh-CN,zh;q=0.9,en;q=0.8", "zh-CN,zh;q=0.9", "zh;q=0.8,en;q=0.6"]

PROXY_POOL: list[str] = [
    # "http://user:pass@127.0.0.1:7890",
    # "http://10.0.0.1:8080",
]

CSV_HEADERS = [
    "comment_id", "platform", "product_id", "product_category",
    "user_id", "comment_time", "is_additional", "comment_text",
    "rating", "useful_votes", "reply_status", "sentiment_label",
    "attribute_labels", "problem_label", "fake_risk_label",
]


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("crawler")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = _setup_logger()


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class CommentItem:
    comment_id: str
    platform: str
    product_id: str
    product_category: str
    user_id: str
    comment_time: str
    is_additional: bool
    comment_text: str
    rating: int
    useful_votes: int
    reply_status: bool
    sentiment_label: str
    attribute_labels: str
    problem_label: bool
    fake_risk_label: str

    def fingerprint(self) -> str:
        return hashlib.md5(self.comment_id.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 工具: 请求头 / 代理 / 限速
# ---------------------------------------------------------------------------
def build_headers(referer: Optional[str] = None) -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": random.choice(ACCEPT_LANGUAGES),
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": referer or "https://www.google.com/",
        "Cookie": f"sessionid={random.getrandbits(64):x}; track_id={random.getrandbits(32):x}",
    }


def pick_proxy() -> Optional[dict]:
    if not PROXY_POOL:
        return None
    proxy = random.choice(PROXY_POOL)
    return {"http": proxy, "https": proxy}


class RateLimiter:
    """令牌桶式限速器，控制最小请求间隔 + 随机抖动。"""

    def __init__(self, min_interval: float = 0.3, jitter: float = 0.4):
        self.min_interval = min_interval
        self.jitter = jitter
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            sleep_for = self.min_interval - elapsed + random.uniform(0, self.jitter)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()


# ---------------------------------------------------------------------------
# robots.txt 合规检查
# ---------------------------------------------------------------------------
def check_robots(url: str, user_agent: str = "*") -> bool:
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        # 演示场景：不真正发起请求，默认允许
        log.debug(f"robots.txt 检查: {robots_url} -> 允许")
        return True
    except Exception as e:
        log.warning(f"robots.txt 检查失败: {e}")
        return True


# ---------------------------------------------------------------------------
# Session 工厂 (带重试)
# ---------------------------------------------------------------------------
def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ---------------------------------------------------------------------------
# 数据源 (演示用): 从本地数据集模拟服务端响应
# ---------------------------------------------------------------------------
class DatasetBackend:
    """将本地 CSV 作为模拟的服务端数据源。线程安全。"""

    _instance: Optional["DatasetBackend"] = None
    _lock = threading.Lock()

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据集不存在: {path}")
        log.info(f"加载数据集: {path}")
        self.df = pd.read_csv(path)
        log.info(f"数据集共 {len(self.df)} 条记录")
        self._cursor_lock = threading.Lock()

    @classmethod
    def get(cls, path: str = DATASET_PATH) -> "DatasetBackend":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(path)
            return cls._instance

    def sample(self, platform: str, n: int) -> list[dict]:
        sub = self.df[self.df["platform"] == platform]
        if len(sub) == 0:
            sub = self.df
        n = min(n, len(sub))
        rows = sub.sample(n=n, replace=False).to_dict(orient="records")
        return rows


# ---------------------------------------------------------------------------
# 抓取层
# ---------------------------------------------------------------------------
class CommentCrawler:
    def __init__(
        self,
        platform: str,
        per_page: int = 20,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.platform = platform
        self.per_page = per_page
        self.host = PLATFORM_HOSTS.get(platform, "example.com")
        self.session = build_session()
        self.limiter = rate_limiter or RateLimiter()
        self.backend = DatasetBackend.get()

    # ------------------ HTTP 层 ------------------
    def _build_url(self, page: int, product_id: Optional[str] = None) -> str:
        params = {"page": page, "pageSize": self.per_page, "order": "time_desc"}
        if product_id:
            params["itemId"] = product_id
        return f"https://{self.host}/feedRateList.json?{urlencode(params)}"

    def _http_get(self, url: str) -> dict:
        """模拟一次真实 HTTP GET。

        在演示场景下，我们不真的请求外网，而是：
            1. 走完限速 + 构造请求头 + 选代理 的全部流程；
            2. 打印请求日志；
            3. 从本地数据集中取样作为"服务端 JSON 响应"。
        替换为真实接口时，仅需把 `data` 替换为 `response.json()` 即可。
        """
        self.limiter.wait()
        headers = build_headers(referer=f"https://{self.host}/")
        proxies = pick_proxy()

        log.info(f"GET {url}")
        log.debug(f"  headers.UA={headers['User-Agent'][:50]}...")
        if proxies:
            log.debug(f"  proxy={proxies['https']}")

        # ===== 真实抓取应使用：=====
        # resp = self.session.get(url, headers=headers, proxies=proxies, timeout=10)
        # resp.raise_for_status()
        # return resp.json()
        # ===========================

        # 演示：模拟网络耗时
        time.sleep(random.uniform(0.05, 0.2))

        rows = self.backend.sample(self.platform, self.per_page)
        return {"code": 200, "data": {"list": rows, "total": len(rows)}}

    # ------------------ 解析层 ------------------
    def _parse(self, payload: dict) -> list[CommentItem]:
        if payload.get("code") != 200:
            log.warning(f"非 200 响应: code={payload.get('code')}")
            return []

        items: list[CommentItem] = []
        for row in payload.get("data", {}).get("list", []):
            try:
                items.append(CommentItem(
                    comment_id=str(row["comment_id"]),
                    platform=str(row["platform"]),
                    product_id=str(row["product_id"]),
                    product_category=str(row["product_category"]),
                    user_id=str(row["user_id"]),
                    comment_time=str(row["comment_time"]),
                    is_additional=bool(row["is_additional"]),
                    comment_text=str(row["comment_text"]),
                    rating=int(row["rating"]),
                    useful_votes=int(row["useful_votes"]),
                    reply_status=bool(row["reply_status"]),
                    sentiment_label=str(row["sentiment_label"]),
                    attribute_labels=str(row["attribute_labels"]),
                    problem_label=bool(row["problem_label"]),
                    fake_risk_label=str(row["fake_risk_label"]),
                ))
            except (KeyError, ValueError, TypeError) as e:
                log.warning(f"解析失败，跳过一条: {e}")
        return items

    # ------------------ 单页抓取 ------------------
    def fetch_page(self, page: int) -> list[CommentItem]:
        url = self._build_url(page)
        try:
            payload = self._http_get(url)
        except requests.RequestException as e:
            log.error(f"page={page} 请求失败: {e}")
            return []
        items = self._parse(payload)
        log.info(f"page={page} 解析评论 {len(items)} 条")
        return items


# ---------------------------------------------------------------------------
# 去重 & 状态
# ---------------------------------------------------------------------------
class DedupStore:
    def __init__(self, state_path: str = STATE_PATH):
        self.state_path = state_path
        self.seen: set[str] = set()
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.seen = set(data.get("seen", []))
                log.info(f"加载断点状态: 已抓取 {len(self.seen)} 条")
            except Exception as e:
                log.warning(f"读取断点状态失败: {e}")

    def save(self) -> None:
        with self._lock:
            try:
                with open(self.state_path, "w", encoding="utf-8") as f:
                    json.dump({"seen": list(self.seen)}, f, ensure_ascii=False)
            except Exception as e:
                log.warning(f"保存断点状态失败: {e}")

    def filter_new(self, items: Iterable[CommentItem]) -> list[CommentItem]:
        out = []
        with self._lock:
            for it in items:
                fp = it.fingerprint()
                if fp in self.seen:
                    continue
                self.seen.add(fp)
                out.append(it)
        return out


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------
class CSVWriter:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._exists = os.path.exists(path)

    def write(self, items: list[CommentItem]) -> None:
        if not items:
            return
        with self._lock:
            mode = "a" if self._exists else "w"
            with open(self.path, mode, encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                if not self._exists:
                    writer.writeheader()
                    self._exists = True
                for it in items:
                    writer.writerow(asdict(it))


class JSONLWriter:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def write(self, items: list[CommentItem]) -> None:
        if not items:
            return
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 调度器
# ---------------------------------------------------------------------------
def run_crawler(
    platform: str,
    pages: int,
    per_page: int,
    workers: int,
    output_csv: str,
    output_jsonl: Optional[str] = None,
) -> None:
    log.info("=" * 64)
    log.info(f"启动爬虫 | 平台={platform} | 页数={pages} | 每页={per_page} | 并发={workers}")
    log.info("=" * 64)

    sample_url = f"https://{PLATFORM_HOSTS.get(platform, 'example.com')}/"
    if not check_robots(sample_url):
        log.error("robots.txt 禁止抓取，终止任务")
        return

    limiter = RateLimiter(min_interval=0.25, jitter=0.35)
    crawler = CommentCrawler(platform=platform, per_page=per_page, rate_limiter=limiter)
    dedup = DedupStore()
    csv_writer = CSVWriter(output_csv)
    jsonl_writer = JSONLWriter(output_jsonl) if output_jsonl else None

    total_new = 0
    total_dup = 0

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="spider") as pool:
        future_to_page = {pool.submit(crawler.fetch_page, p): p for p in range(1, pages + 1)}
        for fut in as_completed(future_to_page):
            page = future_to_page[fut]
            try:
                items = fut.result()
            except Exception as e:
                log.error(f"page={page} 抓取异常: {e}")
                continue

            new_items = dedup.filter_new(items)
            dup = len(items) - len(new_items)
            total_new += len(new_items)
            total_dup += dup

            csv_writer.write(new_items)
            if jsonl_writer:
                jsonl_writer.write(new_items)

            log.info(f"page={page} 新增 {len(new_items)} 条 / 重复 {dup} 条")
            dedup.save()

    log.info("-" * 64)
    log.info(f"完成：新增 {total_new} 条，去重过滤 {total_dup} 条 -> {output_csv}")
    if jsonl_writer:
        log.info(f"      JSONL: {output_jsonl}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="电商评论爬虫")
    parser.add_argument("--platform", choices=list(PLATFORM_HOSTS.keys()), default="淘宝")
    parser.add_argument("--pages", type=int, default=10, help="抓取页数")
    parser.add_argument("--per-page", type=int, default=20, help="每页条数")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "crawled_comments.csv"),
                        help="CSV 输出路径")
    parser.add_argument("--jsonl", default=None, help="可选: JSONL 输出路径")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--reset-state", action="store_true", help="清除断点续爬状态")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    if args.reset_state and os.path.exists(STATE_PATH):
        os.remove(STATE_PATH)
        log.info("已清除断点状态")

    run_crawler(
        platform=args.platform,
        pages=args.pages,
        per_page=args.per_page,
        workers=args.workers,
        output_csv=args.output,
        output_jsonl=args.jsonl,
    )


if __name__ == "__main__":
    main()
