



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project : General_Tool
@File    : run_harness.py
@Desc    : 长时训练流程的运行时保障（harness）：模型调用异常/超时处理 + 中断恢复

现状问题（LLMClient.generate_text / tool_optimizer_main）：

1. 异常不分类：`except Exception` 一把抓，401/400 这类**永远不会成功**的错误
   也照样重试 3 次、每次干等 3 秒，白烧 9 秒；而真正该多等的超时反而等太短。
2. 固定间隔重试：`time.sleep(3)` 无退避无抖动，服务抖动时所有请求同频重试，
   反而加重下游压力。
3. 异常逃逸击穿节点：重试耗尽后 `raise RuntimeError`，一路抛到 __main__ 的
   `except` 里 `continue`——该 resource_id 已经跑完的 epoch 全部作废。
4. 没有熔断：endpoint 整体挂掉时，会把剩余所有 resource_id 逐个重试一遍才结束。
5. 没有中断恢复：state 只在整个 resource_id 跑完后 save_pickle 一次，
   中途崩溃即全丢；`self.checkpointer` 建了却从未传给 graph.compile()，是死代码。

本模块提供两个组件：
- LLMCallGuard   ：异常分类 + 指数退避（带抖动）+ 连续失败熔断 + **不抛异常的降级返回**
- CheckpointStore：state 原子落盘与恢复，支持 epoch 粒度续跑
"""

from __future__ import annotations

import json
import os
import pickle
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

# 异常分类标签
KIND_TIMEOUT = "timeout"
KIND_CONNECTION = "connection"
KIND_RATE_LIMIT = "rate_limit"
KIND_SERVER = "server"
KIND_CLIENT = "client"
KIND_UNKNOWN = "unknown"

@dataclass
class CallStats:
    """调用统计，跑完一轮可以直接看健康度。"""

    calls: int = 0
    ok: int = 0
    failed: int = 0
    attempts: int = 0
    total_wait: float = 0.0
    by_kind: Dict[str, int] = field(default_factory=dict)

    def record_kind(self, kind: str) -> None:
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "ok": self.ok,
            "failed": self.failed,
            "attempts": self.attempts,
            "total_wait": round(self.total_wait, 2),
            "by_kind": dict(self.by_kind),
        }


class CircuitOpenError(RuntimeError):
    """熔断打开，直接快速失败，不再打下游。"""


class LLMCallGuard:
    """模型调用的异常/超时守卫。

    与 LLMClient 内置重试的分工：LLMClient 负责单次 HTTP 的低层重试，
    本守卫负责**分类**、**退避策略**和**降级**——调用方拿到的永远是
    (result, ok, error)，不会被异常击穿。
    """

    # 可重试：瞬时故障，等一等有机会成功
    RETRYABLE = {KIND_TIMEOUT, KIND_CONNECTION, KIND_RATE_LIMIT, KIND_SERVER}
    # 不可重试：请求本身有问题（鉴权失败、参数非法），重试只是浪费时间
    FATAL = {KIND_CLIENT}

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 30.0,
        jitter: float = 0.3,
        circuit_threshold: int = 0,
        sleep_fn: Optional[Callable[[float], None]] = None,
        rand_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        """
        Args:
            max_attempts: 单次调用最多尝试几次。
            base_delay: 退避基数，第 n 次失败后等 base_delay * 2**(n-1)。
            max_delay: 单次等待上限，避免指数爆炸。
            jitter: 抖动比例，实际等待在 [(1-j)d, (1+j)d] 之间随机，打散同频重试。
            circuit_threshold: 连续失败多少次后熔断；0 表示不启用。
            sleep_fn / rand_fn: 便于测试注入，生产留默认。
        """
        self.max_attempts = max(1, int(max_attempts))
        self.base_delay = float(base_delay)
        self.max_delay = float(max_delay)
        self.jitter = float(jitter)
        self.circuit_threshold = int(circuit_threshold)
        self.sleep_fn = sleep_fn or time.sleep
        self.rand_fn = rand_fn or random.random
        self.stats = CallStats()
        self.consecutive_failures = 0
        self.circuit_open = False

    @staticmethod
    def classify(exc: BaseException) -> str:
        """把异常归到可重试/不可重试的类别。

        不硬依赖 requests 的异常类型（按类名匹配），同时尽量从 HTTP 状态码判断：
        429 归 rate_limit、5xx 归 server（都可重试），其余 4xx 归 client（不可重试）。
        """
        name = type(exc).__name__.lower()
        text = str(exc).lower()

        status = None
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
        if status is None:
            # RuntimeError("... failed after N retries: 401 Client Error ...") 这类包装过的信息
            for code in ("429", "500", "502", "503", "504", "401", "403", "400", "404"):
                if code in text:
                    status = int(code)
                    break

        if status is not None:
            if status == 429:
                return KIND_RATE_LIMIT
            if 500 <= status < 600:
                return KIND_SERVER
            if 400 <= status < 500:
                return KIND_CLIENT

        if "timeout" in name or "timedout" in name or "timeout" in text:
            return KIND_TIMEOUT
        if "connection" in name or "connect" in text or "resolve" in text:
            return KIND_CONNECTION
        return KIND_UNKNOWN

    def _delay_for(self, attempt: int) -> float:
        """第 attempt 次失败后的等待时长（指数退避 + 抖动）。"""
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        low = 1.0 - self.jitter
        span = 2.0 * self.jitter
        return max(0.0, delay * (low + span * self.rand_fn()))

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, bool, Dict[str, Any]]:
        """执行调用，返回 (result, ok, error)。**任何情况下都不向外抛异常。**

        Returns:
            result: 成功时为 fn 的返回值，失败时为 None。
            ok: 是否成功。
            error: 失败详情 {"kind","message","attempts","retryable"}；成功时为 {}。
        """
        self.stats.calls += 1
        if self.circuit_open:
            self.stats.failed += 1
            self.stats.record_kind("circuit_open")
            return None, False, {
                "kind": "circuit_open",
                "message": "熔断已打开，跳过本次调用",
                "attempts": 0,
                "retryable": False,
            }

        last_kind = KIND_UNKNOWN
        last_message = ""
        for attempt in range(1, self.max_attempts + 1):
            self.stats.attempts += 1
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - 守卫的职责就是兜住一切
                last_kind = self.classify(exc)
                last_message = "%s: %s" % (type(exc).__name__, exc)
                self.stats.record_kind(last_kind)
                print(f"[guard] 第 {attempt}/{self.max_attempts} 次失败 kind={last_kind} {last_message}")

                if last_kind in self.FATAL:
                    # 鉴权/参数错误重试无意义，立刻停止，把剩余重试次数省下来
                    print("[guard] 判定为不可重试错误，放弃重试")
                    break
                if attempt < self.max_attempts:
                    delay = self._delay_for(attempt)
                    self.stats.total_wait += delay
                    print(f"[guard] 退避 {delay:.2f}s 后重试")
                    self.sleep_fn(delay)
                continue

            self.stats.ok += 1
            self.consecutive_failures = 0
            return result, True, {}

        self.stats.failed += 1
        self.consecutive_failures += 1
        if self.circuit_threshold > 0 and self.consecutive_failures >= self.circuit_threshold:
            self.circuit_open = True
            print(f"[guard] 连续失败 {self.consecutive_failures} 次，熔断打开")
        return None, False, {
            "kind": last_kind,
            "message": last_message,
            "attempts": min(attempt, self.max_attempts),
            "retryable": last_kind in self.RETRYABLE,
        }

    def reset_circuit(self) -> None:
        """手动复位熔断（例如换 endpoint 之后）。"""
        self.circuit_open = False
        self.consecutive_failures = 0

class CheckpointStore:
    """state 的原子落盘与恢复，支持中断后续跑。

    落盘粒度：每个节点跑完都写一次（便于诊断"死在哪个节点"），
    恢复粒度：epoch 级——state 里的 iteration 被保留，重启后 should_continue
    只会跑剩余的 epoch，当前这个未完成的 epoch 从 policy_generate 重做。
    没有做节点级恢复，因为节点内的 LLM 采样结果本身不可复现，
    半个 epoch 的中间态续跑意义不大且容易得出错乱的 group。
    """

    def __init__(self, root: str, resource_id: str, enabled: bool = True) -> None:
        self.root = root
        self.resource_id = str(resource_id)
        self.enabled = enabled

    @property
    def path(self) -> str:
        return os.path.join(self.root, "checkpoints", f"{self.resource_id}.ckpt")

    @property
    def meta_path(self) -> str:
        return os.path.join(self.root, "checkpoints", f"{self.resource_id}.meta.json")

    def save(self, node_name: str, state: Mapping[str, Any]) -> bool:
        """原子写：先写 .tmp 再 os.replace，避免崩在写一半留下坏文件。"""
        if not self.enabled:
            return False
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = self.path + ".tmp"
        try:
            with open(tmp_path, "wb") as writer:
                pickle.dump({"node": node_name, "state": dict(state)}, writer)
            os.replace(tmp_path, self.path)
            with open(self.meta_path, "w", encoding="utf-8") as writer:
                json.dump(
                    {
                        "resource_id": self.resource_id,
                        "last_node": node_name,
                        "iteration": state.get("iteration", 0),
                        "current_version_id": state.get("current_version_id", None),
                    },
                    writer,
                    ensure_ascii=False,
                    indent=2,
                )
            return True
        except Exception as exc:
            print(f"[ckpt] 保存失败（不影响主流程）: {exc}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return False

    def load(self) -> Optional[Dict[str, Any]]:
        """读回 {"node","state"}；文件不存在或损坏都返回 None（当作没有检查点）。"""
        if not self.enabled or not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "rb") as reader:
                payload = pickle.load(reader)
            if isinstance(payload, Mapping) and isinstance(payload.get("state"), Mapping):
                return dict(payload)
            print("[ckpt] 检查点结构异常，忽略")
        except Exception as exc:
            print(f"[ckpt] 检查点损坏，忽略并从头开始: {exc}")
        return None

    def clear(self) -> None:
        """跑完之后清掉检查点，避免下次误判为未完成。"""
        for path in (self.path, self.meta_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    print(f"[ckpt] 清理失败: {exc}")

    def resume_state(self, max_iterations: int) -> Tuple[Optional[Dict[str, Any]], str]:
        """恢复可用的 state。

        Returns:
            (state, reason)。state 为 None 表示无需/无法恢复，reason 说明原因。
        """
        payload = self.load()
        if payload is None:
            return None, "no_checkpoint"
        state = dict(payload["state"])
        iteration = int(state.get("iteration", 0) or 0)
        if iteration >= int(max_iterations):
            return None, "already_finished"
        state["max_iterations"] = int(max_iterations)
        return state, "resume_from_iteration_%d(last_node=%s)" % (iteration, payload.get("node", "?"))

