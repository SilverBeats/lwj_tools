#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass
class TimeProxyResult:
    result: Optional[Any] = None
    timecost: float = 0  # s


def timecost(
    proxy_func, time_func=time.perf_counter, unit: Literal["s", "ms", "ns"] = "s"
):
    """装饰器，计算函数执行时间

    Args:
        proxy_func: 被装饰的函数
        time_func: 计时函数，默认使用 time.perf_counter
        unit: 时间单位，可选值有 "s"、"ms"、"ns"，默认为 "s"

    Returns:
        TimeProxyResult: 装饰器返回结果的数据类，包含 result 和 timecost 两个字段

    Examples:
        >>> @timecost
        ... def foo():
        ...     time.sleep(1)
        ...     return 1
        >>> foo()
        TimeProxyResult(result=1, timecost=1.0)
    """

    def wrapper(*args, **kwargs) -> TimeProxyResult:
        with Timer(time_func, unit) as timer:
            func_result = proxy_func(*args, **kwargs)
        return TimeProxyResult(func_result, timer.elapsed)

    return wrapper


class Timer:
    """计时器
    Examples:
        >>> with Timer() as timer:
        ...     do_something()
        ... print(timer.elapsed)
    """

    def __init__(self, func=time.perf_counter, unit: Literal["s", "ms", "ns"] = "s"):
        """初始化计时器

        Args:
            func: 计时函数，默认使用 time.perf_counter
            unit: 时间单位，可选值有 "s"、"ms"、"ns"，默认为 "s"
        """
        self.elapsed = 0.0
        self._func = func
        self._start = None
        self._unit = unit
        self._unit_factor = {"s": 1, "ms": 1e3, "ns": 1e6}.get(unit, 1)

    def start(self):
        """启动计时器"""
        if self._start is not None:
            raise RuntimeError("Already started")
        self._start = self._func()

    def stop(self):
        """停止计时器"""
        if self._start is None:
            raise RuntimeError("Not started")
        end = self._func()
        self.elapsed += (end - self._start) * self._unit_factor
        self._start = None

    def reset(self):
        """重置计时器"""
        self.elapsed = 0.0

    @property
    def is_running(self):
        """计时器是否正在运行"""
        return self._start is not None

    def __enter__(self):
        """上下文管理器，启动计时器"""
        self.start()
        return self

    def __exit__(self, *args):
        """上下文管理器，停止计时器"""
        self.stop()
