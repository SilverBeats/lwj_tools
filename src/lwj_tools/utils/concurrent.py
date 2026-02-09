#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import traceback
import warnings
from abc import ABC
from concurrent.futures import Future, as_completed
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Iterable, List, Optional, Tuple

from tqdm import tqdm

from .common import get_logger
from ..errors import ConcurrentError

LOGGER = get_logger('lwj_tools')


def wrapper(
    idx: int,
    sample: Any,
    worker_func: Callable,
    callback_func: Optional[Callable] = None,
):
    result = worker_func(idx, sample)
    if callback_func:
        result = callback_func(result, sample)
    return idx, result


class ConcurrentRunner(ABC):
    def __init__(
        self,
        executor_cls: Callable,
        num_workers: int = -1,
        use_pbar: bool = True,
        stop_by_error: bool = False,
        verbose: bool = False,
        need_order: bool = False,
    ):
        """并发运行器

        Args:
            executor_cls: 执行器类，如 `ProcessPoolExecutor` 或 `ThreadPoolExecutor`
            num_workers: 并发数，默认为 CPU 核心数
            use_pbar: 是否使用进度条，默认为 `True`
            stop_by_error: 是否遇到错误就停止，默认为 `False`
            verbose: 是否详细输出，默认为 `False`
            need_order: 是否需要顺序返回结果，默认为 `False`
        """
        if num_workers == -1:
            num_workers = os.cpu_count()
        self.executor_cls = executor_cls
        self.num_workers = num_workers
        self.use_pbar = use_pbar
        self.stop_by_error = stop_by_error
        self.verbose = verbose
        self.need_order = need_order

    def __call__(
        self,
        samples: Iterable,
        worker_func: Callable,
        callback_func: Optional[Callable] = None,
        finished_func: Optional[Callable] = None,
        n_samples: int = None,
        pbar_desc: str = "Running",
    ) -> Tuple[List[Any], List[ConcurrentError]]:
        """ 并发运行

        Args:
            samples: 样本集合
            worker_func: 工作函数
            callback_func: 回调函数，默认为 `None`
            finished_func: 完成回调函数，默认为 `None`
            n_samples: 样本数量，默认为 `None`
            pbar_desc: 进度条描述，默认为 "Running"

        Returns:
            List[Any]: 处理结果。如果有finished_func，则列表中元素类型取决于 finished_func 的返回值，否则取决于 worker_func 的返回值
            List[ConcurrentError]: 报错信息列表
        """
        if n_samples is None:
            try:
                n_samples = len(samples)
            except TypeError:
                try:
                    samples = list(samples)
                    n_samples = len(samples)
                except Exception as e:
                    warnings.warn(f"{e}\n无法获取样本数量。请手动指定`n_samples`参数。")
                    n_samples = None

        wrapper_func = partial(
            wrapper,
            worker_func=worker_func,
            callback_func=callback_func,
        )
        results = [None] * n_samples if self.need_order else []

        with self.executor_cls(max_workers=min(n_samples, self.num_workers)) as executor:
            try:
                tasks: List[Future] = []
                for idx, sample in enumerate(samples):
                    task = executor.submit(wrapper_func, idx, sample)
                    tasks.append(task)
            except Exception as e:
                error = ConcurrentError(f'提交任务失败: {traceback.format_exc()}')
                if self.verbose:
                    LOGGER.error(error)
                if self.stop_by_error:
                    raise error
                else:
                    return results

            pbar = None
            if self.use_pbar:
                pbar = tqdm(total=n_samples, desc=pbar_desc, dynamic_ncols=True, leave=True)

            error_list = []
            try:
                for task in as_completed(tasks):
                    try:
                        idx, result = task.result()
                        if finished_func:
                            finished_func(idx, result)
                        if self.need_order:
                            results[idx] = result
                        else:
                            results.append(result)
                    except Exception as e:
                        error = ConcurrentError(traceback.format_exc())
                        error_list.append(error)

                        if self.verbose:
                            LOGGER.error(error)

                        if self.stop_by_error:
                            for t in tasks:
                                if not t.done():
                                    t.cancel()
                            executor.shutdown(wait=False)
                            raise error
                    finally:
                        if pbar:
                            pbar.update(1)
                            pbar.refresh()
            finally:
                if pbar:
                    pbar.close()
                return results, error_list


class MultiProcessRunner(ConcurrentRunner):
    """多进程运行器"""

    def __init__(
        self,
        num_workers: int = -1,
        use_pbar: bool = True,
        stop_by_error: bool = False,
        verbose: bool = False,
        need_order: bool = False,
    ):
        super().__init__(
            ProcessPoolExecutor,
            num_workers,
            use_pbar,
            stop_by_error,
            verbose,
            need_order,
        )


class MultiThreadingRunner(ConcurrentRunner):
    """多线程运行器"""

    def __init__(
        self,
        num_workers: int = -1,
        use_pbar: bool = True,
        stop_by_error: bool = False,
        verbose: bool = False,
        need_order: bool = False,
    ):
        super().__init__(
            ThreadPoolExecutor,
            num_workers,
            use_pbar,
            stop_by_error,
            verbose,
            need_order,
        )
