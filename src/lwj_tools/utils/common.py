#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import base64
import hashlib
import json
import logging
import os
import random
import re
import shutil
import uuid
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Pattern, Sequence, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import regex


def random_choice(arr: Union[Sequence, Set], n: int = 1) -> List[Any]:
    """从数组中随机选择n个元素

    Args:
        arr: 数组
        n: 随机选择的元素个数

    Returns:
        List[Any]: 随机选择的元素
    """
    return random.sample(arr, min(n, len(arr)))


def shuffle(arr: List[Any], n: int = 1):
    """随机打乱数组中的元素顺序（原地操作）

    Args:
        arr: 数组
        n: 随机打乱的次数
    """
    for _ in range(n):
        random.shuffle(arr)


def str2bool(v) -> bool:
    """将字符串转换为布尔值，适用于 argparse

    Args:
        v: 输入的字符串

    Returns:
        bool: 转换后的布尔值
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("不支持的值")


def get_logger(
    name: str,
    level: str = "info",
    formatter: Optional[str] = None,
    log_path: Optional[str] = None,
) -> logging.Logger:
    """获取

    Args:
        name: logger 名称
        level: log 级别
        formatter: log formatter
        log_path: log 文件路径
    """
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARN,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
    }

    assert level in LEVELS

    logger = logging.getLogger(name)

    if not logger.handlers:
        level_ = LEVELS[level]
        logger.setLevel(level_)

        fmt = (
            formatter
            if formatter is not None
            else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log_formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

        ch = logging.StreamHandler()
        ch.setLevel(level_)
        ch.setFormatter(log_formatter)
        logger.addHandler(ch)

        if log_path is not None:
            dirname = os.path.dirname(log_path)
            os.makedirs(dirname, exist_ok=True)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(level_)
            fh.setFormatter(log_formatter)
            logger.addHandler(fh)

    return logger


def camel_to_snake(name: str) -> str:
    """将驼峰命名法转换为蛇形命名法

    Args:
        name: 驼峰命名法的字符串

    Returns:
        str: 蛇形命名法的字符串
    """
    s1 = regex.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = regex.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    return s2


def get_uuid(prefix: Optional[str] = None) -> str:
    """获取 uuid

    Args:
        prefix: uuid 的前缀

    Returns:
        str: uuid
    """
    if prefix is not None:
        return f"{prefix}-{uuid.uuid4().hex}"
    return uuid.uuid4().hex


def get_md5_id(text: str) -> str:
    """获取文本的MD5值

    Args:
        text: 文本

    Returns:
        str: MD5值
    """
    hash_str = hashlib.md5(text.encode("utf-8")).hexdigest()
    return hash_str


def get_base64(file_path: str) -> str:
    """ 获取文件的base64编码

    Args:
        file_path: 文件路径

    Returns:
        str: base64编码
    """
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded


def is_url(url: str) -> bool:
    """判断是否为有效URL

    Args:
        url: URL字符串

    Returns:
        bool: 是否为有效URL
    """
    try:
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])
    except Exception as e:
        return False


def get_file_name_and_ext(file_path: str, with_dot: bool = True) -> Tuple[str, str]:
    """获取文件名和扩展名

    Args:
        file_path: 文件路径
        with_dot: 是否包含扩展名的点

    Returns:
        str: 文件名
        str: 扩展名，with_dot 为 True 时包含点，为 False 时不含点
    """

    parts = os.path.splitext(os.path.basename(file_path))
    file_name, ext = parts[0], parts[-1]
    if not with_dot:
        ext = ext[1:]
    return file_name, ext


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算两个向量的余弦相似度

    Args:
        a: shape = (a_len, emb_dim)
        b: shape = (b_len, emb_dim)

    Returns:
        np.ndarray: 余弦相似度矩阵 shape = (a_len, b_len)
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    sim_matrix = np.dot(a, b.T) / (a_norm * b_norm.T)
    return sim_matrix


def get_dir_file_path(
    dir_path: str,
    file_exts: Optional[List[str]] = None,
    skip_dirs: Optional[List[Union[str, Pattern]]] = None,
    skip_files: Optional[List[Union[str, Pattern]]] = None,
    return_abs: bool = False,
    should_skip_file: Optional[Callable[[str], bool]] = None,
) -> List[str]:
    """扫描目录并返回具有增强跳过支持的文件路径列表。

    Args:
        dir_path: 目录
        file_exts: 文件扩展名列表（不带点），如果为 None，则包含所有文件
        skip_dirs: 要跳过的目录名（或正则表达式模式）列表。
        skip_files: 要跳过的文件名（或正则表达式模式）列表。
        return_abs: 是否返回绝对路径
        should_skip_file: 可选函数，接受完整文件路径并返回 True 以跳过。

    Returns:
        List[str]: 获得的路径

    Examples:
        >>> import re
        >>> get_dir_file_path(
        ...     dir_path='old',
        ...     skip_files=[re.compile(r'io*.*')],
        ...     skip_dirs=['test'],
        ...     should_skip_file=lambda s: 'nlg' in s,
        ...  )
    """

    if not os.path.isdir(dir_path):
        return []

    file_exts = file_exts or []
    skip_dirs = skip_dirs or []
    skip_files = skip_files or []

    compiled_skip_dirs = [
        (re.compile(pattern) if isinstance(pattern, str) else pattern)
        for pattern in skip_dirs
    ]
    compiled_skip_files = [
        (re.compile(pattern) if isinstance(pattern, str) else pattern)
        for pattern in skip_files
    ]

    file_paths = []

    for root, dirs, files in os.walk(dir_path):
        if root != dir_path:
            # 检查是否跳过当前目录（根据目录名）
            dir_name = os.path.basename(root)
            should_skip_current_dir = False
            for pattern in compiled_skip_dirs:
                if pattern.fullmatch(dir_name):
                    should_skip_current_dir = True
                    break
            if should_skip_current_dir:
                dirs.clear()  # 禁止访问子目录
                continue

        # 处理文件
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # 首先，检查是否应该跳过它（通过自定义函数）
            if should_skip_file and should_skip_file(file_path):
                continue

            # 检查文件名是否与 skip_files 正则表达式匹配
            should_skip = False
            for pattern in compiled_skip_files:
                if pattern.fullmatch(file_name):
                    should_skip = True
                    break
            if should_skip:
                continue

            # 检查扩展名
            ext = get_file_name_and_ext(file_name, False)[-1]
            if file_exts and ext not in file_exts:
                continue

            # 转换为绝对路径
            final_path = os.path.realpath(file_path) if return_abs else file_path
            file_paths.append(final_path)

    return file_paths


def rm_file(file_path: str):
    """删除文件

    Args:
        file_path: 待删除文件
    """
    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)


def rm_dir(dir_path: str, ignore_errors: bool = True, onerror: Optional[Callable] = None):
    """删除目录

    Args:
        dir_path: 待删除的目录
        ignore_errors: 由`shutil.rmtree`使用
        onerror: 由`shutil.rmtree`使用
    """
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path, ignore_errors, onerror)


def clean_dir(dir_path: str, ignore_errors: bool = True, onerror: Optional[Callable] = None):
    """清空目录

    Args:
        dir_path:  待清空目录
        ignore_errors: 由`shutil.rmtree`使用
        onerror: 由`shutil.rmtree`使用
    """
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path, ignore_errors, onerror)
        os.makedirs(dir_path or "./", exist_ok=True)


def get_unprocessed_samples(
    data_file_path: Optional[str] = None,
    output_file_path: Optional[str] = None,
    samples: Optional[List[dict]] = None,
    existed_samples: Optional[List[dict]] = None,
    id_field: str = "index",
    return_iter: bool = False
) -> Union[List[dict], Iterator[dict]]:
    """获取未处理的数据样本特别适合API调用场景.
    该函数通过比较原始数据文件和处理后的数据文件，返回未处理的数据样本。
    支持返回列表或生成器格式，便于处理大量数据时的内存优化。

    Args:
        data_file_path: 待处理样本，jsonl 格式
        output_file_path: 已经处理好的样本，jsonl 格式
        samples: 待处理样本
        existed_samples: 已经处理好的样本
        id_field: 用于标识唯一示例的字段名，默认值为“index”
        return_iter: True 则返回生成器格式，False返回列表

    Returns:
        Union[List[dict], Iterator[dict]]：未处理数据样本
    """
    assert data_file_path or samples is not None, "请指定 `data_file_path` 或 `samples`"
    assert output_file_path or existed_samples is not None, "请指定 `output_file_path` 或 `existed_samples`"

    if data_file_path and samples:
        data_file_path = None
        warnings.warn("如果指定了 `samples`，则忽略 `data_file_path`")

    if output_file_path and existed_samples:
        output_file_path = None
        warnings.warn("如果指定了 `existed_samples`， `output_file_path` 将被忽略")

    if data_file_path:
        data_ext = get_file_name_and_ext(data_file_path, False)[-1]
        assert data_ext == 'jsonl', f"{data_file_path} 文件格式必须为 jsonl"
        assert os.path.exists(data_file_path), f"{data_file_path} 文件不存在"

    def _load_file_data(file_path: str):
        if not (file_path and os.path.exists(file_path)):
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    existed_ides = []
    if output_file_path:
        out_ext = get_file_name_and_ext(output_file_path, False)[-1]
        assert out_ext == 'jsonl', f"{output_file_path} 文件格式必须为 jsonl"

    existed_samples = existed_samples or _load_file_data(output_file_path)
    existed_ides = [sample[id_field] for sample in existed_samples]

    def _filter_fn(samples):
        samples = samples or _load_file_data(data_file_path)
        for sample in samples:
            if sample[id_field] not in existed_ides:
                yield sample

    if return_iter:
        return _filter_fn(samples)
    return list(_filter_fn(samples))


def load_glove(file_path: str, skip_first_row: bool = False, delimiter: str = ' ') -> Dict[str, np.ndarray]:
    """ 加载GloVe词向量文件

    Args:
        file_path: glove 文件路径
        skip_first_row: 有些文件第一行是 (词个数, 维度)，可以设置为True跳过
        delimiter: 分隔符，默认空格

    Returns:
        Dict[str, np.ndarray]: 词向量字典
    """
    word_2_emb = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        if skip_first_row:
            next(f)

        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ws = line.split(delimiter)
                word = ws[0]
                weights = np.array([float(x) for x in ws[1:]], dtype=np.float32)
                word_2_emb[word] = weights
            except Exception as e:
                warnings.warn(f'加载失败: {line}')
    return word_2_emb
