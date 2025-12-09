#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import hashlib
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

from .io import FileReader

__all__ = [
    "random_choice",
    "str2bool",
    "get_logger",
    "get_dir_file_path",
    "camel_to_snake",
    "shuffle",
    "get_uuid",
    "get_md5_id",
    "get_base64",
    "is_url",
    "rm_dir",
    "rm_file",
    "clean_dir",
    "get_file_name_and_ext",
]


def random_choice(arr: Union[Sequence, Set], n: int = 1) -> List[Any]:
    """
    random choice n elements from array
    Args:
        arr: array of elements
        n: the number of elements you want to select
    """
    return random.sample(arr, min(n, len(arr)))


def str2bool(v):
    """Used in argparse, when you want to set a bool parameter"""
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def get_logger(
    name: str,
    level: str = "info",
    formatter: Optional[str] = None,
    log_path: Optional[str] = None,
) -> logging.Logger:
    """get a logger

    Args:
        name: logger name
        level: log level. Defaults to "info".
        formatter: log formatter. Defaults to None.
        log_path: log file path. Defaults to None.
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


def get_dir_file_path(
    dir_path: str,
    file_exts: Optional[List[str]] = None,
    skip_dirs: Optional[List[Union[str, Pattern]]] = None,
    skip_files: Optional[List[Union[str, Pattern]]] = None,
    is_abs: bool = False,
    should_skip_file: Optional[Callable[[str], bool]] = None,
) -> List[str]:
    """
    Scan a directory and return a list of file paths with enhanced skip support.

    Args:
        dir_path: Root directory path.
        file_exts: List of file extensions to include (without dot). If None, include all.
        skip_dirs: List of directory names (or regex patterns) to skip.
        skip_files: List of file names (or regex patterns) to skip.
        is_abs: Return absolute paths.
        should_skip_file: Optional function that takes full file path and returns True to skip.

    Returns:
        List of file paths.
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
            # Check whether the current directory should be skipped (based on the directory name)
            dir_name = os.path.basename(root)
            should_skip_current_dir = False
            for pattern in compiled_skip_dirs:
                if pattern.fullmatch(dir_name):
                    should_skip_current_dir = True
                    break
            if should_skip_current_dir:
                dirs.clear()  # Prevent access to subdirectories
                continue

        # Handle files
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # First, check whether it should be skipped (through a custom function)
            if should_skip_file and should_skip_file(file_path):
                continue

            # Check if the file name matches the skip_files regular expression
            should_skip = False
            for pattern in compiled_skip_files:
                if pattern.fullmatch(file_name):
                    should_skip = True
                    break
            if should_skip:
                continue

            # Check the extension
            ext = get_file_name_and_ext(file_name, False)[-1]
            if file_exts and ext not in file_exts:
                continue

            # change to the absolute path
            final_path = os.path.realpath(file_path) if is_abs else file_path
            file_paths.append(final_path)

    return file_paths


def camel_to_snake(name: str) -> str:
    """use this function to change a camel style name to snake style name"""
    s1 = regex.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return regex.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def shuffle(arr: List[Any], n):
    """shuffle a list"""
    for _ in range(n):
        random.shuffle(arr)


def get_uuid(prefix: Optional[str] = None) -> str:
    """return uuid"""
    if prefix is not None:
        return f"{prefix}-{uuid.uuid4().hex}"
    return uuid.uuid4().hex


def get_md5_id(text: str) -> str:
    """return text's md5 value"""
    hash_str = hashlib.md5(text.encode("utf-8")).hexdigest()
    return hash_str


def get_base64(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def is_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])
    except Exception as e:
        return False


def get_file_name_and_ext(file_path: str, with_dot: bool = True) -> Tuple[str, str]:
    """
    Args:
        file_path:
        with_dot: extension with dot or not

    Returns:
        (file name, extension)
    """

    parts = os.path.splitext(os.path.basename(file_path))
    file_name, ext = parts[0], parts[-1]
    if not with_dot:
        ext = ext[1:]
    return file_name, ext


def rm_file(file_path: str):
    """
    Args:
        file_path: the file path you want to delete
    """
    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)


def rm_dir(dir_path: str, ignore_errors: bool = True, onerror: Optional[Callable] = None):
    """
    Args:
        dir_path: the directory you want to delete
        ignore_errors: used by shutil.rmtree
        onerror: used by shutil.rmtree
    """
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path, ignore_errors, onerror)


def clean_dir(dir_path: str, ignore_errors: bool = True, onerror: Optional[Callable] = None):
    """
    Args:
        dir_path: the directory you want to clean
        ignore_errors: used by shutil.rmtree
        onerror: used by shutil.rmtree
    """
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path, ignore_errors, onerror)
        os.makedirs(dir_path or "./", exist_ok=True)


def get_unprocessed_samples(
    data_file_path: str,
    output_file_path: str,
    id_field: str = "index",
    return_iter: bool = False,
) -> Union[List[dict], Iterator[dict]]:
    """
    Obtaining unprocessed data samples is particularly suitable for API call scenarios

    This function returns unprocessed data samples by comparing the original data file with the processed data file.
    Supports returning list or iterator formats, facilitating memory optimization when processing large amounts of data.

    Args:
        data_file_path: The path of the original data file, containing all the data to be processed
        output_file_path: The file path where data is saved after API invocation, recording the processed data
        id_field: The field name used to identify a unique sample, with the default being "index"
        return_iter: Whether to return the iterator format, True returns the iterator, False returns the list,
        and the default is False

    Returns:
        Union[List[dict], Iterator[dict]]: A list of unprocessed data samples or iterators
    """
    data_ext = get_file_name_and_ext(data_file_path, False)[-1]
    out_ext = get_file_name_and_ext(output_file_path, False)[-1]

    if data_ext != 'jsonl' or out_ext != 'jsonl':
        raise ValueError('The file format must be jsonl')

    existed_ides = set()
    if output_file_path is not None and os.path.exists(output_file_path):
        for sample in FileReader.read(output_file_path, return_iter=True):
            existed_ides.add(sample[id_field])

    def parse_fn():
        for sample in FileReader.read(data_file_path, return_iter=True):
            if sample[id_field] not in existed_ides:
                yield sample

    if return_iter:
        return parse_fn()
    return list(parse_fn())


def load_glove(file_path: str, skip_first_row: bool = False) -> Dict[str, np.ndarray]:
    word_2_emb = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        if skip_first_row:
            f.readline()
        while True:
            line = f.readline().strip()
            if not line:
                break
            try:
                ws = line.split()
                word = ws[0]
                weights = np.asarray(map(float, ws[1:]))
                word_2_emb[word] = weights
            except Exception as e:
                warnings.warn(f'加载失败: {line}')
    return word_2_emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Args:
        a: shape = (a_len, emb_dim)
        b: shape = (b_len, emb_dim)

    Returns:
        cosine similarity matrix. shape = (a_len, b_len)
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T) / (a_norm * b_norm.T)
