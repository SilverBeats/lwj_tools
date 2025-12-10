#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import warnings
from typing import Callable, Dict, Iterator, List, Optional, Union

import numpy as np

from .helper import get_file_name_and_ext
from .io import FileReader


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
    data_file_path: Optional[str] = None,
    output_file_path: Optional[str] = None,
    samples: Optional[List[dict]] = None,
    existed_samples: Optional[List[dict]] = None,
    id_field: str = "index",
    return_iter: bool = False
) -> Union[List[dict], Iterator[dict]]:
    """
    Obtaining unprocessed data samples is particularly suitable for API call scenarios

    This function returns unprocessed data samples by comparing the original data file with the processed data file.
    Supports returning list or iterator formats, facilitating memory optimization when processing large amounts of data.

    Args:
        data_file_path: The path of the original data file, containing all the data to be processed
        output_file_path: The file path where data is saved after API invocation, recording the processed data
        samples:
        existed_samples:
        id_field: The field name used to identify a unique sample, with the default being "index"
        return_iter: Whether to return the iterator format, True returns the iterator, False returns the list,
        and the default is False

    Returns:
        Union[List[dict], Iterator[dict]]: A list of unprocessed data samples or iterators
    """
    assert data_file_path or samples is not None, "Please specify data_file_path or samples"
    assert output_file_path or existed_samples is not None, "Please specify output_file_path or existed_samples"

    if data_file_path and samples:
        data_file_path = None
        warnings.warn("samples is specified, data_file_path will be ignored")

    if output_file_path and existed_samples:
        output_file_path = None
        warnings.warn("existed_samples is specified, output_file_path will be ignored")

    if data_file_path:
        data_ext = get_file_name_and_ext(data_file_path, False)[-1]
        assert data_ext == 'jsonl', "The file format must be jsonl"
        assert os.path.exists(data_file_path), "The data file does not exist"

    if output_file_path:
        out_ext = get_file_name_and_ext(output_file_path, False)[-1]
        assert out_ext == 'jsonl', "The file format must be jsonl"
        assert os.path.exists(output_file_path), "The output file does not exist"

    existed_samples = existed_samples or FileReader.read(output_file_path, return_iter=True)
    existed_ides = set([sample[id_field] for sample in existed_samples])

    def parse_fn(samples):
        samples = samples or FileReader.read(data_file_path, return_iter=True)
        for sample in samples:
            if sample[id_field] not in existed_ides:
                yield sample

    if return_iter:
        return parse_fn(samples)
    return list(parse_fn(samples))


def load_glove(file_path: str, skip_first_row: bool = False) -> Dict[str, np.ndarray]:
    word_2_emb = {}
    for line in FileReader.read(file_path, return_iter=True):
        if skip_first_row:
            continue
        try:
            line = line.strip()
            if not line:
                continue
            ws = line.split()
            word = ws[0]
            weights = np.asarray(map(float, ws[1:]))
            word_2_emb[word] = weights
        except Exception as e:
            warnings.warn(f'加载失败: {line}')
    return word_2_emb
