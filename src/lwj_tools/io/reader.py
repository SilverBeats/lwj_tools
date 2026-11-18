#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文件读取工具集。

:class:`FileReader` 提供按扩展名分发的静态方法读 json / jsonl / txt / yaml /
pkl / npy / npz / csv / tsv / xlsx / xls 等常见格式。所有读方法都用
:func:`lwj_tools.io.tools.ext_check` 做扩展名校验，类型不符会抛
:class:`lwj_tools.errors.FileTypeError`。

:meth:`FileReader.read` 是按扩展名自动分发的统一入口；其余 `read_xxx` 是
对应格式的特化版本（多接受几个格式相关参数）。

Example:
    >>> from lwj_tools.io.reader import FileReader
    >>> data = FileReader.read_json("config.json")
    >>> for row in FileReader.read_csv("data.csv", return_iter=True):
    ...     pass
"""

import csv
import json
import pickle
from typing import Any, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from openpyxl.reader.excel import load_workbook

from ..common._typing import FilePath
from ..common.files import get_file_name_and_ext
from .tools import ext_check


class FileReader:
    """按扩展名分发的静态文件读取工具集。"""

    # ------------------------------------------------------------------ #
    # 单格式读取
    # ------------------------------------------------------------------ #

    @staticmethod
    @ext_check(ext=["json"])
    def read_json(file_path: FilePath, **json_kwargs) -> Any:
        """读取 json 文件。

        Args:
            file_path: json 文件路径。
            **json_kwargs: 透传给 :func:`json.load`。

        Returns:
            json 文件内容。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f, **json_kwargs)

    @staticmethod
    @ext_check(ext=["jsonl"])
    def read_jsonl(
        file_path: FilePath,
        return_iter: bool = False,
        **json_kwargs,
    ) -> Union[List[Any], Iterator[Any]]:
        """读取 jsonl 文件（每行一个 JSON 对象）。

        空行会被跳过。

        Args:
            file_path: jsonl 文件路径。
            return_iter: 是否返回迭代器。
            **json_kwargs: 透传给 :func:`json.loads`。

        Returns:
            `return_iter=False` 返回列表；`True` 返回迭代器。
        """

        def parse_fn() -> Iterator[Any]:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line, **json_kwargs)

        return parse_fn() if return_iter else list(parse_fn())

    @staticmethod
    @ext_check(ext=["txt"])
    def read_txt(
        file_path: FilePath,
        return_iter: bool = False,
        strip: bool = True,
    ) -> Union[List[str], Iterator[str]]:
        """读取 txt 文件，按行返回。

        Args:
            file_path: txt 文件路径。
            return_iter: 是否返回迭代器。
            strip: 是否对每行调用 :meth:`str.strip`。`False` 时仅去掉行尾
                换行符（保留其他空白）。

        Returns:
            `return_iter=False` 返回列表；`True` 返回生成器。
        """

        def parse_fn() -> Iterator[str]:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.strip() if strip else line.rstrip("\n")

        return parse_fn() if return_iter else list(parse_fn())

    @staticmethod
    @ext_check(ext=["yaml", "yml"])
    def read_yaml(file_path: FilePath) -> Any:
        """读取 yaml 文件。

        Args:
            file_path: yaml 文件路径。

        Returns:
            :func:`yaml.safe_load` 的结果，可能是 `dict` / `list` / 标量。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    @ext_check(ext=["pkl"])
    def read_pkl(file_path: FilePath, **pickle_kwargs) -> Any:
        """读取 pkl 文件。

        Args:
            file_path: pkl 文件路径。
            **pickle_kwargs: 透传给 :func:`pickle.load`。

        Returns:
            反序列化结果。
        """
        with open(file_path, "rb") as f:
            return pickle.load(f, **pickle_kwargs)

    @staticmethod
    @ext_check(ext=["npy", "npz"])
    def read_npyz(file_path: FilePath, **numpy_kwargs) -> Any:
        """读取 npy 或 npz 文件。

        Args:
            file_path: npy / npz 文件路径。
            **numpy_kwargs: 透传给 :func:`numpy.load`。

        Returns:
            npy 返回 :class:`numpy.ndarray`；npz 返回 `NpzFile`。
        """
        return np.load(file_path, **numpy_kwargs)

    @staticmethod
    @ext_check(ext=["xlsx", "xls"])
    def read_large_excel(
        file_path: FilePath,
        sheet_name: str = "Sheet1",
    ) -> Iterator[dict]:
        """以流式方式读取大体积 excel，逐行 yield `{列名: 值}`。

        基于 openpyxl 的 read-only 模式，内存占用与文件大小无关。

        Args:
            file_path: excel 文件路径。
            sheet_name: sheet 名。

        Yields:
            每一行组成的 `{列名: 单元格值}` 字典。
        """
        wb = load_workbook(file_path, read_only=True)
        try:
            ws = wb[sheet_name]
            rows = ws.iter_rows()
            header_row = next(rows, None)
            if header_row is None:
                return
            header = [cell.value for cell in header_row]
            for row in rows:
                yield dict(zip(header, (cell.value for cell in row)))
        finally:
            wb.close()

    @staticmethod
    @ext_check(ext=["xlsx", "xls"])
    def read_excel(
        file_path: FilePath,
        sheet_name: str = "Sheet1",
        return_iter: bool = False,
        return_dict: bool = False,
    ) -> Union[pd.DataFrame, Iterator[dict], List[dict]]:
        """读取 excel 文件。

        Args:
            file_path: excel 文件路径。
            sheet_name: sheet 名。
            return_iter: 是否返回流式迭代器（大文件推荐）。
            return_dict: 是否返回 `list[dict]` 而非 :class:`pandas.DataFrame`。
                `return_iter=True` 时此参数被忽略（迭代器恒为 dict 流）。

        Returns:
            - `return_iter=True`：`Iterator[dict]`
            - `return_dict=True`：`List[dict]`
            - 否则：:class:`pandas.DataFrame`
        """
        if return_iter:
            return FileReader.read_large_excel(file_path, sheet_name)

        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df.to_dict("records") if return_dict else df

    @staticmethod
    @ext_check(ext=["csv", "tsv"])
    def read_csv(
        file_path: FilePath,
        delimiter: str = ",",
        return_iter: bool = False,
        return_dict: bool = False,
        has_header: bool = True,
        **pd_kwargs,
    ) -> Union[pd.DataFrame, Iterator[Union[dict, list]], List[Union[dict, list]]]:
        """读取 csv / tsv 文件。

        Args:
            file_path: csv / tsv 文件路径。
            delimiter: 分隔符。`.tsv` 文件会被强制覆盖为 `"\\t"`。
            return_iter: 是否返回迭代器。
            return_dict: 是否以 `dict` 形式返回行。
            has_header: 首行是否为表头。`True` 时迭代模式会跳过首行；
                `False` 且 `return_dict=True` 时用 0-based 索引作列名。
            **pd_kwargs: 透传给 :func:`pandas.read_csv`（仅非迭代模式）。
                `encoding` / `delimiter` 提供默认值但可被覆盖。

        Returns:
            - `return_iter=True`：`Iterator[dict]` 或 `Iterator[list]`
            - `return_dict=True`：`List[dict]`
            - 否则：:class:`pandas.DataFrame`
        """
        ext = get_file_name_and_ext(file_path, False)[-1].lower()
        if ext == "tsv":
            delimiter = "\t"

        def parse_fn() -> Iterator[Union[dict, list]]:
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter=delimiter)

                column_names: Optional[List] = None
                if has_header:
                    header_row = next(reader, None)
                    if header_row is None:
                        return
                    if return_dict:
                        column_names = [s.strip() for s in header_row]
                elif return_dict:
                    first_row = next(reader, None)
                    if first_row is None:
                        return
                    column_names = list(range(len(first_row)))
                    yield dict(zip(column_names, first_row))

                for row in reader:
                    if return_dict:
                        yield dict(zip(column_names, row))
                    else:
                        yield row

        if return_iter:
            return parse_fn()

        pd_kwargs.setdefault("encoding", "utf-8")
        pd_kwargs.setdefault("delimiter", delimiter)
        df = pd.read_csv(file_path, **pd_kwargs)
        return df.to_dict("records") if return_dict else df

    # ------------------------------------------------------------------ #
    # 统一入口
    # ------------------------------------------------------------------ #

    @staticmethod
    @ext_check(
        ext=[
            "json",
            "jsonl",
            "txt",
            "yaml",
            "yml",
            "pkl",
            "xlsx",
            "xls",
            "csv",
            "tsv",
            "npy",
            "npz",
        ],
    )
    def read(
        file_path: FilePath,
        return_iter: bool = False,
        return_dict: bool = False,
        delimiter: str = ",",
        has_header: bool = True,
        sheet_name: str = "Sheet1",
        **specific_kwargs,
    ):
        """按扩展名分发到对应 `read_xxx`。

        Args:
            file_path: 文件路径。
            return_iter: 是否返回迭代器（仅部分格式支持）。
            return_dict: 是否返回 `dict`（仅 excel / csv / tsv 支持）。
            delimiter: csv / tsv 分隔符。
            has_header: csv / tsv 首行是否为表头。
            sheet_name: excel sheet 名。
            **specific_kwargs: 透传给具体 `read_xxx` 的额外参数。

        Returns:
            对应格式的读取结果。
        """
        ext = get_file_name_and_ext(file_path, False)[-1].lower()

        if ext == "json":
            return FileReader.read_json(file_path, **specific_kwargs)
        if ext in ("yaml", "yml"):
            return FileReader.read_yaml(file_path, **specific_kwargs)
        if ext == "pkl":
            return FileReader.read_pkl(file_path, **specific_kwargs)
        if ext in ("npy", "npz"):
            return FileReader.read_npyz(file_path, **specific_kwargs)
        if ext == "jsonl":
            return FileReader.read_jsonl(
                file_path,
                return_iter=return_iter,
                **specific_kwargs,
            )
        if ext == "txt":
            return FileReader.read_txt(
                file_path,
                return_iter=return_iter,
                **specific_kwargs,
            )
        if ext in ("xlsx", "xls"):
            return FileReader.read_excel(
                file_path,
                sheet_name=sheet_name,
                return_iter=return_iter,
                return_dict=return_dict,
                **specific_kwargs,
            )
        if ext in ("csv", "tsv"):
            return FileReader.read_csv(
                file_path,
                delimiter=delimiter,
                return_iter=return_iter,
                return_dict=return_dict,
                has_header=has_header,
                **specific_kwargs,
            )

        # ext_check 已保证 ext 落在上述集合中；此处不可达。
        raise ValueError(f"Unhandled extension: {ext!r}")
