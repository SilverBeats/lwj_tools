#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文件写入工具集。

:class:`FileWriter` 提供按扩展名分发的静态方法写 json / jsonl / txt / yaml /
pkl / npy / npz / csv / tsv / xlsx / xls 等常见格式。所有写方法都用
:func:`lwj_tools.io.tools.ext_check` 做扩展名校验；目录不存在时会自动
:meth:`os.makedirs` 创建。

:meth:`FileWriter.dump` 是按扩展名自动分发的统一入口；其余 `dump_xxx` 是
对应格式的特化版本。

Example:
    >>> from lwj_tools.io.writer import FileWriter
    >>> FileWriter.dump_json({"a": 1}, "out.json")
    >>> FileWriter.dump_csv(df, "out.csv", index=False)
"""

import json
import os
import pickle
from typing import Any, Iterable, List, Mapping, Sequence, Union

import numpy as np
import pandas as pd
import yaml

from ..common._typing import FilePath
from ..common.files import get_file_name_and_ext
from .tools import ext_check


def _ensure_dir(file_path: FilePath) -> None:
    """确保 `file_path` 的父目录存在；纯文件名时使用当前目录。"""
    os.makedirs(os.path.dirname(str(file_path)) or "./", exist_ok=True)


class FileWriter:
    """按扩展名分发的静态文件写入工具集。"""

    # ------------------------------------------------------------------ #
    # 单格式写入
    # ------------------------------------------------------------------ #

    @staticmethod
    @ext_check(ext=["json"])
    def dump_json(data: Any, file_path: FilePath, **json_kwargs) -> None:
        """保存到 json 文件。

        Args:
            data: 待写入的数据。
            file_path: json 文件路径。
            **json_kwargs: 透传给 :func:`json.dump`。`ensure_ascii` 默认
                `False`（保留非 ASCII 字符），可显式覆盖。
        """
        _ensure_dir(file_path)
        json_kwargs.setdefault("ensure_ascii", False)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, **json_kwargs)

    @staticmethod
    @ext_check(ext=["jsonl"])
    def dump_jsonl(data: Iterable[Any], file_path: FilePath, **json_kwargs) -> None:
        """保存到 jsonl 文件（每行一个 JSON 对象）。

        Args:
            data: 任意可迭代对象。
            file_path: jsonl 文件路径。
            **json_kwargs: 透传给 :func:`json.dumps`。`ensure_ascii` 默认
                `False`，可显式覆盖。
        """
        _ensure_dir(file_path)
        json_kwargs.setdefault("ensure_ascii", False)
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, **json_kwargs) + "\n")

    @staticmethod
    @ext_check(ext=["txt"])
    def dump_txt(
        data: Union[str, Iterable[str]],
        file_path: FilePath,
        strip: bool = True,
    ) -> None:
        """保存到 txt 文件。

        Args:
            data: 单个字符串，或字符串的可迭代对象（逐行写入）。
            file_path: txt 文件路径。
            strip: 是否对每行调用 :meth:`str.strip`。`False` 时原样写入，
                仅在行尾补 `\\n`。
        """
        _ensure_dir(file_path)
        lines = [data] if isinstance(data, str) else data
        with open(file_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write((line.strip() if strip else line) + "\n")

    @staticmethod
    @ext_check(ext=["yaml", "yml"])
    def dump_yaml(data: Any, file_path: FilePath, **yaml_kwargs) -> None:
        """保存到 yaml 文件。

        使用 :func:`yaml.safe_dump`，不写出 Python 对象标签；
        `allow_unicode` 默认 `True`，中文不会被转义。

        Args:
            data: 待写入的数据。
            file_path: yaml 文件路径。
            **yaml_kwargs: 透传给 :func:`yaml.safe_dump`，可覆盖默认值。
        """
        _ensure_dir(file_path)
        yaml_kwargs.setdefault("allow_unicode", True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, **yaml_kwargs)

    @staticmethod
    @ext_check(ext=["pkl"])
    def dump_pkl(data: Any, file_path: FilePath, **pickle_kwargs) -> None:
        """保存到 pkl 文件。

        Args:
            data: 待写入的数据。
            file_path: pkl 文件路径。
            **pickle_kwargs: 透传给 :func:`pickle.dump`。
        """
        _ensure_dir(file_path)
        with open(file_path, "wb") as f:
            pickle.dump(data, f, **pickle_kwargs)

    @staticmethod
    @ext_check(ext=["xlsx", "xls"])
    def dump_excel(
        data: Union[pd.DataFrame, Sequence[pd.DataFrame]],
        file_path: FilePath,
        sheets: Union[str, Sequence[str]] = "Sheet1",
        **pd_kwargs,
    ) -> None:
        """保存到 excel 文件。

        Args:
            data: 单个 :class:`pandas.DataFrame`，或 DataFrame 序列。
            file_path: excel 文件路径。
            sheets: sheet 名，或与 `data` 等长的 sheet 名序列。
            **pd_kwargs: 透传给 :meth:`pandas.DataFrame.to_excel`。

        Raises:
            ValueError: `data` 与 `sheets` 长度不一致。
        """
        _ensure_dir(file_path)
        frames = [data] if isinstance(data, pd.DataFrame) else list(data)
        sheet_names = [sheets] if isinstance(sheets, str) else list(sheets)

        if len(frames) != len(sheet_names):
            raise ValueError(
                f"data ({len(frames)}) and sheets ({len(sheet_names)}) "
                "must have the same length"
            )

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            for frame, sheet in zip(frames, sheet_names):
                frame.to_excel(writer, sheet_name=sheet, **pd_kwargs)

    @staticmethod
    @ext_check(ext=["csv", "tsv"])
    def dump_csv(data: pd.DataFrame, file_path: FilePath, **pd_kwargs) -> None:
        """保存到 csv / tsv 文件。

        Args:
            data: :class:`pandas.DataFrame`。
            file_path: csv / tsv 文件路径。`.tsv` 会自动用 `\\t` 作为
                分隔符（可显式覆盖 `sep`）。
            **pd_kwargs: 透传给 :meth:`pandas.DataFrame.to_csv`。
        """
        _ensure_dir(file_path)
        ext = get_file_name_and_ext(file_path, with_dot=False)[-1].lower()
        if ext == "tsv":
            pd_kwargs.setdefault("sep", "\t")
        pd_kwargs.setdefault("encoding", "utf-8")
        data.to_csv(file_path, **pd_kwargs)

    @staticmethod
    @ext_check(ext=["npy", "npz"])
    def dump_npyz(
        data: Union[Any, Mapping[str, Any]],
        file_path: FilePath,
        **numpy_kwargs,
    ) -> None:
        """保存到 npy 或 npz 文件。

        Args:
            data: `.npy` 时写入单个数组；`.npz` 时可传 :class:`Mapping`
                （键作数组名）或任意对象（以 `arr_0` 命名）。
            file_path: npy / npz 文件路径。
            **numpy_kwargs: 透传给 :func:`numpy.save` 或 :func:`numpy.savez`。
        """
        _ensure_dir(file_path)
        ext = get_file_name_and_ext(file_path, with_dot=False)[-1].lower()
        if ext == "npz":
            if isinstance(data, Mapping):
                np.savez(file_path, **data, **numpy_kwargs)
            else:
                np.savez(file_path, data, **numpy_kwargs)
        else:
            np.save(file_path, data, **numpy_kwargs)

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
    def dump(
        data: Any,
        file_path: FilePath,
        sheets: Union[str, List[str]] = "Sheet1",
        **specific_kwargs,
    ) -> None:
        """按扩展名分发到对应 `dump_xxx`。

        Args:
            data: 待写入的数据。
            file_path: 文件路径。
            sheets: excel sheet 名（单 sheet 传 `str`，多 sheet 传 `list`）。
                非 excel 格式忽略此参数。
            **specific_kwargs: 透传给具体 `dump_xxx` 的额外参数。
        """
        ext = get_file_name_and_ext(file_path, with_dot=False)[-1].lower()

        if ext == "json":
            FileWriter.dump_json(data, file_path, **specific_kwargs)
        elif ext == "jsonl":
            FileWriter.dump_jsonl(data, file_path, **specific_kwargs)
        elif ext == "txt":
            FileWriter.dump_txt(data, file_path, **specific_kwargs)
        elif ext in ("yaml", "yml"):
            FileWriter.dump_yaml(data, file_path, **specific_kwargs)
        elif ext == "pkl":
            FileWriter.dump_pkl(data, file_path, **specific_kwargs)
        elif ext in ("xlsx", "xls"):
            FileWriter.dump_excel(data, file_path, sheets=sheets, **specific_kwargs)
        elif ext in ("csv", "tsv"):
            FileWriter.dump_csv(data, file_path, **specific_kwargs)
        elif ext in ("npy", "npz"):
            FileWriter.dump_npyz(data, file_path, **specific_kwargs)
        else:
            # ext_check 已保证 ext 落在上述集合中；此处不可达。
            raise ValueError(f"Unhandled extension: {ext!r}")
