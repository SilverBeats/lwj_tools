#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pickle
from typing import Any, List, Union

import numpy as np
import pandas as pd
import yaml

from .tools import ext_check
from ..utils.common import get_file_name_and_ext


class FileWriter:

    @staticmethod
    @ext_check(
        ext=[
            "json", "jsonl", "txt", "yaml", "yml",
            "pkl", "xlsx", "xls", "csv", "tsv", "npy", "npz",
        ],
    )
    def dump(data: Any, file_path: str, sheets: str = "Sheet1", **specific_kwargs):
        """保存数据到文件

        Args:
            data: 数据
            file_path: 文件路径
            sheets: 工作表名称
            **specific_kwargs: 特定参数
        """
        ext = get_file_name_and_ext(file_path, False)[-1]
        kwargs = {}
        if ext in ["xlsx", "xls"]:
            kwargs["sheets"] = sheets

        if ext == "json":
            func = FileWriter.dump_json
        elif ext == "jsonl":
            func = FileWriter.dump_jsonl
        elif ext == "txt":
            func = FileWriter.dump_txt
        elif ext in ["yaml", "yml"]:
            func = FileWriter.dump_config
        elif ext == "pkl":
            func = FileWriter.dump_pkl
        elif ext in ["xlsx", "xlx"]:
            func = FileWriter.dump_excel
        elif ext in ["csv", "tsv"]:
            func = FileWriter.dump_csv
        else:  # ext in ["npy", "npz"]
            func = FileWriter.dump_npyz

        func(data, file_path, **kwargs, **specific_kwargs)

    @staticmethod
    @ext_check(ext=["json"])
    def dump_json(data: Any, file_path: str, **json_kwargs):
        """保存数据到json文件

        Args:
            data: 数据
            file_path:  文件路径
            **json_kwargs: `json.dump`的参数
        """
        os.makedirs(os.path.dirname(file_path) or "./", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, **json_kwargs)

    @staticmethod
    @ext_check(ext=["jsonl"])
    def dump_jsonl(data: List[Any], file_path: str, **json_kwargs):
        """保存数据到jsonl文件

        Args:
            data: 数据
            file_path: 文件路径
            **json_kwargs: `json.dumps`的参数
        """
        os.makedirs(os.path.dirname(file_path) or "./", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as json_file:
            for item in data:
                line = json.dumps(item, ensure_ascii=False, **json_kwargs)
                json_file.write(line + "\n")

    @staticmethod
    @ext_check(ext=["txt"])
    def dump_txt(data: List[str], file_path: str):
        """保存数据到txt文件

        Args:
            data: 数据
            file_path: 文件路径
        """
        os.makedirs(os.path.dirname(file_path) or "./", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as txt_file:
            for line in data:
                txt_file.write(line.strip() + "\n")

    @staticmethod
    @ext_check(ext=["yaml", "yml"])
    def dump_yaml(data: dict, file_path: str):
        """保存数据到yaml文件

        Args:
            data: 数据
            file_path: 文件路径
        """
        os.makedirs(os.path.dirname(file_path) or "./", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)

    @staticmethod
    @ext_check(ext=["pkl"])
    def dump_pkl(data: Any, file_path: str, **pickle_kwargs):
        """保存数据到pkl文件

        Args:
            data: 数据
            file_path: 文件路径
            **pickle_kwargs: `pickle.dump`的参数
        """
        os.makedirs(os.path.dirname(file_path) or "./", exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(data, f, **pickle_kwargs)

    @staticmethod
    @ext_check(ext=["xlsx", "xls"])
    def dump_excel(
        data: Union[pd.DataFrame, List[pd.DataFrame]],
        file_path: str,
        sheets: Union[str, List[str]] = "Sheet1",
        **pd_kwargs,
    ):
        """保存数据到excel文件

        Args:
            data: 数据
            file_path: 文件路径
            sheets: 工作表名称
            **pd_kwargs: `pandas.DataFrame.to_excel`的参数
        """
        if isinstance(data, pd.DataFrame):
            data = [data]

        if isinstance(sheets, str):
            sheets = [sheets]

        assert len(data) == len(sheets), "data and sheets must have same length"
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            for df, sheet in zip(data, sheets):
                df.to_excel(writer, sheet_name=sheet, **pd_kwargs)

    @staticmethod
    @ext_check(ext=["csv", "tsv"])
    def dump_csv(data: pd.DataFrame, file_path: str, **pd_kwargs):
        """保存数据到csv文件

        Args:
            data: 数据
            file_path: 文件路径
            **pd_kwargs: `pandas.DataFrame.to_csv`的参数
        """
        os.makedirs(os.path.dirname(file_path) or "./", exist_ok=True)
        data.to_csv(file_path, **pd_kwargs)

    @staticmethod
    @ext_check(ext=["npy", "npz"])
    def dump_npyz(data: Any, file_path: str):
        """保存数据到npy或npz文件

        Args:
            data: 数据
            file_path: 文件路径
        """
        os.makedirs(os.path.dirname(file_path) or "./", exist_ok=True)
        ext = get_file_name_and_ext(file_path, with_dot=False)[1]
        if ext == "npz":
            np.savez(file_path, data)
        else:
            np.save(file_path, data)
