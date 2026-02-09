#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Any, Callable, Optional

from ..errors import PromptTemplateGeneratingError, PromptTemplateParsingError


class PromptTemplate:
    """管理提示模板和解析 LLM 返回结果，不建议修改 `generate_prompt` 和 `parse`

    Examples:
        >>> class MyPromptTemplate(PromptTemplate):
        ...     prompt = "{NUM_1} + {NUM_2} 等于多少？"
        ...     def generate_fn(self, num1: int, num2: int):
        ...         return self.prompt.format(NUM_1=num1, NUM_2=num2)
        ...     def parse_fn(self, llm_response: str):
        ...         return llm_response
    """

    def __init__(
        self,
        name: Optional[str] = None,
        generate_fn: Optional[Callable] = None,
        parse_fn: Optional[Callable] = None,
    ):
        self._name = name
        self._generate_fn = generate_fn or self.generate_fn
        self._parse_fn = parse_fn or self.parse_fn

    def generate_prompt(self, *args, **kwargs):
        """生成提示模板"""
        if self._generate_fn is None:
            raise NotImplementedError("Please implement the generate_fn")

        try:
            return self._generate_fn(*args, **kwargs)
        except Exception as e:
            raise PromptTemplateGeneratingError(str(e))

    def parse(self, result: Any, *args, **kwargs):
        """解析 LLM 返回结果"""
        if self._parse_fn is None:
            raise NotImplementedError("Please implement the parse_fn")

        try:
            return self._parse_fn(result, *args, **kwargs)
        except Exception as e:
            raise PromptTemplateParsingError(str(e))

    @property
    def name(self):
        return self._name or __name__

    def generate_fn(self, *args):
        return args

    def parse_fn(self, llm_response):
        return llm_response
