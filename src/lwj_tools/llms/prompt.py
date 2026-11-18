"""Prompt 模板：业务参数 → prompt 字符串 → 解析 → 校验。

:class:`PromptTemplate` 把一次 LLM 调用中与业务强相关的部分收敛到三个钩子：

===============  ======================================================
钩子              职责
===============  ======================================================
generate_fn      把业务参数渲染成 prompt 字符串
parse_fn         把 LLM 原始响应解析成结构化结果
valid_fn         判断结构化结果是否合法（可选，默认不做校验）
===============  ======================================================

钩子既可以由子类覆写，也可以在构造时以函数参数注入，两种写法等价。
每次钩子调用失败都会被包装成对应的业务异常
（:class:`PromptTemplateGeneratingError` /
:class:`PromptTemplateParsingError` /
:class:`PromptTemplateValidError`），异常链（``__cause__``）保留原始异常。

Example:
    >>> from lwj_tools.llms.prompt import PromptTemplate
    >>> class SumPrompt(PromptTemplate):
    ...     def generate_fn(self, num1: int, num2: int) -> str:
    ...         return f"{num1} + {num2} 等于多少？"
    ...     def parse_fn(self, llm_response: str) -> int:
    ...         return int("".join(ch for ch in llm_response if ch.isdigit()))
    >>> pt = SumPrompt()
    >>> pt.generate_prompt(1, 2)
    '1 + 2 等于多少？'
    >>> pt.parse("等于 3")
    3
"""

from typing import Any, Callable, Optional

from ..errors import (
    SUCCEED_CODE,
    PromptTemplateGeneratingError,
    PromptTemplateParsingError,
    PromptTemplateValidError,
)

__all__ = ["PromptTemplate"]


class PromptTemplate:
    """Prompt 模板基类。

    子类需要覆写 :meth:`generate_fn` 与 :meth:`parse_fn`；:meth:`valid_fn`
    是可选的，不覆写时不做任何校验。

    也可以在 :meth:`__init__` 里直接把钩子函数作为参数传入，等价于子类覆写。

    Example:
        >>> class MyPromptTemplate(PromptTemplate):
        ...     prompt = "{NUM_1} + {NUM_2} 等于多少？"
        ...     def generate_fn(self, num1: int, num2: int) -> str:
        ...         return self.prompt.format(NUM_1=num1, NUM_2=num2)
        ...     def parse_fn(self, llm_response: str) -> Any:
        ...         return llm_response
    """

    def __init__(
        self,
        name: Optional[str] = None,
        generate_fn: Optional[Callable] = None,
        parse_fn: Optional[Callable] = None,
        valid_fn: Optional[Callable] = None,
    ) -> None:
        """初始化模板。

        Args:
            name: 模板名（用于日志 / 调试），省略时回退到类名。
            generate_fn: 覆写 :meth:`generate_fn`；省略时使用基类实现。
            parse_fn: 覆写 :meth:`parse_fn`；省略时使用基类实现。
            valid_fn: 覆写 :meth:`valid_fn`；省略时不做校验。
        """
        self._name = name
        self._generate_fn = generate_fn or self.generate_fn
        self._parse_fn = parse_fn or self.parse_fn
        self._valid_fn = valid_fn or self.valid_fn

    def generate_prompt(self, *args, **kwargs) -> str:
        """调用 :meth:`generate_fn` 生成 prompt 字符串。

        Args:
            *args: 透传给 :meth:`generate_fn` 的位置参数。
            **kwargs: 透传给 :meth:`generate_fn` 的关键字参数。

        Returns:
            渲染后的 prompt 字符串。

        Raises:
            PromptTemplateGeneratingError: :meth:`generate_fn` 抛错，已包装为业务异常。
            NotImplementedError: :meth:`generate_fn` 未覆写。
        """
        try:
            return self._generate_fn(*args, **kwargs)
        except (PromptTemplateGeneratingError, NotImplementedError):
            raise
        except Exception as e:
            raise PromptTemplateGeneratingError(str(e)) from e

    def parse(self, result: Any, *args, **kwargs) -> Any:
        """调用 :meth:`parse_fn` 把 LLM 响应解析为结构化结果。

        Args:
            result: LLM 原始响应文本。
            *args: 透传给 :meth:`parse_fn` 的位置参数。
            **kwargs: 透传给 :meth:`parse_fn` 的关键字参数。

        Returns:
            解析后的结构化结果。

        Raises:
            PromptTemplateParsingError: :meth:`parse_fn` 抛错，已包装为业务异常。
            NotImplementedError: :meth:`parse_fn` 未覆写。
        """
        try:
            return self._parse_fn(result, *args, **kwargs)
        except (PromptTemplateParsingError, NotImplementedError):
            raise
        except Exception as e:
            raise PromptTemplateParsingError(str(e)) from e

    def valid(self, result: Any) -> str:
        """调用 :meth:`valid_fn` 校验 :meth:`parse` 的结果。

        :meth:`valid_fn` 返回 ``True`` 表示结果合法；返回 ``False`` 或抛错都
        视为不合法。

        Args:
            result: :meth:`parse` 的返回值。

        Returns:
            :data:`lwj_tools.errors.SUCCEED_CODE`（``"success"``）。

        Raises:
            PromptTemplateValidError: 校验不通过。
        """
        try:
            passed = self._valid_fn(result)
        except PromptTemplateValidError:
            raise
        except Exception as e:
            raise PromptTemplateValidError(str(e)) from e

        if not passed:
            raise PromptTemplateValidError(
                f"{type(self).__name__} 校验未通过：{result!r}"
            )

        return SUCCEED_CODE

    @property
    def name(self) -> str:
        """模板名；构造时未指定则回退到类名。"""
        return self._name if self._name is not None else type(self).__name__

    def generate_fn(self, *args, **kwargs) -> str:
        """把业务参数渲染成 prompt 字符串 —— 必须由子类覆写。

        Raises:
            NotImplementedError: 未覆写时抛出。
        """
        raise NotImplementedError(
            f"{type(self).__name__}.generate_fn must be overridden"
        )

    def parse_fn(self, llm_response: Any, *args, **kwargs) -> Any:
        """把 LLM 响应解析成结构化结果 —— 必须由子类覆写。

        Args:
            llm_response: LLM 原始响应文本。
            *args: 透传的位置参数。
            **kwargs: 透传的关键字参数。

        Raises:
            NotImplementedError: 未覆写时抛出。
        """
        raise NotImplementedError(
            f"{type(self).__name__}.parse_fn must be overridden"
        )

    def valid_fn(self, result: Any) -> bool:
        """校验解析结果 —— 返回 ``False`` 表示不合法。

        默认实现不做任何校验，直接返回 ``True``。

        Args:
            result: :meth:`parse` 的返回值。

        Returns:
            结果是否合法。
        """
        return True
