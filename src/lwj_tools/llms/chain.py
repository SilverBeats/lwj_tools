"""LLM 调用编排层。

把 :class:`lwj_tools.llms.client.LLMClientGroup` 与
:class:`lwj_tools.llms.prompt.PromptTemplate` 组合成
「参数 → prompt → LLM 调用 → 解析 → 校验 → 结果」的工作流。

消息格式沿用 OpenAI 的
:data:`openai.types.chat.ChatCompletionMessageParam`：``history`` 直接传入
OpenAI 格式的消息字典，``system_prompt`` 是纯文本，``images`` 会被编码成
多模态 content part。

两个 chain 的差别：

- :class:`LLMChain` —— 调用一次即返回，失败时按 ``max_retries`` 重试。
- :class:`LLMChainWithValid` —— 结果不合法时把校验错误回传给模型，让它自纠，
  最多重试 ``max_chat_turns`` 轮。

契约：
    ``__call__`` 不向外抛异常，成败一律通过返回的 :class:`ChainResult` 判断：

    - 成功：:attr:`ChainResult.succeeded` 为 ``True``，
      :attr:`ChainResult.status_code` 为 ``"success"``
    - 失败：:attr:`ChainResult.error` 为具体异常，
      :attr:`ChainResult.status_code` 为该异常的错误码

Example:
    >>> from lwj_tools.llms.chain import LLMChain
    >>> from lwj_tools.llms.client import APIConfig, LLMClientGroup, TaskType
    >>> from lwj_tools.llms.prompt import PromptTemplate
    >>> group = LLMClientGroup(TaskType.CHAT, [
    ...     APIConfig(model="gpt-4o-mini", api_base="https://api.openai.com/v1", api_key="sk-..."),
    ... ])
    >>> class MyPrompt(PromptTemplate):
    ...     def generate_fn(self, topic: str) -> str:
    ...         return f"Tell me about {topic} in one sentence."
    ...     def parse_fn(self, response: str) -> str:
    ...         return response.strip()
    >>> chain = LLMChain(group, MyPrompt())
    >>> result = chain("LLM evaluation")
    >>> if result.succeeded:
    ...     print(result.result)
"""

from __future__ import annotations

import threading
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from openai.types.chat import ChatCompletionMessageParam

from ..common.pojo import DictLike
from ..date.timer import Timer
from ..errors import (
    BASE_ERROR_CODE,
    SUCCEED_CODE,
    LLMClientError,
    PromptTemplateParsingError,
    PromptTemplateValidError,
)
from .client import LLMClient, LLMClientGroup, LLMResponse, TaskType, encode_image
from .prompt import PromptTemplate

__all__ = [
    "ChainResult",
    "BaseLLMChain",
    "LLMChain",
    "LLMChainWithValid",
]


@dataclass(eq=False, repr=False)
class ChainResult(DictLike):
    """单次 chain 调用的结果记录。

    Attributes:
        api_params: 实际送出的 API 参数（默认值与调用时覆盖值的合并结果）。
        prompt_args: 传给 :meth:`PromptTemplate.generate_prompt` 的原始参数。
        prompt: 渲染后的 prompt 字符串。
        prompt_template: 使用的 :class:`PromptTemplate` 实例。
        response: LLM 原始响应文本。
        details: 底层响应对象（OpenAI SDK 的 ``ChatCompletion``；流式时是已消费
            的 chunk 列表）。
        result: :meth:`PromptTemplate.parse` 的返回值。
        reasoning_content: 推理类模型返回的思维链内容（无则为 ``None``）。
        error: 调用过程中的异常（成功时为 ``None``）。
        timecost: 端到端耗时（秒）。
        in_tokens: 输入 token 数（跨轮次、跨重试累计）。
        out_tokens: 输出 token 数（跨轮次、跨重试累计）。
        client_key: 命中 client 的 :attr:`LLMClient.encrypted_api_key`。
    """

    api_params: Optional[dict] = None
    prompt_args: Any = None
    prompt: Optional[str] = None
    prompt_template: Optional[PromptTemplate] = None
    response: Optional[str] = None
    details: Optional[Any] = None
    result: Any = None
    reasoning_content: Optional[str] = None
    error: Optional[Exception] = None
    timecost: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    client_key: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """调用是否成功（等价于 :attr:`error` 为 ``None``）。"""
        return self.error is None

    @property
    def status_code(self) -> str:
        """调用状态码：成功为 ``"success"``，失败为异常自身的错误码。"""
        if self.error is None:
            return SUCCEED_CODE
        return getattr(self.error, "code", BASE_ERROR_CODE)

    def to_dict(self, deep: bool = True, *, detect_cycles: bool = True) -> dict:
        """把结果序列化为普通 dict。

        在基类递归序列化之上做两处特化：

        - ``prompt_template`` 序列化为其类名（不暴露模板内部状态）。
        - ``details`` 若来自 OpenAI SDK，则序列化为普通 dict。

        ``error`` 保持原异常对象。
        """
        data = super().to_dict(deep=deep, detect_cycles=detect_cycles)

        if self.prompt_template is not None:
            data["prompt_template"] = type(self.prompt_template).__name__

        details = data.get("details")
        if hasattr(details, "model_dump"):
            data["details"] = details.model_dump()

        return data


class BaseLLMChain(ABC):
    """Chain 抽象基类 —— client 选择、参数拼接、消息组装、单次调用与重试原语。

    子类只需实现 :meth:`__call__` 的编排逻辑；重试循环由
    :meth:`_retry_invoke` 统一提供。

    Args:
        client_group: :class:`LLMClientGroup` 实例，任务类型须为
            :attr:`TaskType.CHAT`。
        prompt_template: :class:`PromptTemplate` 实例。
        **api_params: 所有调用请求的默认 API 参数。

    Raises:
        ValueError: ``client_group`` 的任务类型不是 chat。
    """

    def __init__(
        self,
        client_group: LLMClientGroup,
        prompt_template: PromptTemplate,
        **api_params,
    ) -> None:
        """初始化 Chain。"""
        if client_group.task_type is not TaskType.CHAT:
            raise ValueError(
                f"LLMChain 只支持 chat 任务，当前 client group 的任务类型是 "
                f"{client_group.task_type.value}"
            )

        self._client_group = client_group
        self._prompt_template = prompt_template
        self._lock = threading.Lock()
        self._default_api_params = api_params

    # ------------------------------------------------------------------ #
    # client 与参数
    # ------------------------------------------------------------------ #

    def _choose_client(
        self,
        ignored_clients: Optional[Sequence[LLMClient]] = None,
    ) -> Optional[LLMClient]:
        """从 :attr:`_client_group` 选一个可用 client。

        Args:
            ignored_clients: 本轮需要跳过的 client（失败重试时屏蔽坏 client）。

        Returns:
            选中的 client；没有可用 client 时返回 ``None``。
        """
        with self._lock:
            return self._client_group.find_available_client(
                ignored_clients=ignored_clients,
            )

    def obtain_api_params(
        self,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        **api_params,
    ) -> dict:
        """合并默认 API 参数与调用时参数。

        显式传入的 ``top_p`` / ``temperature`` / ``seed`` 会覆盖默认值；
        OpenAI 不直接支持 ``top_k``，这里把它并入 ``extra_body``。

        Args:
            top_p: 核采样阈值。
            temperature: 采样温度。
            seed: 随机种子。
            **api_params: 其他 API 参数。

        Returns:
            合并后的 API 参数。
        """
        new_api_params = {**self._default_api_params}

        # 复制一份，避免原地污染调用方传入的 extra_body
        extra_body = dict(api_params.pop("extra_body", {}))

        # top_k 不是 OpenAI 的顶层参数，只能通过 extra_body 传给后端
        if "top_k" in api_params:
            extra_body["top_k"] = api_params.pop("top_k")

        if temperature is not None:
            new_api_params["temperature"] = temperature
        if top_p is not None:
            new_api_params["top_p"] = top_p
        if seed is not None:
            new_api_params["seed"] = seed
        if extra_body:
            new_api_params["extra_body"] = extra_body

        new_api_params.update(api_params)
        return new_api_params

    # ------------------------------------------------------------------ #
    # 消息组装
    # ------------------------------------------------------------------ #

    def _build_messages(
        self,
        history: Optional[Sequence[ChatCompletionMessageParam]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[ChatCompletionMessageParam]:
        """组装 system 与 history 部分的消息列表。

        Args:
            history: 对话历史，OpenAI 格式的消息字典序列。
            system_prompt: 系统提示词，置于消息列表最前。

        Returns:
            OpenAI 格式的消息列表。
        """
        messages: List[ChatCompletionMessageParam] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        return messages

    @staticmethod
    def _build_user_message(
        prompt: str,
        images: Optional[Sequence[str]] = None,
    ) -> ChatCompletionMessageParam:
        """组装当前轮的 user 消息。

        有图片时 content 是多模态片段列表：先是文本片段，再是各图片片段；
        本地图片路径会被编码成 data URL。

        Args:
            prompt: 渲染后的 prompt 文本。
            images: 图片 URL 或本地路径。

        Returns:
            OpenAI 格式的 user 消息。
        """
        if not images:
            return {"role": "user", "content": prompt}

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": encode_image(image)}}
            for image in images
        )
        return {"role": "user", "content": content}

    # ------------------------------------------------------------------ #
    # 调用与重试
    # ------------------------------------------------------------------ #

    def _invoke(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        client: LLMClient,
        stream: bool = False,
        timeout: float = 600.0,
        **api_params,
    ) -> LLMResponse:
        """单次（不重试）LLM 调用。

        Args:
            messages: 完整的 OpenAI 消息列表。
            client: 本次使用的 client。
            stream: 是否流式返回。
            timeout: HTTP 超时（秒）。
            **api_params: 其他 API 参数。

        Returns:
            :class:`LLMResponse`。
        """
        return client.response(
            messages,
            stream=stream,
            timeout=timeout,
            **api_params,
        )

    def _retry_invoke(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        max_retries: int = 1,
        stream: bool = False,
        timeout: float = 600.0,
        **api_params,
    ) -> dict:
        """围绕 :meth:`_invoke` 的重试原语。

        异常处理矩阵：

        - :class:`PromptTemplateParsingError` —— 解析瞬时失败，换一次重试机会。
        - :class:`LLMClientError` 且已选中 client —— 屏蔽该 client 后重试。
        - :class:`LLMClientError` 且没有可用 client —— 直接退出。
        - 其他异常 —— 确定性失败，记录后退出。

        :class:`PromptTemplateGeneratingError` 不在本循环内捕获 —— 它发生在
        prompt 渲染阶段，重试没有意义，由调用方单独处理。

        Args:
            messages: 完整的 OpenAI 消息列表。
            max_retries: 最大尝试次数（``1`` 表示只调用一次）。
            stream: 是否流式返回。
            timeout: HTTP 超时（秒）。
            **api_params: 其他 API 参数。

        Returns:
            结果 dict；失败时含 ``error`` 字段。
        """
        ignored_clients: List[LLMClient] = []
        client: Optional[LLMClient] = None
        error: Optional[Exception] = None

        total_in = 0
        total_out = 0
        total_time = 0.0

        for _ in range(max(1, max_retries)):
            try:
                client = self._choose_client(ignored_clients)
                if client is None:
                    raise LLMClientError("没有可用的 LLMClient")

                llm_response = self._invoke(
                    messages,
                    client=client,
                    stream=stream,
                    timeout=timeout,
                    **api_params,
                )
            except LLMClientError as exc:
                error = exc
                if client is None:
                    break
                ignored_clients.append(client)
                continue
            except Exception as exc:
                # 确定性异常，重试无意义
                error = exc
                break

            total_in += llm_response.in_tokens
            total_out += llm_response.out_tokens
            total_time += llm_response.timecost

            try:
                parse_result = self._prompt_template.parse(llm_response.response)
            except PromptTemplateParsingError as exc:
                # 解析瞬时失败：记录后换一次重试机会
                error = exc
                continue

            return {
                "client_key": client.encrypted_api_key,
                "reasoning_content": self._extract_reasoning_content(
                    llm_response.details
                ),
                "response": llm_response.response,
                "details": llm_response.details,
                "in_tokens": total_in,
                "out_tokens": total_out,
                "timecost": total_time,
                "result": parse_result,
            }

        # 所有尝试都失败：保留已经消耗的 token 统计
        return {
            "error": error,
            "in_tokens": total_in,
            "out_tokens": total_out,
            "timecost": total_time,
        }

    @staticmethod
    def _extract_reasoning_content(details: Any) -> Optional[str]:
        """从底层响应中提取推理模型的思维链内容。

        非流式响应的 ``details`` 是 ``ChatCompletion``，思维链在
        ``choices[0].message.reasoning_content``；流式响应的 ``details`` 是已
        消费的 chunk 列表，思维链散落在各 chunk 的 delta 中。

        Args:
            details: :attr:`LLMResponse.details`。

        Returns:
            思维链文本；没有思维链时返回 ``None``。
        """
        if details is None:
            return None

        choices = getattr(details, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            return getattr(message, "reasoning_content", None) or None

        if isinstance(details, list):
            parts: List[str] = []
            for chunk in details:
                chunk_choices = getattr(chunk, "choices", None)
                if not chunk_choices:
                    continue
                delta = getattr(chunk_choices[0], "delta", None)
                reasoning_content = getattr(delta, "reasoning_content", None)
                if reasoning_content:
                    parts.append(reasoning_content)
            return "".join(parts) or None

        return None


class LLMChain(BaseLLMChain):
    """单轮 LLM chain —— 调用一次 LLM 即返回。

    支持简单重试（``max_retries``）。契约见模块文档：``__call__`` 不抛异常，
    成败由 :attr:`ChainResult.succeeded` 判断。
    """

    def __call__(
        self,
        *prompt_tmpl_args,
        history: Optional[Sequence[ChatCompletionMessageParam]] = None,
        images: Optional[Sequence[str]] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        max_retries: int = 1,
        timeout: float = 600.0,
        # generate params
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        **api_params,
    ) -> ChainResult:
        """调用 LLM。

        Args:
            *prompt_tmpl_args: 传给 :meth:`PromptTemplate.generate_prompt` 的参数。
            history: 对话历史，OpenAI 格式的消息字典序列。
            images: 图片 URL 或本地路径；非空时 prompt 以多模态形式送出。
            system_prompt: 系统提示词。
            stream: 是否流式返回。
            max_retries: 最大尝试次数（``1`` 表示只调用一次）。
            timeout: HTTP 超时（秒）。
            top_p: 核采样阈值。
            temperature: 采样温度。
            seed: 随机种子。
            **api_params: 其他 API 参数。

        Returns:
            :class:`ChainResult`。
        """
        chain_result = ChainResult(
            prompt_template=self._prompt_template,
            prompt_args=prompt_tmpl_args,
        )

        with Timer() as timer:
            try:
                api_params = self.obtain_api_params(
                    top_p=top_p,
                    temperature=temperature,
                    seed=seed,
                    **api_params,
                )
                chain_result.api_params = api_params

                prompt = self._prompt_template.generate_prompt(*prompt_tmpl_args)
                chain_result.prompt = prompt

                messages = self._build_messages(
                    history=history,
                    system_prompt=system_prompt,
                )
                messages.append(self._build_user_message(prompt, images))

                llm_result = self._retry_invoke(
                    messages,
                    max_retries=max_retries,
                    stream=stream,
                    timeout=timeout,
                    **api_params,
                )
                chain_result.update(llm_result)
            except Exception as exc:
                chain_result.error = exc

        chain_result.timecost = timer.elapsed
        return chain_result


class LLMChainWithValid(LLMChain):
    """多轮校验 LLM chain —— LLM 返回非法结果时，把错误作为新 prompt 反馈给模型重试。

    每轮的对话历史会累积：上一轮的 user 消息、assistant 回复、校验错误反馈
    依次追加，模型能看到完整的上下文。
    """

    def __call__(
        self,
        *prompt_tmpl_args,
        history: Optional[Sequence[ChatCompletionMessageParam]] = None,
        images: Optional[Sequence[str]] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        max_chat_turns: int = 1,
        max_retries: int = 1,
        timeout: float = 600.0,
        # generate params
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        **api_params,
    ) -> ChainResult:
        """调用 LLM；结果不合法时把校验错误反馈给模型让它自纠。

        Args:
            *prompt_tmpl_args: 传给 :meth:`PromptTemplate.generate_prompt` 的参数。
            history: 对话历史，OpenAI 格式的消息字典序列。
            images: 图片 URL 或本地路径；非空时 prompt 以多模态形式送出。
            system_prompt: 系统提示词。
            stream: 是否流式返回。
            max_chat_turns: 校验重试轮数上限（非法结果触发重轮的次数）。
            max_retries: 单轮内的最大尝试次数。
            timeout: HTTP 超时（秒）。
            top_p: 核采样阈值。
            temperature: 采样温度。
            seed: 随机种子。
            **api_params: 其他 API 参数。

        Returns:
            :class:`ChainResult`。
        """
        chain_result = ChainResult(
            prompt_template=self._prompt_template,
            prompt_args=prompt_tmpl_args,
        )

        try:
            api_params = self.obtain_api_params(
                top_p=top_p,
                temperature=temperature,
                seed=seed,
                **api_params,
            )
            chain_result.api_params = api_params
        except Exception as exc:
            chain_result.error = exc
            return chain_result

        messages = self._build_messages(
            history=history,
            system_prompt=system_prompt,
        )

        max_chat_turns = max(1, max_chat_turns)
        prompt = ""
        total_in = 0
        total_out = 0

        with Timer() as timer:
            for turn_idx in range(max_chat_turns):
                try:
                    if turn_idx == 0:
                        prompt = self._prompt_template.generate_prompt(*prompt_tmpl_args)
                        chain_result.prompt = prompt

                    messages.append(self._build_user_message(prompt, images))

                    llm_result = self._retry_invoke(
                        messages,
                        max_retries=max_retries,
                        stream=stream,
                        timeout=timeout,
                        **api_params,
                    )
                    total_in += llm_result.get("in_tokens", 0)
                    total_out += llm_result.get("out_tokens", 0)
                    chain_result.update(llm_result)

                    if llm_result.get("error") is not None:
                        break

                    self._prompt_template.valid(chain_result.result)
                    break
                except PromptTemplateValidError as exc:
                    if turn_idx == max_chat_turns - 1:
                        # 最后一轮仍不合法，直接记为失败
                        chain_result.error = exc
                        break
                    # 把上一轮的回复与校验错误回传，让模型自纠
                    messages.append(
                        {"role": "assistant", "content": chain_result.response}
                    )
                    prompt = f"**Valid Error Message as Follow:**\n{exc.message}"
                except Exception as exc:
                    # 包含 PromptTemplateGeneratingError 及其他确定性异常
                    chain_result.error = exc
                    break

        chain_result.timecost = timer.elapsed
        chain_result.in_tokens = total_in
        chain_result.out_tokens = total_out
        return chain_result
