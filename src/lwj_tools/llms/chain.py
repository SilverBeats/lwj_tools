#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import threading
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generator, List, Optional, Union

from .client import LLMClient, LLMClientGroup
from .prompt import PromptTemplate
from ..date.timer import Timer
from ..errors import (
    BaseError,
    LLMClientError,
    PromptTemplateGeneratingError,
    PromptTemplateParsingError,
    SUCCEED_CODE,
)

warnings.simplefilter(action="once", category=UserWarning)


@dataclass
class ChainResult:
    api_params: Optional[dict] = None
    prompt_args: Any = None
    prompt: Optional[str] = None
    prompt_template: Optional[PromptTemplate] = None
    response: Union[str, Generator, Any] = None
    result: Any = None
    reasoning_content: Optional[str] = None
    error: Optional[BaseError] = None
    status_code: Optional[str] = None
    timecost: float = 0
    in_tokens: int = 0
    out_tokens: int = 0
    client_key: Optional[str] = None

    def to_dict(self) -> dict:
        data_dict = asdict(self)

        if self.prompt_template:
            data_dict["prompt_template"] = self.prompt_template.__class__.__name__

        if isinstance(data_dict["response"], Generator):
            data_dict["response"] = list(data_dict["response"])

        if self.error:
            data_dict["error"] = self.error

        return data_dict


class LLMChain:
    def __init__(
        self,
        client_group: LLMClientGroup,
        prompt_template: PromptTemplate,
        **api_params,
    ):
        """初始化LLMChain实例

        Args:
            client_group: 一组LLMClient实例
            prompt_template: PromptTemplate实例
            **api_params: 所有调用请求的默认api参数
        """
        self._client_group = client_group
        self._prompt_template = prompt_template
        self._lock = threading.Lock()
        self._default_api_params = api_params

    def _choose_client(self, ignored_clients) -> LLMClient:
        """选择一个可用的LLMClient实例

        Args:
            ignored_clients: 忽略的客户端
        """
        with self._lock:
            client = self._client_group.find_available_client(
                ignored_clients=ignored_clients,
            )
        return client

    def obtain_api_params(
        self, top_p: float, temperature: float, seed: int, **api_params
    ) -> dict:
        """获取调用请求的api参数

        Args:
            top_p: 采样参数
            temperature: 采样参数
            seed: 随机种子
            **api_params: 其他api参数
        """
        new_api_params = {**self._default_api_params}

        extra_body = {}
        if "extra_body" in api_params:
            extra_body = api_params.pop("extra_body")

        if "top_k" in api_params:
            # OpenAI do not support `top_k` directly
            # put `top_k` into `extra_body` to use it
            extra_body["top_k"] = api_params["top_k"]

        if temperature is not None:
            new_api_params["temperature"] = temperature
        if top_p is not None:
            new_api_params["top_p"] = top_p
        if seed is not None:
            new_api_params["seed"] = seed
        if len(extra_body) > 0:
            new_api_params["extra_body"] = extra_body
        new_api_params.update(api_params)

        return new_api_params

    def _invoke(
        self,
        *prompt_tmpl_args,
        history: Optional[List[Dict[str, str]]] = None,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        max_retries: int = 1,
        timeout: float = 600,
        # generate params
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        **api_params,
    ) -> ChainResult:
        """调用LLM模型

        Args:
            *prompt_tmpl_args: PromptTemplate的参数
            history (Optional[List[Dict[str, str]]], optional): 对话历史
            images (Optional[List[str]], optional): 图像路径
            system_prompt (Optional[str], optional): 系统提示
            stream (bool, optional): 是否流式返回
            max_retries (int, optional): 最大重试次数
            timeout (float, optional): 超时时间
            top_p (Optional[float], optional): 采样参数
            temperature (Optional[float], optional): 采样参数
            seed (Optional[int], optional): 随机种子
            **api_params: 其他api参数

        Returns:
            ChainResult: 调用结果
        """
        chain_result = ChainResult(
            prompt_template=self._prompt_template,
            prompt_args=prompt_tmpl_args,
        )
        api_params = self.obtain_api_params(
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            **api_params,
        )

        chain_result.api_params = api_params
        ignored_clients = []
        client = None

        for _ in range(max_retries):
            try:
                prompt = self._prompt_template.generate_prompt(*prompt_tmpl_args)
                chain_result.prompt = prompt

                client = self._choose_client(ignored_clients)
                if client is None:
                    chain_result.error = LLMClientError("No available LLMClient")
                    break

                chain_result.client_key = client.encrypted_api_key

                llm_response = client.response(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history=history,
                    images=images,
                    stream=stream,
                    timeout=timeout,
                    # generate params
                    **api_params,
                )
                chain_result.response = llm_response.response
                chain_result.in_tokens = llm_response.in_tokens
                chain_result.out_tokens = llm_response.out_tokens
                chain_result.timecost = llm_response.timecost

                try:
                    reasoning_content = (
                        llm_response.details.choices[0].message.reasoning_content
                        if hasattr(
                            llm_response.details.choices[0].message,
                            "reasoning_content",
                        )
                        else None
                    )
                except:
                    reasoning_content = None
                chain_result.reasoning_content = reasoning_content

                parse_result = self._prompt_template.parse(llm_response.response)
                chain_result.result = parse_result
                chain_result.status_code = SUCCEED_CODE
                break
            except (
                    PromptTemplateGeneratingError,
                    PromptTemplateParsingError,
            ) as e:
                chain_result.error = e
            except LLMClientError as e:
                chain_result.error = e
                if client is not None:
                    ignored_clients.append(client)
            except Exception as e:
                chain_result.error = BaseError(str(e))

        return chain_result

    def invoke(
        self,
        *prompt_tmpl_args,
        history: Optional[List[Dict[str, str]]] = None,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        timeout: float = 600,
        max_retries: int = 1,
        # generate params
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        **api_params,
    ) -> ChainResult:
        with Timer() as t:
            chain_result = self._invoke(
                *prompt_tmpl_args,
                history=history,
                images=images,
                system_prompt=system_prompt,
                stream=stream,
                max_retries=max_retries,
                timeout=timeout,
                top_p=top_p,
                temperature=temperature,
                seed=seed,
                **api_params,
            )
        chain_result.timecost = t.elapsed
        return chain_result

    def __call__(
        self,
        *prompt_tmpl_args,
        history: Optional[List[Dict[str, str]]] = None,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        timeout: float = 600,
        max_retries: int = 1,
        # generate params
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        **api_params,
    ):
        """调用LLM模型

        Args:
            *prompt_tmpl_args: PromptTemplate的参数
            history: 聊天记录
            images: 图像路径
            system_prompt: 系统提示
            stream 是否流式返回
            max_retries: 最大重试次数
            timeout: 超时时间
            top_p: 采样参数
            temperature: 采样参数
            seed: 随机种子
            **api_params: 其他api参数

        Returns:
            ChainResult: 调用结果
        """
        return self.invoke(
            *prompt_tmpl_args,
            history=history,
            images=images,
            system_prompt=system_prompt,
            stream=stream,
            timeout=timeout,
            max_retries=max_retries,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            **api_params,
        )
