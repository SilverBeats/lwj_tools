"""LLM 客户端。

底层按任务类型拆成三个 client，:class:`LLMClient` 依据 :class:`TaskType` 动态创建
其中之一，并把调用转发过去：

============  ==========================  ========================================
任务类型        底层 client                 入参格式
============  ==========================  ========================================
chat          :class:`ChatClient`          OpenAI ``ChatCompletionMessageParam`` 列表
embedding     :class:`EmbeddingClient`     ``str`` 或 ``str`` 序列
decision      :class:`DecisionClient`      ``state`` + ``questions``
============  ==========================  ========================================

chat 与 embedding 沿用 OpenAI 的报文格式；decision 服务有自己的报文格式
（``state`` + 以问题名为键的 ``questions``），由 :class:`DecisionClient` 负责
组装与解析。

三个接口的路径分别由 ``chat_path`` / ``embedding_path`` / ``decision_path``
配置，默认与 OpenAI 官方接口一致；路径是相对 ``api_base`` 的，填完整 URL 时
该次调用会被指向到另一个域名。

Example:
    >>> from lwj_tools.llms.client import LLMClient, TaskType
    >>> client = LLMClient(
    ...     task_type=TaskType.CHAT,
    ...     model="gpt-4o-mini",
    ...     api_base="https://api.openai.com/v1",
    ...     api_key="sk-...",
    ... )
    >>> rst = client.response([{"role": "user", "content": "你好"}])
    >>> print(rst.response)
"""

from __future__ import annotations

import base64
import enum
import functools
import mimetypes
import sys
import threading
from abc import ABC
from array import array
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from openai import OpenAI, Stream
from openai.types import CreateEmbeddingResponse
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)

from ..common.ids import get_base64, is_url
from ..common.logging import get_logger
from ..common.pojo import DictLike
from ..date.timer import Timer
from ..errors import LLMClientError

LOGGER = get_logger(__name__)

#: 决策服务支持的问题类型
_QUESTION_TYPES = ("choice", "noul", "score")

#: 「是/否」类问题的别名，统一按 noul 处理
_BOOL_QUESTION_TYPES = ("predicate", "bool", "boolean", "null", "yesno", "yes_no")

__all__ = [
    "TaskType",
    "APIConfig",
    "LLMResponse",
    "ClientBase",
    "ChatClient",
    "EmbeddingClient",
    "DecisionClient",
    "LLMClient",
    "LLMClientGroup",
    "encrypted_api_key",
    "tasks_num_manage",
    "encode_image",
    "build_decision_questions",
]


class TaskType(str, enum.Enum):
    """LLM 任务类型，决定 :class:`LLMClient` 创建哪一个底层 client。

    Attributes:
        CHAT: 对话补全，底层为 :class:`ChatClient`。
        EMBEDDING: 文本嵌入，底层为 :class:`EmbeddingClient`。
        DECISION: 决策（SystemOne），底层为 :class:`DecisionClient`。
    """

    CHAT = "chat"
    EMBEDDING = "embedding"
    DECISION = "decision"

    def __str__(self) -> str:
        """返回任务类型的字符串值。"""
        return self.value


@dataclass
class LLMResponse(DictLike):
    """LLM 调用的统一响应结构。

    Attributes:
        response: 解析后的响应内容。chat 调用返回 ``str``；embedding 调用返回
            ``list[float]``（批量时为 ``list[list[float]]``）；decision 调用返回
            ``list[Answer]``，顺序与请求中的 ``questions`` 一致。
        details: 底层调用细节。chat 非流式为 OpenAI SDK 的
            :class:`ChatCompletion`，流式为已消费完的 chunk 列表；embedding 为
            :class:`CreateEmbeddingResponse`；decision 为服务端返回的原始
            响应体。
        timecost: 调用耗时（秒）。流式调用包含完整的流消费时间。
        in_tokens: 输入 token 数；不可用时为 ``0``。
        out_tokens: 输出 token 数；不可用时为 ``0``（embedding 恒为 ``0``）。
    """

    response: Optional[Any] = None
    details: Optional[Any] = None
    timecost: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0


@dataclass
class APIConfig(DictLike):
    """单个 LLM endpoint 的配置。

    Attributes:
        model: 模型名称。
        api_base: API 基础地址，如 ``https://api.openai.com/v1``。
        api_key: API 密钥。
        proxy: 代理配置；``None`` 表示不使用代理。
        mask_api_key_keep_size: API 密钥脱敏时保留的前缀长度。
        timeout: 单次请求超时（秒）。
        chat_path: 对话补全接口路径（相对 ``api_base``）。
        embedding_path: 嵌入接口路径（相对 ``api_base``）。
        decision_path: 决策接口路径（相对 ``api_base``）。

    三个 ``*_path`` 都是相对 :attr:`api_base` 的路径，默认与 OpenAI 官方接口
    一致；填完整 URL 时该次调用会被指向到另一个域名。
    """

    model: str
    api_base: str
    api_key: str = "xx"
    proxy: Optional[Union[Dict, Any]] = None
    mask_api_key_keep_size: int = 6
    timeout: float = 600.0
    chat_path: str = "chat/completions"
    embedding_path: str = "embeddings"
    decision_path: str = "decisions"


def encrypted_api_key(api_key: str, keep_size: int = 6) -> str:
    """对 API key 做掩码：保留前 ``keep_size`` 个字符，其余替换为 ``*``。

    Args:
        api_key: 原始 API key。
        keep_size: 保留前缀长度。

    Returns:
        掩码后的字符串；``api_key`` 为 ``None`` 时返回字面量 ``"None"``；
        长度 <= ``keep_size`` 时原样返回。
    """
    if api_key is None:
        return "None"

    if len(api_key) <= keep_size:
        return api_key

    return api_key[:keep_size] + "*" * (len(api_key) - keep_size)


def tasks_num_manage(func):
    """方法装饰器：维护 ``self._running_tasks_num`` 计数器。

    进入被装饰方法时 ``+1``，退出时 ``-1``（``finally`` 保证异常路径也计数）。
    计数器的读写由 ``self._tasks_lock`` 保护。

    Args:
        func: 被装饰的方法，首参应为 ``self``。

    Returns:
        包装后的方法。
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._tasks_lock:
            self._running_tasks_num += 1
        try:
            return func(self, *args, **kwargs)
        finally:
            with self._tasks_lock:
                self._running_tasks_num -= 1

    return wrapper


def encode_image(image: str) -> str:
    """把本地图像路径编码为 data URL；已经是 URL 时原样返回。

    MIME 类型按扩展名推断，未识别时退化为 ``image/jpeg``。

    Args:
        image: 本地图像路径或图像 URL。

    Returns:
        可直接用于 OpenAI ``image_url`` 字段的字符串。
    """
    if is_url(image):
        return image

    mime, _ = mimetypes.guess_type(image)
    if mime is None:
        mime = "image/jpeg"

    base64_image = get_base64(image)
    return f"data:{mime};base64,{base64_image}"


def build_decision_questions(spec: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """把问题描述整理成决策服务的 ``questions`` 映射。

    每个问题整理成 ``{"type": ..., "instructions": ..., "criteria": ...}``：

    - ``"choice"``：从若干选项中选一个，``criteria`` 为 ``{选项: 选项说明}``。
    - ``"score"``：在若干等级中打分，``criteria`` 为有序的等级名称列表。
    - ``"noul"``：是/否判断，不需要 ``criteria``。

    ``"predicate"`` / ``"bool"`` 等是/否类写法会统一成 ``"noul"``；缺省类型按
    ``"noul"`` 处理。

    Args:
        spec: 问题名到问题描述的映射，每项包含 ``type`` / ``instructions``
            / ``criteria``。

    Returns:
        问题名到问题描述的映射，可直接作为请求体的 ``questions`` 字段。

    Raises:
        ValueError: 问题缺少 ``instructions``，或 ``choice`` / ``score`` 类型
            缺少 ``criteria``。

    Example:
        >>> build_decision_questions({
        ...     "team": {
        ...         "type": "choice",
        ...         "instructions": "Which team should handle this ticket?",
        ...         "criteria": {"billing": "Payments, refunds and invoices"},
        ...     },
        ...     "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
        ... })
        {'team': {'type': 'choice', 'instructions': 'Which team should handle this ticket?',
                  'criteria': {'billing': 'Payments, refunds and invoices'}},
         'urgent': {'type': 'noul', 'instructions': 'Should this ticket be answered today?'}}
    """
    questions: Dict[str, Dict[str, Any]] = {}
    for name, item in spec.items():
        question_type = str(item.get("type", "noul")).lower()
        if question_type in _BOOL_QUESTION_TYPES:
            question_type = "noul"

        instructions = item.get("instructions", "")
        question: Dict[str, Any] = {
            "type": question_type,
            "instructions": instructions,
        }

        criteria = item.get("criteria")
        if question_type == "choice":
            question["criteria"] = _normalize_choice_criteria(criteria)
        elif question_type == "score":
            question["criteria"] = _normalize_score_criteria(criteria)

        questions[name] = question

    _validate_questions(questions)
    return questions


def _normalize_choice_criteria(criteria: Any) -> Dict[str, Any]:
    """把 ``choice`` 问题的选项整理成 ``{选项: 选项说明}``。

    Args:
        criteria: 选项描述，可以是 ``{选项: 说明}`` 映射或选项列表。

    Returns:
        选项到说明的映射。
    """
    if isinstance(criteria, Mapping):
        return dict(criteria)

    if isinstance(criteria, (list, tuple)):
        return {value: "" for value in criteria}

    return {}


def _normalize_score_criteria(criteria: Any) -> List[str]:
    """把 ``score`` 问题的等级整理成有序列表。

    Args:
        criteria: 等级描述，可以是等级列表或 ``{等级: 说明}`` 映射。

    Returns:
        等级名称列表。
    """
    if isinstance(criteria, Mapping):
        return list(criteria.keys())

    if isinstance(criteria, (list, tuple)):
        return [level if isinstance(level, str) else str(level) for level in criteria]

    return []


def _validate_questions(questions: Mapping[str, Mapping[str, Any]]) -> None:
    """校验问题描述是否符合决策服务的要求。

    Args:
        questions: 问题名到问题描述的映射。

    Raises:
        ValueError: 问题类型、必填字段或选项不符合要求。
    """
    for name, question in questions.items():
        question_type = question.get("type")
        if question_type not in _QUESTION_TYPES:
            raise ValueError(
                f"问题 {name!r} 的 type 非法：{question_type!r}，"
                f"应为 {sorted(_QUESTION_TYPES)} 之一"
            )

        if not question.get("instructions"):
            raise ValueError(f"问题 {name!r} 缺少 instructions")

        criteria = question.get("criteria")
        if question_type == "choice":
            if not criteria:
                raise ValueError(f"问题 {name!r}（type='choice'）缺少 criteria")
            if len(criteria) < 2:
                raise ValueError(f"问题 {name!r}（type='choice'）至少需要 2 个选项")
        elif question_type == "score":
            if not criteria:
                raise ValueError(f"问题 {name!r}（type='score'）缺少 criteria")


def _resolve_http_module():
    """返回 OpenAI SDK 实际使用的 http 库模块。

    ``openai>=3`` 使用 ``httpx2``，``openai 2.x`` 使用 ``httpx``；这里直接读取
    SDK 内部已导入的模块名，避免手工维护版本号判断。

    Returns:
        http 库模块（拥有 ``Client`` 属性）。
    """
    base_client = sys.modules.get("openai._base_client")
    for name in ("httpx2", "httpx"):
        module = getattr(base_client, name, None)
        if module is not None:
            return module

    import httpx

    return httpx


def _normalize_proxy(proxy: Any) -> Any:
    """把 dict 形式的代理配置规整成单个代理地址。

    ``httpx>=0.28`` 只接受单个代理地址；旧式的
    ``{"http://": ..., "https://": ...}`` 写法会取 ``https://``（缺失时取
    ``http://``），只有一项时直接取该项的值。

    Args:
        proxy: 代理配置。

    Returns:
        可直接交给 http 客户端的代理地址。
    """
    if not isinstance(proxy, Mapping):
        return proxy

    for key in ("https://", "http://", "all://"):
        if key in proxy:
            return proxy[key]

    if len(proxy) == 1:
        return next(iter(proxy.values()))

    return proxy


def _build_http_client(proxy: Any):
    """构造带代理的 http client，兼容不同 httpx 版本的 proxy 参数名。

    Args:
        proxy: 代理配置；``None`` 表示不使用代理，此时返回 ``None``，由 SDK 自己
            创建默认的 http client。

    Returns:
        构造好的 http client，或 ``None``。
    """
    if proxy is None:
        return None

    http = _resolve_http_module()
    proxy = _normalize_proxy(proxy)
    try:
        return http.Client(proxy=proxy)
    except TypeError:
        # httpx < 0.28 只支持 proxies= 参数
        return http.Client(proxies=proxy)


def _wrap_error(exc: Exception) -> LLMClientError:
    """把底层异常包装成 :class:`LLMClientError`，尽量保留结构化信息。

    尽力从原始异常提取 ``status_code`` / ``code`` / ``type`` 等属性，附加到
    :class:`LLMClientError` 实例上；若目标类不支持属性赋值则静默跳过。

    Args:
        exc: 原始异常。

    Returns:
        包装后的异常实例。
    """
    error = LLMClientError(message=str(exc))
    for attr in ("status_code", "code", "type"):
        value = getattr(exc, attr, None)
        if value is not None:
            try:
                setattr(error, attr, value)
            except AttributeError:
                pass
    return error


def _parse_embedding(value: Any) -> List[float]:
    """把嵌入向量规整成 float 列表。

    服务端在 ``encoding_format="base64"`` 时返回 base64 字符串，这里解码成
    float 列表，保证两种编码下结果一致。

    Args:
        value: 服务端返回的嵌入向量。

    Returns:
        float 列表。
    """
    if isinstance(value, str):
        return array("f", base64.b64decode(value)).tolist()

    return list(value)


class ClientBase(ABC):
    """所有 LLM client 的公共基础设施。

    子类按任务类型实现自己的调用方法：:meth:`ChatClient.response`、
    :meth:`EmbeddingClient.embedding`、:meth:`DecisionClient.decision`；其余
    方法在本类中给出「当前任务不支持」的实现。

    Attributes:
        task_type: 该 client 对应的任务类型。
        residual_credit: 预留信用额度字段（当前固定为 ``1``）。
    """

    task_type: ClassVar[TaskType]

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "xxx",
        proxy: Optional[Union[Dict, Any]] = None,
        mask_api_key_keep_size: int = 6,
        timeout: float = 600.0,
        chat_path: str = "chat/completions",
        embedding_path: str = "embeddings",
        decision_path: str = "decisions",
    ) -> None:
        """初始化客户端。

        Args:
            model: 模型名称。
            api_base: API 基础地址。
            api_key: API 密钥。
            proxy: 代理配置；``None`` 表示不使用代理。
            mask_api_key_keep_size: API 密钥脱敏保留前缀长度。
            timeout: 单次请求超时（秒）。
            chat_path: 对话补全接口路径（相对 ``api_base``）。
            embedding_path: 嵌入接口路径（相对 ``api_base``）。
            decision_path: 决策接口路径（相对 ``api_base``）。
        """
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._encrypted_api_key = encrypted_api_key(
            api_key,
            keep_size=mask_api_key_keep_size,
        )
        self._timeout = timeout
        self._chat_path = chat_path
        self._embedding_path = embedding_path
        self._decision_path = decision_path

        # 计数器与保护锁：初始化时即创建，避免惰性初始化竞态。
        self._running_tasks_num = 0
        self._tasks_lock = threading.Lock()

        # 显式持有 http client 引用：OpenAI SDK 不会接管外部传入的 http_client
        # 生命周期，需要我们自己关闭。
        self._http_client = _build_http_client(proxy)
        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            http_client=self._http_client,
            timeout=timeout,
        )

        self.residual_credit = 1
        LOGGER.debug(
            "创建 %s: model=%s api_base=%s",
            self.__class__.__name__,
            model,
            api_base,
        )

    @property
    def model(self) -> str:
        """模型名称。"""
        return self._model

    @property
    def api_base(self) -> str:
        """API 基础地址。"""
        return self._api_base

    @property
    def chat_path(self) -> str:
        """对话补全接口路径（相对 :attr:`api_base`）。"""
        return self._chat_path

    @property
    def embedding_path(self) -> str:
        """嵌入接口路径（相对 :attr:`api_base`）。"""
        return self._embedding_path

    @property
    def decision_path(self) -> str:
        """决策接口路径（相对 :attr:`api_base`）。"""
        return self._decision_path

    @property
    def encrypted_api_key(self) -> str:
        """:func:`encrypted_api_key` 脱敏后的 API key。"""
        return self._encrypted_api_key

    @property
    def running_tasks_num(self) -> int:
        """正在并发执行的任务数（由 :func:`tasks_num_manage` 维护）。"""
        return self._running_tasks_num

    @property
    def is_deprecated(self) -> bool:
        """是否被标记为不可用；由 :attr:`residual_credit <= 0` 决定。"""
        return self.residual_credit <= 0

    def close(self) -> None:
        """关闭底层 OpenAI 与 http 客户端，释放连接池。

        OpenAI SDK 不会接管外部传入的 ``http_client`` 生命周期，因此这里显式
        关闭我们持有的 http client。两个 ``close()`` 均为幂等操作。
        """
        try:
            self._client.close()
        finally:
            if self._http_client is not None:
                self._http_client.close()

    def _unsupported(self, method: str) -> TypeError:
        """构造「当前任务类型不支持该调用」的异常。

        Args:
            method: 被调用的方法名。

        Returns:
            待抛出的 :class:`TypeError`。
        """
        return TypeError(
            f"{self.__class__.__name__} 对应任务类型 {self.task_type.value}，"
            f"不支持 {method} 调用"
        )

    def _post(
        self,
        path: str,
        *,
        cast_to,
        body: Dict[str, Any],
        timeout: Optional[float] = None,
        stream: bool = False,
        stream_cls=None,
        **kwargs,
    ):
        """按指定路径发起 POST 请求。

        ``path`` 是相对 :attr:`api_base` 的路径（例如 ``"chat/completions"``）；
        填完整 URL 时该次调用会被指向到另一个域名。

        Args:
            path: 请求路径。
            cast_to: 响应模型类型。
            body: 请求体字段。
            timeout: 本次请求超时（秒）；``None`` 时用客户端默认值。
            stream: 是否流式返回。
            stream_cls: 流式响应的类型。
            **kwargs: 追加到请求体的字段；``extra_headers`` / ``extra_query`` /
                ``extra_body`` 会被单独处理。

        Returns:
            解析后的响应对象。
        """
        options: Dict[str, Any] = {
            "timeout": timeout if timeout is not None else self._timeout,
        }

        extra_headers = kwargs.pop("extra_headers", None)
        if extra_headers:
            options["headers"] = extra_headers

        extra_query = kwargs.pop("extra_query", None)
        if extra_query:
            options["params"] = extra_query

        extra_body = kwargs.pop("extra_body", None)
        if extra_body:
            body.update(extra_body)

        body.update(kwargs)

        return self._client.post(
            path,
            cast_to=cast_to,
            body=body,
            options=options,
            stream=stream,
            stream_cls=stream_cls,
        )

    def response(self, *args, **kwargs) -> LLMResponse:
        """chat 任务专用；其他任务类型会抛出 :class:`TypeError`。"""
        raise self._unsupported("response")

    def embedding(self, *args, **kwargs) -> LLMResponse:
        """embedding 任务专用；其他任务类型会抛出 :class:`TypeError`。"""
        raise self._unsupported("embedding")

    def decision(self, *args, **kwargs) -> LLMResponse:
        """decision 任务专用；其他任务类型会抛出 :class:`TypeError`。"""
        raise self._unsupported("decision")


class ChatClient(ClientBase):
    """对话补全（chat completions）客户端。

    消息格式直接使用 OpenAI 的 :data:`openai.types.chat.ChatCompletionMessageParam`，
    调用方自行组织 ``role`` / ``content``；多模态内容（图片、音频、文件）同样
    按 OpenAI 的 content part 格式传入。

    请求路径由 ``chat_path`` 配置，默认 ``chat/completions``。
    """

    task_type = TaskType.CHAT

    @tasks_num_manage
    def response(
        self,
        messages: Union[str, Sequence[ChatCompletionMessageParam]],
        *,
        stream: bool = False,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """发起一次对话补全。

        Args:
            messages: OpenAI 格式的消息列表；传字符串时等价于单条 user 消息。
            stream: 是否流式返回。``True`` 时会消费完整个流并返回拼接好的文本，
                :attr:`LLMResponse.details` 为已消费的 chunk 列表。
            timeout: 本次请求超时（秒）；``None`` 时用客户端默认值。
            **kwargs: 透传给 OpenAI ``chat.completions.create``。

        Returns:
            :class:`LLMResponse`。``response`` 为模型回复文本；``in_tokens`` /
            ``out_tokens`` 不可用时为 ``0``。

        Raises:
            LLMClientError: 调用失败时包装原始异常。

        Example:
            >>> client = ChatClient(model="gpt-4o-mini", api_base="https://api.openai.com/v1", api_key="sk-...")
            >>> rst = client.response([
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "你好"},
            ... ])
            >>> print(rst.response)
        """
        payload = self._normalize_messages(messages)
        body: Dict[str, Any] = {
            "model": self._model,
            "messages": payload,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}

        try:
            with Timer() as timer:
                rst = self._post(
                    self._chat_path,
                    cast_to=ChatCompletion,
                    body=body,
                    timeout=timeout if timeout is not None else self._timeout,
                    stream=stream,
                    stream_cls=Stream[ChatCompletionChunk],
                    **kwargs,
                )

                if stream:
                    answer, details, in_tokens, out_tokens = self._consume_stream(rst)
                else:
                    answer, details, in_tokens, out_tokens = self._parse_completion(rst)

            return LLMResponse(
                response=answer,
                details=details,
                timecost=timer.elapsed,
                in_tokens=in_tokens,
                out_tokens=out_tokens,
            )
        except Exception as e:
            raise _wrap_error(e) from e

    @staticmethod
    def _normalize_messages(
        messages: Union[str, Sequence[ChatCompletionMessageParam]],
    ) -> List[ChatCompletionMessageParam]:
        """把入参规整成 OpenAI 的消息列表。

        Args:
            messages: 消息列表或纯文本。

        Returns:
            OpenAI 格式的消息列表。
        """
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        return list(messages)

    @staticmethod
    def _parse_completion(
        completion: ChatCompletion,
    ) -> Tuple[Optional[str], ChatCompletion, int, int]:
        """解析非流式的补全结果。

        Args:
            completion: OpenAI 返回的补全对象。

        Returns:
            ``(回复文本, 原始对象, 输入 token, 输出 token)``。
        """
        usage = getattr(completion, "usage", None)
        in_tokens = getattr(usage, "prompt_tokens", 0) or 0
        out_tokens = getattr(usage, "completion_tokens", 0) or 0
        choices = completion.choices
        answer = choices[0].message.content if choices else None
        return answer, completion, in_tokens, out_tokens

    @staticmethod
    def _consume_stream(stream: Any) -> Tuple[str, List[Any], int, int]:
        """消费完整个流式响应。

        Args:
            stream: OpenAI 返回的流对象。

        Returns:
            ``(拼接后的文本, chunk 列表, 输入 token, 输出 token)``。
        """
        chunks: List[Any] = []
        parts: List[str] = []
        in_tokens = 0
        out_tokens = 0

        for chunk in stream:
            chunks.append(chunk)

            usage = getattr(chunk, "usage", None)
            if usage is not None:
                in_tokens = getattr(usage, "prompt_tokens", 0) or 0
                out_tokens = getattr(usage, "completion_tokens", 0) or 0
                continue

            if not chunk.choices:
                # 只带 usage 或只带 tool_call 的 chunk 没有 choices
                continue

            delta = chunk.choices[0].delta.content
            parts.append("" if delta is None else delta)

        return "".join(parts), chunks, in_tokens, out_tokens


class EmbeddingClient(ClientBase):
    """文本嵌入（embeddings）客户端。

    请求路径由 ``embedding_path`` 配置，默认 ``embeddings``。
    """

    task_type = TaskType.EMBEDDING

    @tasks_num_manage
    def embedding(
        self,
        input: Union[str, Sequence[str]],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """计算文本的嵌入向量。

        Args:
            input: 待嵌入的文本；传字符串序列时批量计算。
            timeout: 本次请求超时（秒）；``None`` 时用客户端默认值。
            **kwargs: 透传给 OpenAI ``embeddings.create``。

        Returns:
            :class:`LLMResponse`。``response`` 在单条输入时为 ``list[float]``，
            批量输入时为 ``list[list[float]]``；``out_tokens`` 恒为 ``0``。

        Raises:
            LLMClientError: 调用失败时包装原始异常。

        Example:
            >>> client = EmbeddingClient(model="text-embedding-3-small", api_base="https://api.openai.com/v1", api_key="sk-...")
            >>> rst = client.embedding("你好")
            >>> len(rst.response)
            1536
        """
        try:
            with Timer() as timer:
                rst = self._post(
                    self._embedding_path,
                    cast_to=CreateEmbeddingResponse,
                    body={"model": self._model, "input": input},
                    timeout=timeout if timeout is not None else self._timeout,
                    **kwargs,
                )

            vectors = [_parse_embedding(item.embedding) for item in rst.data]
            answer = vectors[0] if isinstance(input, str) else vectors
            usage = getattr(rst, "usage", None)
            in_tokens = getattr(usage, "prompt_tokens", 0) or 0

            return LLMResponse(
                response=answer,
                details=rst,
                timecost=timer.elapsed,
                in_tokens=in_tokens,
                out_tokens=0,
            )
        except Exception as e:
            raise _wrap_error(e) from e


class DecisionClient(ClientBase):
    """决策（SystemOne）客户端。

    请求体形如：

    .. code-block:: json

        {
            "model": "qwen2.5-7B",
            "state": {"ticket": "I was charged twice for order #4411."},
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this ticket?",
                    "criteria": {"billing": "Payments, refunds and invoices"}
                },
                "urgent": {"type": "noul", "instructions": "Should this be answered today?"}
            }
        }

    响应体形如：

    .. code-block:: json

        {
            "answers": {
                "team": {"type": "choice", "choice": "billing", "confidence": 0.99},
                "urgent": {"type": "noul", "noul": 0.84}
            },
            "usage": {"input_tokens": 164, "output_tokens": 0}
        }

    ``questions`` 用「问题名 -> 问题描述」的映射描述，请求前会被
    :func:`build_decision_questions` 整理成服务要求的结构。

    请求路径由 ``decision_path`` 配置，默认 ``decisions``。
    """

    task_type = TaskType.DECISION

    @tasks_num_manage
    def decision(
        self,
        state: Union[str, Mapping[str, Any]],
        questions: Mapping[str, Mapping[str, Any]],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """发起一次决策请求。

        Args:
            state: 待判断的状态，可以是文本或键值字典。
            questions: 问题描述，「问题名 -> 问题描述」的映射；每项包含 ``type``
                （``choice`` / ``noul`` / ``score``）、``instructions``，以及
                ``choice`` / ``score`` 类型所需的 ``criteria``。
            timeout: 本次请求超时（秒）；``None`` 时用客户端默认值。
            **kwargs: 追加到请求体的字段。

        Returns:
            :class:`LLMResponse`。``response`` 为「问题名 -> 答案」的映射，
            ``details`` 为完整的响应体。

        Raises:
            LLMClientError: 调用失败时包装原始异常。
            ValueError: 问题描述不合法。

        Example:
            >>> client = DecisionClient(model="qwen2.5-7B", api_base="http://127.0.0.1:8100/v1", api_key="xxx")
            >>> rst = client.decision(
            ...     "I was charged twice for order #4411.",
            ...     {
            ...         "team": {
            ...             "type": "choice",
            ...             "instructions": "Which team should handle this ticket?",
            ...             "criteria": {"billing": "Payments, refunds and invoices"},
            ...         },
            ...     },
            ... )
            >>> rst.response["team"]["choice"]
            'billing'
        """
        state_key = 'state' if 'starlux' in self.model.lower() else 'input'
        body: Dict[str, Any] = {
            "model": self._model,
            state_key: state,
            "questions": build_decision_questions(questions),
        }
        body.update(kwargs)

        try:
            with Timer() as timer:
                rst = self._post(
                    self._decision_path,
                    cast_to=Dict[str, Any],
                    body=body,
                    timeout=timeout,
                )

            answers = self._parse_answers(rst)
            usage = rst.get("usage") or {}

            return LLMResponse(
                response=answers,
                details=rst,
                timecost=timer.elapsed,
                in_tokens=usage.get("input_tokens", 0) or 0,
                out_tokens=usage.get("output_tokens", 0) or 0,
            )
        except Exception as e:
            raise _wrap_error(e) from e

    @staticmethod
    def _parse_answers(rst: Mapping[str, Any]) -> Dict[str, Any]:
        """从响应体中取出「问题名 -> 答案」的映射。

        服务返回的 ``answers`` 是以问题名为键的字典；若服务返回的是列表形式
        （元素带 ``name``），则转成字典。

        Args:
            rst: 响应体。

        Returns:
            问题名到答案的映射。
        """
        answers = rst.get("answers")
        if isinstance(answers, Mapping):
            return dict(answers)

        if isinstance(answers, (list, tuple)):
            return {
                str(answer.get("name", index)): answer
                for index, answer in enumerate(answers)
            }

        return {}


_CLIENT_CLASSES = {
    TaskType.CHAT: ChatClient,
    TaskType.EMBEDDING: EmbeddingClient,
    TaskType.DECISION: DecisionClient,
}


class LLMClient:
    """统一的 LLM 客户端入口。

    构造时给出 :class:`TaskType`，内部创建对应的底层 client 并转发调用：

    .. code-block:: python

        client = LLMClient(TaskType.CHAT, model=..., api_base=..., api_key=...)
        client.response([{"role": "user", "content": "你好"}])

    与任务类型不符的调用会抛出 :class:`TypeError`。三个接口的路径可用
    ``chat_path`` / ``embedding_path`` / ``decision_path`` 分别配置。
    """

    def __init__(
        self,
        task_type: Union[TaskType, str],
        model: str,
        api_base: str,
        api_key: str = "xxx",
        proxy: Optional[Union[Dict, Any]] = None,
        mask_api_key_keep_size: int = 6,
        timeout: float = 600.0,
        chat_path: str = "chat/completions",
        embedding_path: str = "embeddings",
        decision_path: str = "decisions",
    ) -> None:
        """初始化客户端。

        Args:
            task_type: 任务类型，决定创建哪个底层 client。
            model: 模型名称。
            api_base: API 基础地址。
            api_key: API 密钥。
            proxy: 代理配置；``None`` 表示不使用代理。
            mask_api_key_keep_size: API 密钥脱敏保留前缀长度。
            timeout: 单次请求超时（秒）。
            chat_path: 对话补全接口路径（相对 ``api_base``）。
            embedding_path: 嵌入接口路径（相对 ``api_base``）。
            decision_path: 决策接口路径（相对 ``api_base``）。

        Raises:
            ValueError: ``task_type`` 不是合法的任务类型。
        """
        self._task_type = TaskType(task_type)
        client_class = _CLIENT_CLASSES[self._task_type]
        self._client: ClientBase = client_class(
            model=model,
            api_base=api_base,
            api_key=api_key,
            proxy=proxy,
            mask_api_key_keep_size=mask_api_key_keep_size,
            timeout=timeout,
            chat_path=chat_path,
            embedding_path=embedding_path,
            decision_path=decision_path,
        )

    @property
    def task_type(self) -> TaskType:
        """当前 client 的任务类型。"""
        return self._task_type

    @property
    def client(self) -> ClientBase:
        """底层 client 实例。"""
        return self._client

    @property
    def model(self) -> str:
        """模型名称。"""
        return self._client.model

    @property
    def api_base(self) -> str:
        """API 基础地址。"""
        return self._client.api_base

    @property
    def chat_path(self) -> str:
        """对话补全接口路径（相对 :attr:`api_base`）。"""
        return self._client.chat_path

    @property
    def embedding_path(self) -> str:
        """嵌入接口路径（相对 :attr:`api_base`）。"""
        return self._client.embedding_path

    @property
    def decision_path(self) -> str:
        """决策接口路径（相对 :attr:`api_base`）。"""
        return self._client.decision_path

    @property
    def encrypted_api_key(self) -> str:
        """脱敏后的 API key。"""
        return self._client.encrypted_api_key

    @property
    def running_tasks_num(self) -> int:
        """正在并发执行的任务数。"""
        return self._client.running_tasks_num

    @property
    def is_deprecated(self) -> bool:
        """是否被标记为不可用。"""
        return self._client.is_deprecated

    def response(self, *args, **kwargs) -> LLMResponse:
        """chat 任务的对话补全调用，转发给 :meth:`ChatClient.response`。

        Raises:
            TypeError: 当前 client 不是 chat 任务。
        """
        self._check_task(TaskType.CHAT)
        return self._client.response(*args, **kwargs)

    def embedding(self, *args, **kwargs) -> LLMResponse:
        """embedding 任务的嵌入调用，转发给 :meth:`EmbeddingClient.embedding`。

        Raises:
            TypeError: 当前 client 不是 embedding 任务。
        """
        self._check_task(TaskType.EMBEDDING)
        return self._client.embedding(*args, **kwargs)

    def decision(self, *args, **kwargs) -> LLMResponse:
        """decision 任务的决策调用，转发给 :meth:`DecisionClient.decision`。

        Raises:
            TypeError: 当前 client 不是 decision 任务。
        """
        self._check_task(TaskType.DECISION)
        return self._client.decision(*args, **kwargs)

    def close(self) -> None:
        """关闭底层客户端，释放连接池。"""
        self._client.close()

    def _check_task(self, expected: TaskType) -> None:
        """校验当前任务类型是否与调用匹配。

        Args:
            expected: 期望的任务类型。

        Raises:
            TypeError: 任务类型不匹配。
        """
        if self._task_type is not expected:
            raise TypeError(
                f"当前 client 的任务类型是 {self._task_type.value}，"
                f"不支持 {expected.value} 调用"
            )


class LLMClientGroup:
    """同类型多个 endpoint 的客户端容器。

    按 ``api_configs`` 顺序构造 :attr:`clients`，索引一一对应；
    :meth:`find_available_client` 选择 running 任务数最少的非 deprecated client。

    Example:
        >>> group = LLMClientGroup(TaskType.CHAT, [
        ...     APIConfig(model="gpt-4o-mini", api_base="https://api.openai.com/v1", api_key="sk-..."),
        ...     APIConfig(model="gpt-4o", api_base="https://api.openai.com/v1", api_key="sk-..."),
        ... ])
        >>> client = group.find_available_client()
    """

    def __init__(
        self,
        task_type: Union[TaskType, str],
        api_configs: Sequence[APIConfig],
    ) -> None:
        """顺序构造 :attr:`clients`，索引与 ``api_configs`` 一一对应。

        Args:
            task_type: 任务类型，决定创建哪一类 client。
            api_configs: :class:`APIConfig` 列表。
        """
        self._task_type = TaskType(task_type)
        self.clients = [
            LLMClient(self._task_type, **api_config.to_dict())
            for api_config in api_configs
        ]

    @property
    def task_type(self) -> TaskType:
        """当前客户端组的任务类型。"""
        return self._task_type

    @property
    def available_clients(self) -> List[LLMClient]:
        """过滤掉 :attr:`LLMClient.is_deprecated` 为 True 的 client。"""
        return [client for client in self.clients if not client.is_deprecated]

    def find_available_client(
        self,
        ignored_clients: Optional[Sequence[LLMClient]] = None,
    ) -> Optional[LLMClient]:
        """从可用 client 中挑 ``running_tasks_num`` 最小的一个。

        Args:
            ignored_clients: 本轮调用中需要跳过的 client（一般用于失败重试时屏蔽
                坏 client）。

        Returns:
            选中的 :class:`LLMClient`；没有可用 client 时返回 ``None``。
        """
        if ignored_clients is None:
            ignored_clients = []

        candidates = [
            client
            for client in self.available_clients
            if client not in ignored_clients
        ]

        if not candidates:
            return None

        return min(candidates, key=lambda client: client.running_tasks_num)

    def close(self) -> None:
        """关闭组内所有 client。"""
        for client in self.clients:
            client.close()
