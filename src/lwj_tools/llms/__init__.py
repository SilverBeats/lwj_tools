"""LLM 调用层，按职责拆分为 3 个子模块：

- :mod:`.client`  底层 client（chat / embedding / decision）+ :class:`LLMClient` 门面
- :mod:`.prompt`  :class:`PromptTemplate`，业务参数 → prompt → 解析 → 校验
- :mod:`.chain`   :class:`LLMChain` / :class:`LLMChainWithValid`，把上面两者串成工作流

典型用法::

    from lwj_tools.llms import APIConfig, LLMClientGroup, TaskType
    from lwj_tools.llms.chain import LLMChain
    from lwj_tools.llms.prompt import PromptTemplate

    group = LLMClientGroup(TaskType.CHAT, [
        APIConfig(model="gpt-4o-mini", api_base="https://api.openai.com/v1", api_key="sk-..."),
    ])
    chain = LLMChain(group, MyPrompt())
    result = chain("某个问题")
"""

from .chain import BaseLLMChain, ChainResult, LLMChain, LLMChainWithValid
from .client import (
    APIConfig,
    ChatClient,
    ClientBase,
    DecisionClient,
    EmbeddingClient,
    LLMClient,
    LLMClientGroup,
    LLMResponse,
    TaskType,
    build_decision_questions,
    encode_image,
    encrypted_api_key,
    tasks_num_manage,
)
from .prompt import PromptTemplate

__all__ = [
    # client
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
    # prompt
    "PromptTemplate",
    # chain
    "ChainResult",
    "BaseLLMChain",
    "LLMChain",
    "LLMChainWithValid",
]
