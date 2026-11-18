"""Kappa 一致性系数计算。

包含三个公开入口：

- :func:`cohen_kappa` —— 2 个评分者，Cohen's Kappa。
- :func:`fleiss_kappa` —— 任意数量（≥2）评分者，Fleiss' Kappa。
- :func:`kappa` —— 公共入口，按评分者数自动分发：
  ``2 → Cohen``，``≥3 → Fleiss``。

以及一个从计数矩阵出发的 Fleiss 入口 :func:`fleiss_kappa_from_counts`。

两个系数的核心区别在期望一致度 ``P̄_e`` 的算法：

- Cohen 保留两个评分者各自的边际，``P̄_e = Σ_j p_1j · p_2j``。
- Fleiss 使用所有评分者的合并边际，``P̄_e = Σ_j p_j²``。

因此当 ``m=2`` 但两个评分者类别偏好不同时，两者结果不相等。
需要严格对齐文献惯例时：2 人用 Cohen，3 人及以上用 Fleiss。

References:
    Cohen, J. (1960). A coefficient of agreement for nominal scales.
        Educational and Psychological Measurement, 20(1), 37-46.
    Fleiss, J. L. (1971). Measuring nominal scale agreement among many
        raters. Psychological Bulletin, 76(5), 378-382.
    Landis, J. R., & Koch, G. G. (1977). The measurement of observer
        agreement for categorical data. Biometrics, 33(1), 159-174.

Example:
    >>> from lwj_tools.stats.kappa import kappa
    >>> ratings = [[1, 1], [2, 2], [1, 1]]      # 2 个评分者 → Cohen
    >>> kappa(ratings)
    1.0
    >>> ratings = [[1, 1, 1], [2, 2, 2], [1, 2, 3]]   # 3 个评分者 → Fleiss
    >>> round(kappa(ratings), 4)
    0.4375
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

# 判定 P̄_e 分母是否退化的阈值。
_EPS = 1e-12


@dataclass
class KappaResult:
    """Kappa 计算结果（Cohen / Fleiss 共用结构）。

    Attributes:
        kappa: Kappa 系数。范围通常 ``[-1, 1]``，``1`` 表示完全一致。
            当期望一致度 ``P̄_e → 1`` 时公式无定义，此字段为 ``NaN``。
        p_observed: 观测一致度 ``P̄``。
        p_expected: 随机期望一致度 ``P̄_e``。
        n_subjects: 对象数 ``N``。
        n_raters: 评分者数 ``m``。
        n_categories: 类别数 ``k``。
        method: ``"cohen"`` 或 ``"fleiss"``。
        category_proportions: Fleiss 的合并边际 ``p_j``，形状 ``(k,)``；
            Cohen 下为 ``None``。
        rater_proportions: Cohen 的每个评分者边际，形状 ``(2, k)``；
            Fleiss 下为 ``None``。
        categories: 类别标签列表；未知时为 ``None``。
    """

    kappa: float
    p_observed: float
    p_expected: float
    n_subjects: int
    n_raters: int
    n_categories: int
    method: str
    category_proportions: Optional[np.ndarray] = None
    rater_proportions: Optional[np.ndarray] = None
    categories: Optional[List[Any]] = None

    @property
    def interpretation(self) -> str:
        """按 Landis & Koch (1977) 标准给出定性解释。"""
        return interpret_kappa(self.kappa)


# 向后兼容上一版的命名。
FleissKappaResult = KappaResult


def interpret_kappa(kappa: float) -> str:
    """按 Landis & Koch (1977) 的标准给出 Kappa 的定性解释。

    区间：``poor`` / ``slight`` / ``fair`` / ``moderate`` / ``substantial``
    / ``almost perfect``。``NaN`` 返回 ``"undefined"``。
    """
    if np.isnan(kappa):
        return "undefined"
    if kappa < 0:
        return "poor"
    if kappa <= 0.20:
        return "slight"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost perfect"


# ---------------------------------------------------------------------- #
# 内部工具
# ---------------------------------------------------------------------- #


def _encode(
    ratings: Any,
    labels: Optional[Sequence[Any]],
) -> Tuple[np.ndarray, List[Any]]:
    """把评分矩阵编码为整数索引矩阵。

    Args:
        ratings: ``(n_subjects, n_raters)`` 的二维数组，元素为类别标签。
        labels: 类别全集；``None`` 表示从数据推断。

    Returns:
        ``(encoded, categories)``。``encoded`` 形状 ``(n_subjects, n_raters)``，
        dtype ``int64``，值域 ``[0, k)``。

    Raises:
        ValueError: 维度不对、评分者不足、出现未知标签、类别数不足。
    """
    arr = np.asarray(ratings)
    if arr.ndim != 2:
        raise ValueError(
            f"ratings must be a 2D array of shape (n_subjects, n_raters), "
            f"got {arr.ndim}D"
        )

    n_subjects, n_raters = arr.shape
    if n_subjects < 1:
        raise ValueError("ratings must contain at least one subject")
    if n_raters < 2:
        raise ValueError(f"at least 2 raters required, got {n_raters}")

    if labels is None:
        categories_arr, encoded = np.unique(arr, return_inverse=True)
        categories: List[Any] = categories_arr.tolist()
    else:
        categories = list(labels)
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        flat: List[int] = []
        for value in arr.ravel().tolist():
            if value not in cat_to_idx:
                raise ValueError(f"unknown label {value!r} found in ratings")
            flat.append(cat_to_idx[value])
        encoded = np.asarray(flat, dtype=np.int64)

    # NumPy < 2.0 返回一维 inverse，>= 2.0 保留输入形状；统一 reshape。
    encoded = np.asarray(encoded, dtype=np.int64).reshape(n_subjects, n_raters)

    if len(categories) < 2:
        raise ValueError(f"at least 2 categories required, got {len(categories)}")

    return encoded, categories


def _counts_from_encoded(encoded: np.ndarray, n_categories: int) -> np.ndarray:
    """从整数索引矩阵生成计数矩阵 ``(n_subjects, k)``。"""
    n_subjects = encoded.shape[0]
    counts = np.zeros((n_subjects, n_categories), dtype=np.int64)
    for i in range(n_subjects):
        counts[i] = np.bincount(encoded[i], minlength=n_categories)
    return counts


def _validate_counts(counts: np.ndarray) -> int:
    """校验计数矩阵，返回每个对象的评分者数 ``m``。"""
    if counts.ndim != 2:
        raise ValueError(
            f"counts must be a 2D array of shape (n_subjects, n_categories), "
            f"got {counts.ndim}D"
        )
    if counts.size == 0:
        raise ValueError("counts must not be empty")
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError(f"counts must have integer dtype, got {counts.dtype}")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")

    n_subjects, n_categories = counts.shape
    if n_subjects < 1:
        raise ValueError("counts must contain at least one subject")
    if n_categories < 2:
        raise ValueError(f"at least 2 categories required, got {n_categories}")

    m_per_subject = counts.sum(axis=1)
    if m_per_subject[0] < 2:
        raise ValueError(
            f"at least 2 raters per subject required, got {m_per_subject[0]}"
        )
    if not np.all(m_per_subject == m_per_subject[0]):
        raise ValueError(
            "all subjects must have the same number of ratings; "
            f"got per-subject totals {m_per_subject.tolist()}"
        )
    return int(m_per_subject[0])


def _safe_kappa(p_observed: float, p_expected: float) -> float:
    """``(P̄ − P̄_e) / (1 − P̄_e)``，分母退化时返回 NaN。"""
    denom = 1.0 - p_expected
    if denom <= _EPS:
        return float("nan")
    return (p_observed - p_expected) / denom


# ---------------------------------------------------------------------- #
# Cohen's Kappa
# ---------------------------------------------------------------------- #


def cohen_kappa(
    ratings: Any,
    *,
    labels: Optional[Sequence[Any]] = None,
    return_details: bool = False,
):
    """从评分矩阵计算 Cohen's Kappa（要求恰好 2 个评分者）。

    Args:
        ratings: ``(n_subjects, 2)`` 的二维数组。第 ``i`` 行是两位评分者对
            第 ``i`` 个对象的评分。
        labels: 可选的类别全集。若不提供，从数据推断。
        return_details: 为 ``True`` 时返回 :class:`KappaResult`；否则只返回
            ``kappa`` 浮点数。

    Returns:
        Kappa 系数，或 :class:`KappaResult`（``method="cohen"``）。

    Raises:
        ValueError: 输入不符合要求，或评分者数不为 2。
    """
    encoded, categories = _encode(ratings, labels)
    if encoded.shape[1] != 2:
        raise ValueError(
            f"Cohen's kappa requires exactly 2 raters, "
            f"got {encoded.shape[1]}; use fleiss_kappa for 3+ raters"
        )

    n_subjects = encoded.shape[0]
    n_categories = len(categories)

    # 混淆矩阵，行 = 评分者 1，列 = 评分者 2
    confusion = np.zeros((n_categories, n_categories), dtype=np.int64)
    for a, b in zip(encoded[:, 0].tolist(), encoded[:, 1].tolist()):
        confusion[a, b] += 1

    p_observed = float(np.trace(confusion)) / n_subjects
    p_rater1 = confusion.sum(axis=1) / n_subjects  # 评分者 1 边际
    p_rater2 = confusion.sum(axis=0) / n_subjects  # 评分者 2 边际
    p_expected = float(np.dot(p_rater1, p_rater2))

    kappa_val = _safe_kappa(p_observed, p_expected)

    if not return_details:
        return kappa_val

    return KappaResult(
        kappa=kappa_val,
        p_observed=p_observed,
        p_expected=p_expected,
        n_subjects=n_subjects,
        n_raters=2,
        n_categories=n_categories,
        method="cohen",
        rater_proportions=np.stack([p_rater1, p_rater2]),
        categories=categories,
    )


# ---------------------------------------------------------------------- #
# Fleiss' Kappa
# ---------------------------------------------------------------------- #


def fleiss_kappa_from_counts(
    counts: Any,
    *,
    categories: Optional[Sequence[Any]] = None,
    return_details: bool = False,
):
    """从计数矩阵计算 Fleiss' Kappa。

    Args:
        counts: ``(n_subjects, n_categories)`` 的整数矩阵。
            ``counts[i, j]`` 表示第 ``i`` 个对象被分到第 ``j`` 个类别的评分者数。
            每行和必须相同。
        categories: 可选的类别标签列表，仅用于结果展示。
        return_details: 为 ``True`` 时返回 :class:`KappaResult`；否则只返回
            ``kappa`` 浮点数。

    Returns:
        Kappa 系数，或 :class:`KappaResult`（``method="fleiss"``）。
    """
    counts_arr = np.asarray(counts)
    m = _validate_counts(counts_arr)

    n_subjects, n_categories = counts_arr.shape

    p_i = (np.sum(counts_arr**2, axis=1) - m) / (m * (m - 1))
    p_observed = float(p_i.mean())

    p_j = counts_arr.sum(axis=0) / (n_subjects * m)
    p_expected = float(np.sum(p_j**2))

    kappa_val = _safe_kappa(p_observed, p_expected)

    if not return_details:
        return kappa_val

    return KappaResult(
        kappa=kappa_val,
        p_observed=p_observed,
        p_expected=p_expected,
        n_subjects=n_subjects,
        n_raters=m,
        n_categories=n_categories,
        method="fleiss",
        category_proportions=p_j,
        categories=list(categories) if categories is not None else None,
    )


def fleiss_kappa(
    ratings: Any,
    *,
    labels: Optional[Sequence[Any]] = None,
    return_details: bool = False,
):
    """从评分矩阵计算 Fleiss' Kappa。

    Args:
        ratings: ``(n_subjects, n_raters)`` 的二维数组，``n_raters ≥ 2``。
        labels: 可选的类别全集。若不提供，从数据推断。
        return_details: 为 ``True`` 时返回 :class:`KappaResult`；否则只返回
            ``kappa`` 浮点数。

    Returns:
        Kappa 系数，或 :class:`KappaResult`（``method="fleiss"``）。
    """
    encoded, categories = _encode(ratings, labels)
    counts = _counts_from_encoded(encoded, len(categories))
    return fleiss_kappa_from_counts(
        counts,
        categories=categories,
        return_details=return_details,
    )


# ---------------------------------------------------------------------- #
# 公共入口
# ---------------------------------------------------------------------- #


def kappa(
    ratings: Any,
    *,
    labels: Optional[Sequence[Any]] = None,
    return_details: bool = False,
):
    """按评分者数自动分发到 Cohen（2 人）或 Fleiss（≥3 人）。

    Args:
        ratings: ``(n_subjects, n_raters)`` 的二维数组，``n_raters ≥ 2``。
        labels: 可选的类别全集。若不提供，从数据推断。
        return_details: 为 ``True`` 时返回 :class:`KappaResult`；否则只返回
            ``kappa`` 浮点数。

    Returns:
        Kappa 系数，或 :class:`KappaResult`。详情结果中的 ``method`` 字段
        标明实际使用的算法（``"cohen"`` / ``"fleiss"``）。

    Raises:
        ValueError: 评分者数 < 2，或输入不符合所选算法的要求。
    """
    arr = np.asarray(ratings)
    if arr.ndim != 2:
        raise ValueError(
            f"ratings must be a 2D array of shape (n_subjects, n_raters), "
            f"got {arr.ndim}D"
        )
    n_raters = arr.shape[1]
    if n_raters < 2:
        raise ValueError(f"at least 2 raters required, got {n_raters}")

    if n_raters == 2:
        return cohen_kappa(ratings, labels=labels, return_details=return_details)
    return fleiss_kappa(ratings, labels=labels, return_details=return_details)
