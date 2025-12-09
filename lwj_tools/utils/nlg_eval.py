import math
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum, unique
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from bert_score import score
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from transformers import BartForConditionalGeneration, BartTokenizer

from .constant import LOGGER
from .tools import cosine_similarity, load_glove

__all__ = ["NLGMetric", "BartScoreConfig", "BertScoreConfig", "NLGEvaluator"]


@unique
class NLGMetric(Enum):
    BLEU = 'bleu'
    GLEU = 'gleu'
    METEOR = 'meteor'
    ROUGE = 'rouge'
    BERT_SCORE = 'bert_score'
    BART_SCORE = 'bart_score'
    DISTINCT = 'distinct'
    GREEDY_MATCH = 'greedy_match'
    COSINE_SIMILAR = 'cosine_similar'
    EXTREMA_COSINE_SIMILAR = 'extrema_cosine_similar'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


@dataclass
class BartScoreConfig:
    name_or_path: str = 'facebook/bart-large-cnn'
    batch_size: int = 16
    device: str = 'cuda:0'
    max_length: int = 1024

    def to_dict(self):
        return asdict(self)


@dataclass
class BertScoreConfig:
    """
    In most cases, only four parameters are needed:
    model_type, num_layers, device, and batch_size

    If model_type uses the local path, then num_layers need to be explicitly given
    If model_type is the name on hf, then num_layers can be omitted
    """
    model_type: str
    num_layers: Optional[int] = None
    verbose: bool = False
    idf: bool = False
    device: Optional[str] = None  # default cuda:0
    batch_size: int = 64
    nthreads: int = 4
    all_layers: bool = False
    lang: Optional[str] = None
    return_hash: bool = False
    rescale_with_baseline: bool = False
    baseline_path: Optional[str] = None
    use_fast_tokenizer: bool = False

    def to_dict(self):
        return asdict(self)


class GLEU:
    def __init__(
        self,
        sources: List[str],
        references: List[List[str]],
        order: int = 4
    ):
        self.order = order
        source_size = len(sources)
        ref_size = len(references[0])  # maybe you have more than one standard answer
        assert source_size == ref_size
        self.samples = source_size

        self.all_source_ngrams: List[List[Counter]] = []
        self.process_sources(sources)
        self.refs_group: List[List[str]] = []
        self.ref_lens: List[List[int]] = []
        self.all_ref_ngrams_freq: List[Counter] = [Counter() for _ in range(order)]
        self.all_ref_ngrams: List[List[Counter]] = [[] for _ in range(self.samples)]
        self.process_references(references)

    @staticmethod
    def get_n_gram(sentence: str, n) -> Counter:
        words = sentence.split()
        return Counter(
            [
                tuple(words[j:j + n])
                for j in range(len(words) + 1 - n)
            ],
        )

    @staticmethod
    def get_ngram_diff(a, b) -> Counter:
        diff = Counter(a)
        for k in (set(a) & set(b)):
            del diff[k]
        return diff

    @staticmethod
    def gleu(stats, smooth=False):
        if smooth:
            stats = [s if s != 0 else 1 for s in stats]
        if len(list(filter(lambda x: x == 0, stats))) > 0:
            return 0
        c, r = stats[: 2]
        log_gleu_prec = sum([math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]) / 4
        return math.exp(min([0, 1 - float(r) / c]) + log_gleu_prec)

    def process_sources(self, sources: List[str]):
        self.all_source_ngrams = [
            [GLEU.get_n_gram(s, n) for n in range(1, self.order + 1)]
            for s in sources
        ]

    def process_references(self, references: List[List[str]], ):
        r"""
            references = [
                [r00, r01, r02, ...], reference file 0
                [r10, r11, r12, ...], reference file 1
                ...
            ]
            ref_groups: List[List[str]]= [
                [r00, r10, ...],
                [r01, r11, ...],
                [r02, r12, ...],
                ...
            ]
            ref_lens: List[List[int]] = [
                [len(r00.split()), len(r10.split()), ...],
                [len(r01.split()), len(r11.split()), ...],
                [len(r02.split()), len(r12.split()), ...],
                ...
            ]
        """
        for i in range(self.samples):
            self.refs_group.append([references[j][i] for j in range(len(references))])

        for i in range(self.samples):
            self.ref_lens.append([len(references[j][i].split()) for j in range(len(references))])

        for i, refs in enumerate(self.refs_group):
            for n in range(1, self.order + 1):
                ngrams: Counter = GLEU.get_n_gram(refs[0], n)
                self.all_ref_ngrams[i].append(ngrams)
                for k in ngrams.keys():
                    self.all_ref_ngrams_freq[n - 1][k] += 1
                for ref in refs[1:]:
                    new_ngrams = GLEU.get_n_gram(ref, n)
                    for nn in new_ngrams.elements():
                        if new_ngrams[nn] > ngrams.get(nn, 0):
                            ngrams[nn] = new_ngrams[nn]

    def normalization(self, ngram, n):
        return 1.0 * self.all_ref_ngrams_freq[n - 1][ngram] / len(self.ref_lens[0])

    def gleu_stats(self, hypothesis: str, hyp_ind: int, ref_ind: int):
        hyp_len = len(hypothesis.split())
        hyp_ngrams = [GLEU.get_n_gram(hypothesis, n) for n in range(1, self.order + 1)]

        ref_len = self.ref_lens[hyp_ind][ref_ind]

        yield hyp_len
        yield ref_len

        for n in range(1, self.order + 1):
            h_ngrams = hyp_ngrams[n - 1]
            s_ngrams = self.all_source_ngrams[hyp_ind][n - 1]
            r_ngrams = GLEU.get_n_gram(self.refs_group[hyp_ind][ref_ind], n)
            s_ngram_diff = GLEU.get_ngram_diff(s_ngrams, r_ngrams)
            yield max(
                [
                    sum((h_ngrams & r_ngrams).values()) - sum((h_ngrams & s_ngram_diff).values()),
                    0
                ],
            )
            yield max([hyp_len + 1 - n, 0])


def calc_bleu(
    references: List[str],
    hypothesis: List[str],
    tokenizer: Callable = str.split,
    n: Union[int, List[int]] = 4,
    weights: List[Tuple[float, ...]] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    if metrics is None:
        metrics = ['corpus-bleu', 'sentence-bleu']

    assert all(m in metrics for m in ['corpus-bleu', 'sentence-bleu']), \
        'The metrics should be "corpus-bleu" and "sentence-bleu"'

    n_samples = len(references)
    hyp_texts = [tokenizer(sent) for sent in hypothesis]
    ref_texts = [[tokenizer(sent)] for sent in references]

    ns = n
    if isinstance(ns, int):
        ns = [n]

    assert all(n > 0 and isinstance(n, int) for n in ns), \
        'The order should be an integer greater than 0'

    if weights is None:
        weights = [(1.0 / n,) * n for n in ns]

    if isinstance(weights, tuple):
        weights = [weights]

    assert len(weights) == len(ns), \
        'The number of weights and the number of orders are not consistent'

    results = {}

    if 'corpus-bleu' in metrics:
        corpus_bleu_scores = corpus_bleu(ref_texts, hyp_texts, weights)
        results['corpus-bleu'] = corpus_bleu_scores

    if 'sentence-bleu' in metrics:
        sentence_bleu_scores = np.asarray([0.0] * len(weights))
        for ref, hyp in zip(ref_texts, hyp_texts):
            cur_sent_scores = np.asarray(sentence_bleu(ref, hyp, weights))
            sentence_bleu_scores = sentence_bleu_scores + cur_sent_scores
        sentence_bleu_scores /= n_samples
        sentence_bleu_scores = sentence_bleu_scores.tolist()
        results['sentence-bleu'] = sentence_bleu_scores

    return results


def calc_rouge(
    references: List[str],
    hypothesis: List[str],
    tokenizer: Callable[[str], List[str]] = str.split,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    references = [
        ' '.join(tokenizer(ref))
        for ref in references
    ]
    hypothesis = [
        ' '.join(tokenizer(hyp))
        for hyp in hypothesis
    ]
    rouge_score = Rouge(metrics).get_scores(hypothesis, references, avg=True)
    return rouge_score


def calc_meteor(
    references: List[str],
    hypothesis: List[str],
    tokenizer: Callable = str.split,
) -> float:
    n_samples = len(references)
    hyp_texts = [tokenizer(sent) for sent in hypothesis]
    ref_texts = [[tokenizer(sent)] for sent in references]
    meteor = sum(meteor_score(ref, hyp) for ref, hyp in zip(ref_texts, hyp_texts)) / n_samples
    return meteor


def calc_gleu(
    sources: List[str],
    references: List[str],
    hypothesis: List[str],
    n: Union[int, List[int]] = 4,
) -> List[float]:
    n_samples = len(sources)
    ns = [n] if isinstance(n, int) else n
    assert all(n > 0 and isinstance(n, int) for n in ns), \
        'The order should be an integer greater than 0'

    gleu_scores = []
    for n_order in ns:
        gleu_calculator = GLEU(sources, [references], n_order)
        indices = [0] * n_samples
        stats = [0] * len(range(2 * n_order + 2))

        for i, hyp in enumerate(hypothesis):
            stats = [sum(scores) for scores in zip(
                stats,
                [s for s in gleu_calculator.gleu_stats(hyp, i, indices[i])],
            )]

        gleu_scores.append(GLEU.gleu(stats))
    return gleu_scores


def calc_bert_score(
    references: List[str],
    hypothesis: List[str],
    score_config: BertScoreConfig,
    reduction: str = 'mean',
    round_bits: int = 6,
    iter_size: int = 5000
) -> Union[Dict[str, float], Dict[str, List[float]]]:
    n_samples = len(references)
    P, R, F = [], [], []
    for i in range(0, n_samples, iter_size):
        p, r, f = score(
            cands=hypothesis[i:i + iter_size],
            refs=references[i:i + iter_size],
            **score_config.to_dict(),
        )
        P.extend(p.tolist())
        R.extend(r.tolist())
        F.extend(f.tolist())

    if reduction is None:
        reduction = 'none'

    reduction = reduction.lower()
    if reduction != 'none':
        if reduction == 'mean':
            P = sum(P) / n_samples
            R = sum(R) / n_samples
            F = sum(F) / n_samples
        elif reduction == 'sum':
            P = sum(P)
            R = sum(R)
            F = sum(F)
        else:
            raise ValueError(f'Unknown reduction = {reduction}')

        P = round(P, round_bits)
        R = round(R, round_bits)
        F = round(F, round_bits)

    return {
        'P': P,
        'R': R,
        'F': F,
    }


@torch.no_grad()
def calc_bart_score(
    references: List[str],
    hypothesis: List[str],
    score_config: BartScoreConfig,
) -> float:
    device = score_config.device
    batch_size = score_config.batch_size

    model = BartForConditionalGeneration.from_pretrained(score_config.name_or_path).to(device).eval()
    tokenizer = BartTokenizer.from_pretrained(score_config.name_or_path)
    loss_fct = nn.NLLLoss(ignore_index=model.config.pad_token_id, reduction='none')
    lsm = nn.LogSoftmax(dim=1)

    n_samples = len(references)
    score_list = []
    for i in range(0, n_samples, batch_size):
        src_list = hypothesis[i: i + batch_size]
        tgt_list = references[i: i + batch_size]

        try:
            encoded_src = tokenizer(
                src_list,
                max_length=score_config.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt',
            )
            encoded_tgt = tokenizer(
                tgt_list,
                max_length=score_config.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt',
            )

            src_tokens = encoded_src['input_ids'].to(device)
            src_mask = encoded_src['attention_mask'].to(device)

            tgt_tokens = encoded_tgt['input_ids'].to(device)
            tgt_mask = encoded_tgt['attention_mask']
            tgt_len = tgt_mask.sum(dim=1).to(device)

            output = model(
                input_ids=src_tokens,
                attention_mask=src_mask,
                labels=tgt_tokens,
            )

            logits = output.logits.view(-1, model.config.vocab_size)
            loss = loss_fct(lsm(logits), tgt_tokens.view(-1))
            loss = loss.view(tgt_tokens.shape[0], -1)  # [bsz, tgt_len]
            loss = loss.sum(dim=1) / tgt_len
            curr_score_list = [-x.item() for x in loss]
            score_list += curr_score_list

        except Exception as e:
            traceback.print_exc()
            LOGGER.error(e)

    bart_score = sum(score_list) / n_samples
    return bart_score


def calc_greedy_match_score(
    references: List[str],
    hypothesis: List[str],
    word_2_emb: Dict[str, np.ndarray],
    tokenizer: Optional[Callable] = str.split,
    unk_emb: Optional[np.ndarray] = None,
):
    emb_dim = list(word_2_emb.values())[0].shape[0]
    if unk_emb is None:
        unk_emb = np.zeros(emb_dim, dtype=np.float32)

    scores = []
    for hyp, ref in zip(hypothesis, references):
        # (seq_len, emb)
        hyp_emb = np.vstack([word_2_emb.get(hyp_token, unk_emb) for hyp_token in tokenizer(hyp)])
        ref_emb = np.vstack([word_2_emb.get(ref_token, unk_emb) for ref_token in tokenizer(ref)])

        # sim_matrix.shape = (hyp_len, ref_len)
        sim_matrix = cosine_similarity(hyp_emb, ref_emb)
        scores.append(sim_matrix.max())

    return np.mean(scores)


def calc_embedding_average_score(
    references: List[str],
    hypothesis: List[str],
    word_2_emb: Dict[str, np.ndarray],
    tokenizer: Optional[Callable] = str.split,
    unk_emb: Optional[np.ndarray] = None,
):
    emb_dim = list(word_2_emb.values())[0].shape[0]
    if unk_emb is None:
        unk_emb = np.zeros(emb_dim, dtype=np.float32)
    scores = []
    for hyp, ref in zip(hypothesis, references):
        # (seq_len, emb) -> (1, emb)
        hyp_emb = np.vstack([word_2_emb.get(hyp_token, unk_emb) for hyp_token in tokenizer(hyp)])
        hyp_emb = hyp_emb.mean(axis=0, keepdims=True)

        ref_emb = np.vstack([word_2_emb.get(ref_token, unk_emb) for ref_token in tokenizer(ref)])
        ref_emb = ref_emb.mean(axis=0, keepdims=True)

        # sim_matrix.shape = (1, 1)
        sim_matrix = cosine_similarity(hyp_emb, ref_emb)
        scores.append(sim_matrix[0][0])

    return np.mean(scores)


def calc_extrema_cosine_similar_score(
    references: List[str],
    hypothesis: List[str],
    word_2_emb: Dict[str, np.ndarray],
    tokenizer: Optional[Callable] = str.split,
    unk_emb: Optional[np.ndarray] = None,
):
    emb_dim = list(word_2_emb.values())[0].shape[0]
    if unk_emb is None:
        unk_emb = np.zeros(emb_dim, dtype=np.float32)

    scores = []
    for hyp, ref in zip(hypothesis, references):
        # (seq_len, emb) -> (1, emb)
        hyp_emb = np.vstack([word_2_emb.get(hyp_token, unk_emb) for hyp_token in tokenizer(hyp)])
        hyp_emb = hyp_emb.max(axis=0, keepdims=True)

        ref_emb = np.vstack([word_2_emb.get(ref_token, unk_emb) for ref_token in tokenizer(ref)])
        ref_emb = ref_emb.max(axis=0, keepdims=True)

        # sim_matrix.shape = (1, 1)
        sim_matrix = cosine_similarity(hyp_emb, ref_emb)
        scores.append(sim_matrix[0][0])

    return np.mean(scores)


def calc_distinct(
    hypothesis: List[str],
    tokenizer: Callable = str.split,
    n: Union[int, List[int]] = 4,
) -> List[float]:
    ns = [n] if isinstance(n, int) else n
    assert all(n > 0 and isinstance(n, int) for n in ns), 'The order should be an integer greater than 0.'

    ngrams: Dict[int, Counter] = {n: Counter() for n in ns}
    for hyp in hypothesis:
        hyp_words: List[str] = tokenizer(hyp)
        for n in ns:
            for i in range(len(hyp_words) - n + 1):
                ngrams[n][tuple(hyp_words[i:i + n])] += 1

    distinct_scores = [len(ngrams[n]) / sum(ngrams[n].values()) for n in ns]
    return distinct_scores


class NLGEvaluator:
    """Calculation of text generation evaluation indicators"""
    METRICS = {
        'overlap': [NLGMetric.BLEU, NLGMetric.METEOR, NLGMetric.ROUGE, NLGMetric.GLEU],
        'embedding': [NLGMetric.GREEDY_MATCH, NLGMetric.COSINE_SIMILAR, NLGMetric.EXTREMA_COSINE_SIMILAR],
        'pretrained': [NLGMetric.BART_SCORE, NLGMetric.BERT_SCORE],
        'diversity': [NLGMetric.DISTINCT],
    }

    FUN_MAP = {
        NLGMetric.BLEU: calc_bleu,
        NLGMetric.ROUGE: calc_rouge,
        NLGMetric.GLEU: calc_gleu,
        NLGMetric.METEOR: calc_meteor,
        NLGMetric.BERT_SCORE: calc_bert_score,
        NLGMetric.BART_SCORE: calc_bart_score,
        NLGMetric.GREEDY_MATCH: calc_greedy_match_score,
        NLGMetric.COSINE_SIMILAR: calc_extrema_cosine_similar_score,
        NLGMetric.EXTREMA_COSINE_SIMILAR: calc_extrema_cosine_similar_score,
        NLGMetric.DISTINCT: calc_distinct,
    }

    def __init__(
        self,
        metric_list: List[NLGMetric],
        tokenizer: Callable[[str], List[str]] = str.split,
        use_overlap: bool = False,
        use_embedding: bool = False,
        use_pretrained: bool = False,
        use_diversity: bool = False,
        omit_metric_list: List[NLGMetric] = None,
        bert_score_config: Dict[str, Any] = None,
        bart_score_config: Dict[str, Any] = None,
        glove_path: Optional[str] = None,
        word_2_emb: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
        glove_skip_first_row: bool = False,
    ):
        """
        Args:
            metric_list: 具体使用的指标
            tokenizer: 分词器, 默认str.split
            use_overlap: 是否使用 词重叠指标
            use_embedding: 是否使用 词嵌入指标
            use_pretrained: 是否使用 预训练模型指标
            use_diversity: 是否使用 多样性指标
            omit_metric_list: 不使用的指标
            bert_score_config: 配置请参考 bert_score.score
            bart_score_config: 配置请参考 https://github.com/neulab/BARTScore
            glove_path: 词向量文件
            word_2_emb: 词嵌入字典 (优先级更高)
            glove_skip_first_row: glove 文件是否跳过第一行, 有些文件第一行可能记录着词汇数量和维度
        """
        if omit_metric_list is None:
            omit_metric_list = []

        if not use_overlap:
            omit_metric_list.extend(self.METRICS['overlap'])

        if not use_embedding:
            omit_metric_list.extend(self.METRICS['embedding'])

        if not use_pretrained:
            omit_metric_list.extend(self.METRICS['pretrained'])

        if not use_diversity:
            omit_metric_list.extend(self.METRICS['diversity'])

        self.metric_list = [metric for metric in metric_list if metric not in omit_metric_list]

        assert len(self.metric_list) > 0, 'No indicators were specified'

        self.word_2_emb = None
        if use_embedding:
            assert glove_path or word_2_emb, \
                'Use the word vector metric, glove_path and word_2_emb provide at least one'

            if word_2_emb is not None:
                self.word_2_emb = word_2_emb
            else:
                LOGGER.info(f'Loading glove file from : {glove_path}')
                self.word_2_emb = load_glove(glove_path, glove_skip_first_row)

        self.tokenizer = tokenizer

        self.bert_score_config = bert_score_config if bert_score_config else BertScoreConfig('roberta-large')
        self.bart_score_config = bart_score_config if bart_score_config else BartScoreConfig()

    def __call__(
        self,
        hypothesis: List[str],
        references: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        specific_config: Optional[Dict[NLGMetric, Dict[str, Any]]] = None
    ):
        """
        Args:
            hypothesis: Generate text
            references: Reference text
            sources: Optional, suitable for translation/style transfer tasks with original text
            specific_config: If there are special parameter Settings for the metric calculation method, an example is
            as follows

        To calculate BLEU-1 to 4, you can directly configure the order `n`, like:
        >>>   specific_config[NLGMetric.BLEU] = {'n': 4}
        >>>   specific_config[NLGMetric.BLEU] = {'n': [1, 2, 3, 4]}
        Alternatively, you can specify the `weights` for each order.
        >>>   specific_config[NLGMetric.BLEU] = {
        >>>      'weights': [
        >>>           (1),
        >>>           (0.5, 0.5),
        >>>           (0.3, 0.3, 0.3),
        >>>           (0.25, 0.25, 0.25, 0.25)
        >>>      ]
        >>>  }
        Also, you can choose to calculate `corpus-bleu`, or `sentence-bleu`, like:
        >>> specific_config[NLGMetric.BLEU] = {'metrics': ['corpus-bleu', 'sentence-bleu']}

        To calculate Rouge, you can directly configure the metrics, like:
        >>>  # only the following are accepted: rouge-1, rouge-2, rouge-3, rouge-4, rouge-5, and rouge-l
        >>>  specific_config[NLGMetric.ROUGE] = {'metrics': ['rouge-1', 'rouge-2']}

        To calculate GLEU-1 to 4, you can directly configure the order `n`, like:
        >>>   specific_config[NLGMetric.GLEU] = {'n': 4}
        >>>   specific_config[NLGMetric.GLEU] = {'n': [1, 2, 3, 4]}

        To calculate diversity score, you can directly configure the order `n`, like:
        >>>   specific_config[NLGMetric.DISTINCT] = {'n': 4}
        >>>   specific_config[NLGMetric.DISTINCT] = {'n': [1, 2, 3, 4]}

        To calculate bertscore, you can set `round_bits`, the number of decimal places to be retained,  `reduction`,
        whether to process the result, and `iter_size`.
        Note that `iter_size` is set to address the issue of memory explosion in cases of large data volumes.
        For instance, if there are 100,000 samples, and they are directly input to `bert_score.score`, the required
        memory will exceed 64g.
        Therefore, the bert_score of the samples can be calculated in batches through iter_size, for example, 5000,
        and then summarized and sumed/averaged/not processed
        >>> specific_config[NLGMetric.BERT_SCORE] = {'reduction': 'mean', 'iter_size': 5000, 'round_bits': 6}
        """
        assert len(hypothesis) == len(references), \
            'hypothesis and references must have the same length'

        result = {}
        for metric in self.metric_list:
            try:
                func = self.FUN_MAP[metric]
                func_kwargs: Dict[str, Any] = {
                    'hypothesis': hypothesis,
                    'references': references
                }
                if metric in self.METRICS['overlap']:
                    func_kwargs['tokenizer'] = self.tokenizer

                    if metric == NLGMetric.GLEU:
                        assert sources is not None, ('GLEU requires sources')
                        assert len(sources) == len(hypothesis)
                        func_kwargs['sources'] = sources

                if metric in self.METRICS['pretrained']:
                    func_kwargs['score_config'] = self.bert_score_config \
                        if metric == NLGMetric.BERT_SCORE \
                        else self.bart_score_config

                if metric in self.METRICS['embedding']:
                    func_kwargs['word_2_emb'] = self.word_2_emb
                    func_kwargs['tokenizer'] = self.tokenizer

                if specific_config.get(metric, None):
                    func_kwargs.update(specific_config[metric])

                result[str(metric)] = func(**func_kwargs)
            except Exception as e:
                LOGGER.error(f'The calculation of the evaluation indicators failed, metric:{metric}')
                LOGGER.error(e)

        return result
