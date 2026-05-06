# Korean News Deduplicator: STS-NLI Combined Pipeline

This is the repository for **"코사인 유사도의 한계를 넘어서: 한국어 뉴스에서의 STS-NLI 결합 기반 신규 정보 탐지"**<br>
(Beyond Cosine Similarity: Novel Information Detection in Korean News via Combined STS-NLI Approach). <br>
We propose a two-stage pipeline that combines Semantic Textual Similarity (STS) and Natural Language Inference (NLI) for detecting novel information in Korean news articles.

## Overview
코사인 유사도 기반 중복 탐지는 두 텍스트의 비교를 단일 스칼라 값으로 압축하여, 어떤 부분이 새로운 정보인지 식별하는 능력을 상실합니다. 본 파이프라인은 이를 다음과 같이 해결합니다:<br>
Cosine similarity-based deduplication compresses pairwise comparisons into a single scalar, losing the ability to identify *which* portions contain new information. <br>
Our pipeline addresses this by:<br>

- Stage 1 (STS, Bi-Encoding): 거시적 필터링 — "얼마나 비슷한가?" / Fast macro-level filtering — "How similar are they?"
- Stage 2 (NLI, Cross-Encoding): 문장 수준 미시 분석 — "어디가 다른가?" / Sentence-level micro-analysis — "Where do they differ?"
<br>

NLI의 Neutral 판정은 신규 정보의 대리 지표로 기능합니다. 새 기사의 문장이 기존 기사의 어떤 문장으로부터도 함의(entailment)되지 않을 경우, 해당 문장을 신규 정보로 식별합니다.<br>
The NLI **Neutral** label serves as a proxy for novel information: sentences in a new article that cannot be entailed from any sentence in the existing article are identified as carrying new information.

## Models

### STS Models

We compare three SBERT models for cosine similarity computation:

| Model | Type | HuggingFace |
|-------|------|-------------|
| `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual lightweight | [sentence-transformers](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| `jhgan/ko-sroberta-multitask` | Korean multi-task | [jhgan](https://huggingface.co/jhgan/ko-sroberta-multitask) |
| `jhgan/ko-sbert-sts` | Korean STS-specialized | [jhgan](https://huggingface.co/jhgan/ko-sbert-sts) |

### NLI Model

We fine-tune [KLUE-RoBERTa-base](https://huggingface.co/klue/roberta-base) on the [KorNLI](https://github.com/kakaobrain/kor-nlu-datasets) dataset (942,854 training pairs).

| | Train | Dev | Test |
|---|---|---|---|
| Source | SNLI + MNLI (ko) | XNLI (ko) | XNLI (ko) |
| Size | 942,854 | 5,010 | 5,010 |

**Fine-tuning configuration**: lr=2e-5, batch=256, epochs=3, max_length=128, AdamW, NVIDIA A100 80GB

The fine-tuned model is available at: [`nomad0884/korean-nli`](https://huggingface.co/nomad0884/korean-nli)

## Results

### NLI Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | **83.3%** |
| Best Epoch | 2 |
| Neutral→Entailment Error Rate | 9.9% |

**Confusion Matrix** (XNLI test set, 5,010 samples):

| | Pred Entailment | Pred Neutral | Pred Contradiction |
|---|:---:|:---:|:---:|
| True Entailment | 1,369 | 224 | 77 |
| True Neutral | 165 | 1,352 | 153 |
| True Contradiction | 48 | 172 | 1,450 |

### STS Distribution (1,184 Korean news pairs)

| Range | paraphrase-multilingual | ko-sroberta-multitask | ko-sbert-sts |
|-------|:---:|:---:|:---:|
| 0.0–0.5 | 5 (0.4%) | 9 (0.8%) | 9 (0.8%) |
| 0.5–0.7 | 120 (10.1%) | 122 (10.3%) | 149 (12.6%) |
| 0.7–0.9 | 320 (27.0%) | 205 (17.3%) | 255 (21.5%) |
| 0.9–1.0 | 739 (62.4%) | 848 (71.6%) | 770 (65.1%) |

### Novel Information Detection (100 stratified samples)

| Cosine Range | Samples | Avg. Cosine | Avg. Novelty Ratio |
|:---:|:---:|:---:|:---:|
| 0.5–0.7 | 20 | 0.614 | 44.5% |
| 0.7–0.8 | 25 | 0.746 | 39.1% |
| **0.8–0.9** | **30** | **0.849** | **26.9%** |
| 0.9–1.0 | 25 | 0.985 | 3.1% |

> **Key finding**: In the 0.8–0.9 cosine similarity range, where articles would typically be classified as duplicates, **26.9% of sentences contained novel information** — approximately 1 in 4 sentences.

## Installation

```bash
pip install git+https://github.com/nomad0884/Beyond-Cosine-Similarity.git
```

or from source:

```bash
git clone https://github.com/nomad0884/Beyond-Cosine-Similarity.git
cd Beyond-Cosine-Similarity
pip install -e .
```

**Requirements**: Python ≥ 3.8, torch, transformers, sentence-transformers, numpy

## Usage

```python
from korean_news_dedup import NewsDeduplicator

# Initialize (models are downloaded automatically)
dedup = NewsDeduplicator()

# Load existing news
dedup.load_from_json("news_data.json")

# Check new article with STS + NLI
result = dedup.check_with_nli(
    title="삼성전자 HBM4 양산 시작",
    context="삼성전자가 HBM4 양산을 시작했다. KB증권은 목표주가를 상향했다."
)

print(f"Duplicate: {result['is_duplicate']}")
print(f"Cosine Similarity: {result['cosine_similarity']}")
print(f"Novel Sentences: {result['novel_count']}")

for sent in result['novel_sentences']:
    print(f"  [Novel] {sent['sentence']}")
```

### Batch Processing

```python
unique, duplicates, has_novel = dedup.filter_news_with_nli(new_news_dict)

# unique: 신규 기사
# duplicates: 완전 중복
# has_novel: 중복이지만 신규 정보 포함
```

### JSON Format

```json
{
    "doc_id_or_url": {
        "title": "뉴스 제목",
        "context": "뉴스 본문",
        "timeStamp": "Wed, 07 Jan 2026 16:13:00 +0900"
    }
}
```

## Dataset

- [test_data.json](https://drive.google.com/file/d/1jRaBhnGHn1ARsd3CDLQ5ZL8nkMcSIUxR/view?usp=sharing) — 1,184 Korean news articles (semiconductor/stock market), collected via Naver News API (keyword: 반도체)
- [korean_nli_model](https://huggingface.co/nomad0884/korean-nli) - Hugginhface model weight pth.
### External Datasets

- [KorNLI & KorSTS](https://github.com/kakaobrain/kor-nlu-datasets) — KakaoBrain (Ham et al., EMNLP 2020 Findings)
- [KLUE](https://github.com/KLUE-benchmark/KLUE) — Korean Language Understanding Evaluation

## Project Structure

```
korean-news-dedup/
├── korean_news_dedup/
│   ├── __init__.py
│   ├── deduplicator.py      # Main pipeline class
│   ├── sts.py                # STS model wrapper
│   └── nli.py                # NLI model wrapper
├── README.md
├── setup.py
├── requirements.txt
└── LICENSE
```
## CopyWrite
국민대학교 국제통상학부/소프트웨어융합대학 김수만 <br>

<br>
## References

```
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={Findings of EMNLP},
  year={2020}
}
```

1. Ham, J. et al. (2020). "KorNLI and KorSTS: New Benchmark Datasets for Korean NLU." *Findings of EMNLP 2020*
2. Park, S. et al. (2021). "KLUE: Korean Language Understanding Evaluation." *arXiv:2105.09680*
3. Bowman, S. R. et al. (2015). "A Large Annotated Corpus for Learning Natural Language Inference." *EMNLP 2015*
4. Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP 2019*
5. Liu, Y. et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." *arXiv:1907.11692*
6. Zhelezniak, V. et al. (2019). "Correlation Coefficients and Semantic Textual Similarity." *NAACL 2019*
7. Ghosal, T. et al. (2022). "Novelty Detection: A Perspective from NLP." *Computational Linguistics*
8. Laban, P. et al. (2022). "SummaC: NLI-based Inconsistency Detection in Summarization." *TACL*

## License

[MIT License](LICENSE)
