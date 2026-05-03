"""
NewsDeduplicator — STS-NLI 결합 뉴스 중복 탐지 파이프라인

Stage 1: STS (Bi-Encoding) — "얼마나 비슷한가" → 빠른 후보 필터링
Stage 2: NLI (Cross-Encoding) — "어디가 다른가" → 문장 수준 신규 정보 탐지
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

from .sts import STSModel
from .nli import NLIModel


class NewsDeduplicator:
    """
    STS-NLI 결합 뉴스 중복 탐지기

    사용법:
        dedup = NewsDeduplicator()
        dedup.load_from_json("news_data.json")
        result = dedup.check_with_nli(title="...", context="...")
    """

    def __init__(
        self,
        sts_model: str = "jhgan/ko-sbert-sts",
        nli_model: str = "nomad0884/korean-nli",
        sts_threshold: float = 0.8,
        nli_threshold: float = 0.5,
        novelty_threshold: float = 0.9,
    ):
        """
        Args:
            sts_model: STS 임베딩 모델 이름 (HuggingFace)
            nli_model: NLI 분류 모델 이름 (HuggingFace)
            sts_threshold: 코사인 유사도 중복 임계값
            nli_threshold: NLI entailment 임계값 (미만이면 신규)
            novelty_threshold: 이 이상이면 완전 중복으로 판정 (NLI 스킵)
        """
        print(f"\n{'='*50}")
        print(f"  Korean News Deduplicator 초기화")
        print(f"{'='*50}")

        self.sts = STSModel(sts_model)
        self.nli = NLIModel(nli_model)
        self.sts_threshold = sts_threshold
        self.nli_threshold = nli_threshold
        self.novelty_threshold = novelty_threshold

        # 기존 뉴스 저장소
        self.news_data: Dict[str, Dict] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []

        print(f"\n  STS 임계값: {sts_threshold}")
        print(f"  NLI 임계값: {nli_threshold}")
        print(f"  완전중복 임계값: {novelty_threshold}")
        print(f"{'='*50}\n")

    def _combine_text(self, title: str, context: str) -> str:
        """제목 2회 반복 + 본문 500자"""
        return f"{title} {title} {context[:500]}"

    # =========================================================
    # 데이터 로드/저장
    # =========================================================

    def load_from_json(self, filepath: str) -> int:
        """JSON 파일에서 기존 뉴스 로드"""
        if not os.path.exists(filepath):
            print(f"[!] 파일 없음: {filepath}")
            return 0

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = []
        for doc_id, item in data.items():
            self.news_data[doc_id] = item
            self.doc_ids.append(doc_id)
            texts.append(self._combine_text(
                item.get("title", ""), item.get("context", "")
            ))

        print(f"[STS] {len(texts)}개 뉴스 임베딩 생성 중...")
        self.embeddings = self.sts.encode_batch(texts)
        print(f"[STS] 로드 완료: {len(texts)}개")
        return len(texts)

    def save_to_json(self, filepath: str):
        """현재 뉴스 데이터를 JSON으로 저장"""
        output = {doc_id: self.news_data[doc_id] for doc_id in self.doc_ids}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[저장] {filepath} ({len(output)}개)")

    # =========================================================
    # Stage 1: STS 중복 검사
    # =========================================================

    def check_sts(self, title: str, context: str) -> dict:
        """STS 기반 코사인 유사도 검사"""
        if self.embeddings is None or len(self.doc_ids) == 0:
            return {"is_duplicate": False, "cosine_similarity": 0.0,
                    "similar_id": None, "similar_title": None}

        combined = self._combine_text(title, context)
        query_emb = self.sts.encode(combined)
        sims = self.sts.cosine_similarity_batch(query_emb, self.embeddings)

        max_idx = np.argmax(sims)
        max_sim = float(sims[max_idx])
        similar_id = self.doc_ids[max_idx]
        similar_title = self.news_data.get(similar_id, {}).get("title", "")

        return {
            "is_duplicate": max_sim >= self.sts_threshold,
            "cosine_similarity": round(max_sim, 4),
            "similar_id": similar_id,
            "similar_title": similar_title
        }

    # =========================================================
    # Stage 2: NLI 신규 정보 탐지
    # =========================================================

    def check_with_nli(self, title: str, context: str) -> dict:
        """
        STS + NLI 결합 검사

        Returns:
            {
                "is_duplicate": bool,
                "cosine_similarity": float,
                "has_novel_info": bool,
                "novel_count": int,
                "novel_sentences": [...],
                "novelty_ratio": float,
                ...
            }
        """
        # Stage 1: STS
        sts_result = self.check_sts(title, context)

        # 유사한 기사가 없으면 신규 기사
        if not sts_result["is_duplicate"]:
            return {
                **sts_result,
                "has_novel_info": True,
                "novel_count": 0,
                "novel_sentences": [],
                "redundant_sentences": [],
                "novelty_ratio": 1.0,
                "nli_applied": False,
                "reason": "STS 유사도 미달 → 신규 기사"
            }

        # 완전 중복 (0.9 이상) → NLI 스킵
        if sts_result["cosine_similarity"] >= self.novelty_threshold:
            return {
                **sts_result,
                "has_novel_info": False,
                "novel_count": 0,
                "novel_sentences": [],
                "redundant_sentences": [],
                "novelty_ratio": 0.0,
                "nli_applied": False,
                "reason": f"cos >= {self.novelty_threshold} → 완전 중복"
            }

        # Stage 2: NLI (0.8 ~ 0.9 구간)
        existing_context = self.news_data.get(
            sts_result["similar_id"], {}
        ).get("context", "")

        nli_result = self.nli.detect_novel(
            existing_text=existing_context,
            new_text=context,
            threshold=self.nli_threshold
        )

        has_novel = nli_result["novel_count"] > 0

        return {
            **sts_result,
            "has_novel_info": has_novel,
            "novel_count": nli_result["novel_count"],
            "novel_sentences": nli_result["novel"],
            "redundant_sentences": nli_result["redundant"],
            "novelty_ratio": nli_result["novelty_ratio"],
            "nli_applied": True,
            "reason": f"NLI 분석 → 신규 {nli_result['novel_count']}/{nli_result['total']}문장"
        }

    # =========================================================
    # 배치 처리
    # =========================================================

    def add_news(self, doc_id: str, item: Dict):
        """새 뉴스를 히스토리에 추가"""
        combined = self._combine_text(
            item.get("title", ""), item.get("context", "")
        )
        emb = self.sts.encode(combined)

        self.news_data[doc_id] = item
        self.doc_ids.append(doc_id)

        if self.embeddings is None:
            self.embeddings = emb.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

    def filter_news_with_nli(
        self,
        news_dict: Dict[str, Dict],
        verbose: bool = True
    ) -> Tuple[Dict, Dict, Dict]:
        """
        배치 뉴스 필터링 (STS + NLI)

        Returns:
            (신규 기사, 완전 중복, 중복이지만 신규 정보 포함)
        """
        unique = {}
        duplicates = {}
        has_novel = {}

        if verbose:
            print(f"\n[검사] {len(news_dict)}개 뉴스 vs 기존 {len(self.doc_ids)}개")
            print(f"{'='*60}")

        for doc_id, item in news_dict.items():
            title = item.get("title", "")
            context = item.get("context", "")

            result = self.check_with_nli(title, context)

            if not result["is_duplicate"]:
                unique[doc_id] = {**item, "_result": result}
                self.add_news(doc_id, item)
                if verbose:
                    print(f"✅ 신규 [{result['cosine_similarity']:.2f}]: {title[:50]}")

            elif result["has_novel_info"]:
                has_novel[doc_id] = {**item, "_result": result}
                self.add_news(doc_id, item)
                if verbose:
                    print(f"🔶 중복+신규정보 [{result['cosine_similarity']:.2f}]"
                          f" 신규 {result['novel_count']}문장: {title[:40]}")

            else:
                duplicates[doc_id] = {**item, "_result": result}
                if verbose:
                    print(f"❌ 완전중복 [{result['cosine_similarity']:.2f}]: {title[:50]}")

        if verbose:
            print(f"{'='*60}")
            print(f"[결과] 신규: {len(unique)} | 중복+신규정보: {len(has_novel)}"
                  f" | 완전중복: {len(duplicates)}")

        return unique, duplicates, has_novel

    # =========================================================
    # 통계
    # =========================================================

    def get_stats(self) -> dict:
        return {
            "total_documents": len(self.doc_ids),
            "sts_model": self.sts.model_name,
            "nli_model": self.nli.model_name,
            "sts_threshold": self.sts_threshold,
            "nli_threshold": self.nli_threshold,
            "novelty_threshold": self.novelty_threshold,
        }
