import os, re, json, torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer , AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# ========== Google Drive ==========
try:
    from google.colab import drive
    drive.mount('/content/drive')
    NLI_MODEL_PATH = "/content/drive/MyDrive/nli_model"
    DATA_PATH = "/content/drive/MyDrive/언어 임베딩/test_data.json"
    OUTPUT_PATH = "/content/drive/MyDrive/언어 임베딩/추론 결과/nli_results.json"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print("[+] Google Drive 마운트 완료")
except:
    NLI_MODEL_PATH = "./nli_model"
    DATA_PATH = "./test_data.json"
    OUTPUT_PATH = "./nli_results.json"
    print("[i] 로컬 환경")
 
# ========== STS 모델 (코사인 유사도 계산용) ==========
STS_MODEL_NAME = "jhgan/ko-sbert-sts"
 
LABEL_NAMES = ["entailment", "neutral", "contradiction"]



# ========== NLI 추론 클래스 ==========
class NLIPredictor:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[NLI] 모델 로드 완료 ({self.device})")
 
    def predict(self, premise, hypothesis):
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)
 
        with torch.no_grad():
            probs = F.softmax(self.model(**inputs).logits, dim=1)[0]
 
        label = LABEL_NAMES[probs.argmax().item()]
        scores = {n: round(p.item(), 4) for n, p in zip(LABEL_NAMES, probs)}
        return {"label": label, "scores": scores}
 
 
# ========== 문장 분리 ==========
def split_sentences(text):
    """뉴스 본문을 문장 단위로 분리"""
    # 줄바꿈 → 마침표 기준 분리
    text = text.replace('\n\n', '. ').replace('\n', '. ')
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # 10자 이상인 문장만 (너무 짧은 건 제외)
    return [s.strip() for s in sentences if len(s.strip()) > 10]
 
 
# ========== 뉴스 쌍 신규 정보 탐지 ==========
def detect_novel_sentences(nli, existing_sents, new_sents, threshold=0.5):
    """
    새 기사의 각 문장에 대해 기존 기사와 NLI 비교
    
    Args:
        existing_sents: 기존 기사의 문장 리스트 (전제)
        new_sents: 새 기사의 문장 리스트 (가설)
        threshold: entailment 확률 임계값
    
    Returns:
        각 문장별 판정 결과
    """
    results = []
 
    for h_sent in new_sents:
        max_ent = 0.0
        max_con = 0.0
        best_match = ""
        best_label = "neutral"
 
        for p_sent in existing_sents:
            pred = nli.predict(premise=p_sent, hypothesis=h_sent)
            ent = pred["scores"]["entailment"]
            con = pred["scores"]["contradiction"]
 
            if ent > max_ent:
                max_ent = ent
                best_match = p_sent
 
            if con > max_con:
                max_con = con
 
        # 판정: entailment가 높으면 중복, 아니면 신규
        if max_ent >= threshold:
            category = "redundant"
        elif max_con >= threshold:
            category = "contradiction"
        else:
            category = "novel"
 
        results.append({
            "sentence": h_sent,
            "category": category,
            "max_entailment": round(max_ent, 4),
            "max_contradiction": round(max_con, 4),
            "best_match": best_match
        })
 
    return results
 
 
# ========== 코사인 유사도 계산 ==========
def compute_cosine_pairs(news_data, sts_model):
    """뉴스 데이터에서 순차적으로 코사인 유사도 계산"""
    print(f"\n[STS] 코사인 유사도 계산 중...")
 
    def combine_text(title, context):
        return f"{title} {title} {context[:500]}"
 
    pairs = []
    embeddings = None
    doc_ids = []
    items_list = list(news_data.items())
 
    for i, (doc_id, item) in enumerate(items_list):
        title = item.get("title", "")
        context = item.get("context", "")
        combined = combine_text(title, context)
        emb = sts_model.encode(combined, convert_to_numpy=True)
 
        if embeddings is not None and len(doc_ids) > 0:
            # 기존 뉴스들과 유사도 계산
            query_norm = emb / (np.linalg.norm(emb) + 1e-10)
            doc_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
            sims = np.dot(doc_norms, query_norm)
 
            max_idx = np.argmax(sims)
            max_sim = float(sims[max_idx])
            similar_id = doc_ids[max_idx]
 
            pairs.append({
                "new_id": doc_id,
                "existing_id": similar_id,
                "cosine_similarity": round(max_sim, 4)
            })
 
        doc_ids.append(doc_id)
        if embeddings is None:
            embeddings = emb.reshape(1, -1)
        else:
            embeddings = np.vstack([embeddings, emb])
 
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(items_list)}")
 
    return pairs
 
 
# ========== 층화 추출 ==========
def stratified_sample(pairs, n_per_bin=None):
    """코사인 유사도 구간별 등화 추출"""
    bins = {
        "0.5-0.7": [], "0.7-0.8": [],
        "0.8-0.9": [], "0.9-1.0": []
    }
 
    # 기본 배분
    if n_per_bin is None:
        n_per_bin = {"0.5-0.7": 20, "0.7-0.8": 25, "0.8-0.9": 30, "0.9-1.0": 25}
 
    for p in pairs:
        sim = p["cosine_similarity"]
        if 0.5 <= sim < 0.7:
            bins["0.5-0.7"].append(p)
        elif 0.7 <= sim < 0.8:
            bins["0.7-0.8"].append(p)
        elif 0.8 <= sim < 0.9:
            bins["0.8-0.9"].append(p)
        elif 0.9 <= sim <= 1.0:
            bins["0.9-1.0"].append(p)
 
    sampled = []
    for bin_name, bin_pairs in bins.items():
        n = min(n_per_bin.get(bin_name, 20), len(bin_pairs))
        if n > 0:
            indices = np.random.choice(len(bin_pairs), n, replace=False)
            for idx in indices:
                sampled.append({**bin_pairs[idx], "bin": bin_name})
        print(f"  구간 {bin_name}: {len(bin_pairs)}개 중 {n}개 추출")
 
    return sampled
 
 
# ========== 메인 ==========
def main():
    print(f"\n{'='*60}")
    print(f"  NLI 기반 뉴스 신규 정보 탐지")
    print(f"{'='*60}")
 
    # 1. 데이터 로드
    print(f"\n[1/5] 뉴스 데이터 로드")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        news_data = json.load(f)
    print(f"  총 {len(news_data)}개 뉴스")
 
    # 2. STS 모델로 코사인 유사도 계산
    print(f"\n[2/5] STS 모델 로드: {STS_MODEL_NAME}")
    sts_model = SentenceTransformer(STS_MODEL_NAME)
    pairs = compute_cosine_pairs(news_data, sts_model)
    print(f"  총 {len(pairs)}개 뉴스 쌍 생성")
 
    # 유사도 분포 출력
    print(f"\n  [유사도 분포]")
    for low, high in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        count = sum(1 for p in pairs if low <= p["cosine_similarity"] < high)
        pct = count / len(pairs) * 100
        print(f"    [{low:.1f}-{min(high,1.0):.1f}]: {count:4d} ({pct:.1f}%)")
 
    # 3. 층화 추출
    print(f"\n[3/5] 층화 추출 (100개)")
    sampled = stratified_sample(pairs)
    print(f"  추출 완료: {len(sampled)}개")
 
    # 4. NLI 모델 로드
    print(f"\n[4/5] NLI 모델 로드: {NLI_MODEL_PATH}")
    nli = NLIPredictor(NLI_MODEL_PATH)
 
    # 5. 추론
    print(f"\n[5/5] NLI 추론 시작")
    print(f"{'='*60}")
 
    all_results = []
 
    for i, sample in enumerate(sampled):
        new_item = news_data[sample["new_id"]]
        existing_item = news_data[sample["existing_id"]]
 
        new_sents = split_sentences(new_item.get("context", ""))
        existing_sents = split_sentences(existing_item.get("context", ""))
 
        if not new_sents or not existing_sents:
            continue
 
        # NLI 적용
        sent_results = detect_novel_sentences(nli, existing_sents, new_sents)
 
        # 집계
        novel_count = sum(1 for r in sent_results if r["category"] == "novel")
        redundant_count = sum(1 for r in sent_results if r["category"] == "redundant")
        total = len(sent_results)
        novelty_ratio = novel_count / total if total > 0 else 0
 
        result = {
            "pair_index": i + 1,
            "bin": sample["bin"],
            "cosine_similarity": sample["cosine_similarity"],
            "new_title": new_item.get("title", ""),
            "existing_title": existing_item.get("title", ""),
            "total_sentences": total,
            "novel_count": novel_count,
            "redundant_count": redundant_count,
            "novelty_ratio": round(novelty_ratio, 4),
            "sentences": sent_results
        }
        all_results.append(result)
 
        # 진행률
        print(f"\n  [{i+1}/{len(sampled)}] 구간: {sample['bin']} | cos: {sample['cosine_similarity']}")
        print(f"    기존: {existing_item.get('title', '')[:40]}...")
        print(f"    신규: {new_item.get('title', '')[:40]}...")
        print(f"    결과: 신규 {novel_count}개 / 전체 {total}개 (비율: {novelty_ratio:.1%})")
 
    # 6. 구간별 요약
    print(f"\n{'='*60}")
    print(f"  구간별 신규 정보 탐지 요약")
    print(f"{'='*60}")
 
    for bin_name in ["0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]:
        bin_results = [r for r in all_results if r["bin"] == bin_name]
        if bin_results:
            avg_novelty = np.mean([r["novelty_ratio"] for r in bin_results])
            avg_cos = np.mean([r["cosine_similarity"] for r in bin_results])
            print(f"  {bin_name}: 평균 cos={avg_cos:.3f} | 평균 신규비율={avg_novelty:.1%} | {len(bin_results)}쌍")
 
    # 7. 결과 저장
    output = {
        "total_pairs": len(all_results),
        "summary": {},
        "results": all_results
    }
 
    for bin_name in ["0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]:
        bin_results = [r for r in all_results if r["bin"] == bin_name]
        if bin_results:
            output["summary"][bin_name] = {
                "count": len(bin_results),
                "avg_cosine": round(np.mean([r["cosine_similarity"] for r in bin_results]), 4),
                "avg_novelty_ratio": round(np.mean([r["novelty_ratio"] for r in bin_results]), 4)
            }
 
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
 
    print(f"\n  결과 저장: {OUTPUT_PATH}")
    print(f"\n{'='*60}")
    print(f"  완료!")
    print(f"{'='*60}")
 
 
if __name__ == "__main__":
    main()
