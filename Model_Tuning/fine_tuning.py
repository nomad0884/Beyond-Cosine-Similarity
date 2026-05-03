# wrote at Google Colab
# Colab 셀 1: 설치
!pip install transformers torch

# Colab 셀 2: NLI_Finetuning.py 업로드 후 실행
!python NLI_Finetuning.py


"""
KLUE-RoBERTa-base KorNLI Fine-Tuning
======================================
Google Colab + Drive 저장 / 사전 토크나이징 버전

실행 전: !pip install transformers torch
"""

import os, json, torch, numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

# ========== Google Drive ==========
try:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = "/content/drive/MyDrive/nli_model"
    print("[+] Google Drive 마운트 완료")
except:
    SAVE_DIR = "./nli_model"
    print("[i] 로컬 환경")

# ========== 설정 ==========
CONFIG = {
    "model_name": "klue/roberta-base",
    "max_length": 128,
    "batch_size": 256,
    "learning_rate": 2e-5,
    "epochs": 3,
    "warmup_ratio": 0.1,
    "num_labels": 3,
    "save_dir": SAVE_DIR,
}
LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}
LABEL_NAMES = ["entailment", "neutral", "contradiction"]


# ========== 데이터셋 (사전 토크나이징) ==========
class KorNLIDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        premises, hypotheses, labels = [], [], []
        with open(filepath, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    label = parts[2].strip().lower()
                    if label in LABEL_MAP:
                        premises.append(parts[0])
                        hypotheses.append(parts[1])
                        labels.append(LABEL_MAP[label])

        print(f"  토크나이징: {len(labels)}개 ({os.path.basename(filepath)})...")
        encodings = tokenizer(
            premises, hypotheses,
            truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt"
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)
        print(f"  완료: {len(labels)}개")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }


# ========== 데이터 다운로드 ==========
def download_kornli():
    import urllib.request
    base = "https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorNLI"
    files = ["multinli.train.ko.tsv", "snli_1.0_train.ko.tsv",
             "xnli.dev.ko.tsv", "xnli.test.ko.tsv"]
    os.makedirs("./kornli_data", exist_ok=True)
    for f in files:
        path = f"./kornli_data/{f}"
        if not os.path.exists(path):
            print(f"  다운로드: {f}")
            urllib.request.urlretrieve(f"{base}/{f}", path)
        else:
            print(f"  존재: {f}")
    return "./kornli_data"


# ========== 학습 ==========
def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for i, batch in enumerate(loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += out.loss.item()
        correct += (torch.argmax(out.logits, dim=1) == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 100 == 0:
            print(f"    batch {i+1}/{len(loader)} | loss: {total_loss/(i+1):.4f} | acc: {correct/total*100:.1f}%")

    return total_loss / len(loader), correct / total * 100


# ========== 평가 ==========
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            total_loss += out.loss.item()
            preds = torch.argmax(out.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(loader), correct / total * 100,
            np.array(all_preds), np.array(all_labels))


# ========== Confusion Matrix ==========
def print_confusion_matrix(preds, labels):
    matrix = np.zeros((3, 3), dtype=int)
    for p, l in zip(preds, labels):
        matrix[l][p] += 1
    print(f"\n  {'':>15} | {'Pred Ent':>10} | {'Pred Neu':>10} | {'Pred Con':>10}")
    print(f"  {'-'*55}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {'True '+name:>15} | {matrix[i][0]:>10} | {matrix[i][1]:>10} | {matrix[i][2]:>10}  ({matrix[i].sum()})")
    if matrix[1].sum() > 0:
        print(f"\n  Neutral->Entailment 오분류율: {matrix[1][0]/matrix[1].sum()*100:.1f}%")
    return matrix


# ========== 메인 ==========
def main():
    print(f"\n{'='*60}")
    print(f"  KLUE-RoBERTa NLI Fine-Tuning")
    print(f"  Save: {CONFIG['save_dir']}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 1. 데이터
    print(f"\n[1/5] 데이터 다운로드")
    data_dir = download_kornli()

    # 2. 토크나이저 & 모델
    print(f"\n[2/5] 모델 로드: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"], num_labels=3
    ).to(device)

    # 3. 데이터셋 (사전 토크나이징 — 시간 소요되지만 학습이 빨라짐)
    print(f"\n[3/5] 데이터셋 로드 + 사전 토크나이징")
    print(f"  (이 단계에서 시간이 걸리지만, 학습 속도가 크게 향상됩니다)")
    train_dataset = ConcatDataset([
        KorNLIDataset(f"{data_dir}/snli_1.0_train.ko.tsv", tokenizer, CONFIG["max_length"]),
        KorNLIDataset(f"{data_dir}/multinli.train.ko.tsv", tokenizer, CONFIG["max_length"])
    ])
    dev_dataset = KorNLIDataset(f"{data_dir}/xnli.dev.ko.tsv", tokenizer, CONFIG["max_length"])
    test_dataset = KorNLIDataset(f"{data_dir}/xnli.test.ko.tsv", tokenizer, CONFIG["max_length"])
    print(f"\n  Train: {len(train_dataset)} | Dev: {len(dev_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=CONFIG["batch_size"],
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                             num_workers=4, pin_memory=True)

    # 4. 옵티마이저
    print(f"\n[4/5] 학습 설정")
    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"  Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['learning_rate']}")
    print(f"  Total steps: {total_steps} | Warmup: {warmup_steps}")

    # 5. 학습
    print(f"\n[5/5] 학습 시작")
    print(f"{'='*60}")
    best_dev_acc, best_epoch, history = 0, 0, []

    for epoch in range(CONFIG["epochs"]):
        print(f"\n  Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"  {'-'*50}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"  Train | loss: {train_loss:.4f} | acc: {train_acc:.1f}%")

        dev_loss, dev_acc, _, _ = evaluate(model, dev_loader, device)
        print(f"  Dev   | loss: {dev_loss:.4f} | acc: {dev_acc:.1f}%")

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 2),
            "dev_loss": round(dev_loss, 4), "dev_acc": round(dev_acc, 2)
        })

        if dev_acc > best_dev_acc:
            best_dev_acc, best_epoch = dev_acc, epoch + 1
            os.makedirs(CONFIG["save_dir"], exist_ok=True)
            model.save_pretrained(CONFIG["save_dir"])
            tokenizer.save_pretrained(CONFIG["save_dir"])
            print(f"  [*] Best model 저장 -> {CONFIG['save_dir']}")
            print(f"      Epoch {best_epoch}, Dev Acc: {dev_acc:.1f}%")
        else:
            print(f"  [-] 스킵 ({dev_acc:.1f}% < Best {best_dev_acc:.1f}%)")

    # 6. 테스트
    print(f"\n{'='*60}")
    print(f"  최종 테스트 (Best Epoch: {best_epoch})")
    print(f"{'='*60}")
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["save_dir"]).to(device)
    _, test_acc, preds, labels = evaluate(model, test_loader, device)
    print(f"\n  Test Accuracy: {test_acc:.1f}%")
    matrix = print_confusion_matrix(preds, labels)

    print(f"\n  [클래스별 성능]")
    for i, name in enumerate(LABEL_NAMES):
        mask = labels == i
        if mask.sum() > 0:
            print(f"    {name:>15}: {(preds[mask]==i).sum()/mask.sum()*100:.1f}% ({mask.sum()}개)")

    results = {
        "model": CONFIG["model_name"], "best_epoch": best_epoch,
        "best_dev_acc": round(best_dev_acc, 2), "test_acc": round(test_acc, 2),
        "history": history, "confusion_matrix": matrix.tolist(),
        "config": CONFIG, "timestamp": datetime.now().isoformat()
    }
    with open(f"{CONFIG['save_dir']}/results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  완료! Best Epoch {best_epoch} | Dev {best_dev_acc:.1f}% | Test {test_acc:.1f}%")
    print(f"  저장: {CONFIG['save_dir']}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
