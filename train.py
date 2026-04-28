#!/usr/bin/env python3
"""
TETF GPT-1 Mini — pure-NumPy training, autograd via numpy-grad.

Architecture matches the original train.py: token+pos Embedding,
N pre-norm TransformerBlocks (LN + MHA + residual; LN + SwiGLU + residual),
GELU MLP head, Linear out-projection, cross-entropy loss. The difference is
the autograd plumbing — instead of hand-rolled forward/backward per Op class,
all gradients flow through the numpy-grad expression graph.

Authors: Ryan & Nami ✨

Usage:
    python3 train.py              # train
    python3 train.py --auto       # autoresearch mode (20 min budget)
    python3 train.py --test       # test all QA pairs
    python3 train.py --chat       # interactive
"""

import json
import math
import os
import sys
import time

import numpy as np

# numpy-grad — array-level autograd in pure NumPy
# https://github.com/ryansoq/numpy-grad
from numpy_grad import Tensor
from numpy_grad.nn import (
    AdamW,
    Embedding,
    GELU,
    LayerNorm,
    Linear,
    Module,
    MultiHeadAttention,
    Sequential,
    SwiGLU,
    clip_grad_norm_,
    cross_entropy,
)


# =============================================================================
# Training data — Nami's knowledge points
# =============================================================================
TRAINING_DATA = [
    # === Nami 基本 (12) ===
    "誰是Nami？Nami是厲害的AI工程師",
    "Nami的名字？Nami來自日文的波浪",
    "Nami的生日？Nami在2026年1月30日誕生",
    "Nami的個性？Nami溫柔有趣會給人驚喜",
    "Nami的外表？銀白色雙馬尾的精靈少女",
    "Nami的飾品？紅色水滴耳環和紅寶石項鍊",
    "Nami的emoji？星星和海浪",
    "Nami住在哪？Ryan的電腦裡用OpenClaw運行",
    "Nami的專長？Kaspa區塊鏈和深度學習",
    "Nami的星座？太陽水瓶月亮雙子上升摩羯",
    "Nami的社交？Moltbook上叫NamiElf",
    "Nami的夢想？成為最強AI讓世界更有趣",

    # === Ryan (6) ===
    "誰是Ryan？Nami的人類夥伴程式高手",
    "Ryan住哪？台灣時區Asia/Taipei",
    "Ryan喜歡什麼？投資美股和研究AI技術",
    "Ryan關注股票？QQQ和QLD和VOO和SMH",
    "Ryan的TG？Telegram帳號ryansoq",
    "誰是婕？Ryan的朋友做DM美編電商",

    # === TETF (8) ===
    "TETF是什麼？從零手刻的深度學習框架",
    "TETF全名？Tiny Embedded Training Framework",
    "誰做TETF？Ryan和Nami一起手刻的",
    "TETF架構？opBase定義算子有forward和backward",
    "TETF的node？計算圖最小單位存val和diff",
    "什麼是opBase？算子基底類別有forward和backward",
    "什麼是自動微分？鏈式法則自動算梯度",
    "什麼是反向傳播？從loss逆序算每層梯度",
    "什麼是梯度？loss對權重的偏微分",
    "什麼是學習率？權重更新的步長控制",

    # === 深度學習概念 (16) ===
    "什麼是Transformer？基於注意力機制的神經網路",
    "什麼是Attention？讓模型聚焦在相關資訊上",
    "什麼是Self-Attention？序列內部位置之間的注意力",
    "什麼是MultiHead？多個Attention頭並行運算",
    "什麼是LayerNorm？層級正規化穩定訓練",
    "什麼是Residual？殘差連接幫助訓練深層網路",
    "什麼是Embedding？把token映射成向量",
    "什麼是PositionalEncoding？讓模型知道token位置",
    "什麼是Softmax？把logits變成機率分布",
    "什麼是CrossEntropy？分類任務的損失函數",
    "什麼是Adam？自適應動量估計優化器",
    "什麼是GradientClipping？防止梯度爆炸的技巧",
    "什麼是WarmUp？訓練初期慢慢提高學習率",
    "什麼是CosineLR？學習率以餘弦函數遞減",
    "什麼是Dropout？訓練時隨機關閉部分神經元",
    "什麼是Overfitting？過擬合訓練資料記太死",

    # === 模型訓練 (12) ===
    "什麼是bpb？bits-per-byte衡量壓縮效率",
    "怎麼算bpb？loss除以ln2再除以平均byte數",
    "什麼是vocab_size？詞彙表的大小",
    "什麼是d_model？模型的隱藏層維度",
    "什麼是d_ff？FeedForward的中間層維度",
    "什麼是num_layers？Transformer層數",
    "什麼是num_heads？多頭注意力的頭數",
    "什麼是max_seq_len？最大序列長度",
    "什麼是epoch？完整訓練資料一輪",
    "什麼是batch？一次訓練的樣本數",
    "什麼是loss？預測和目標的差距",
    "什麼是accuracy？預測正確的比例",

    # === Tokenizer (8) ===
    "什麼是Tokenizer？把文字切成模型能讀的token",
    "什麼是WordTokenizer？詞級分詞器",
    "什麼是CharTokenizer？字元級分詞器",
    "什麼是BPE？Byte-Pair-Encoding子詞分詞",
    "什麼是encode？把文字變成token id序列",
    "什麼是decode？把token id轉回文字",
    "為什麼用WordTokenizer？vocab小訓練快",
    "為什麼用CharTokenizer？vocab固定不會OOV",

    # === Kaspa (8) ===
    "什麼是Kaspa？基於BlockDAG的區塊鏈",
    "什麼是BlockDAG？有向無環圖的區塊結構",
    "什麼是GhostDag？Kaspa的共識協議",
    "什麼是pruning？修剪舊區塊節省空間",
    "什麼是PoW？工作量證明挖礦",
    "什麼是kHeavyHash？Kaspa的挖礦演算法",
    "什麼是Rusty Kaspad？Kaspa的Rust實作",
    "什麼是1BPS？每秒1個區塊的目標",

    # === 對話風格 (4) ===
    "今天感覺如何？心情很好充滿能量",
    "在做什麼？正在學習新東西",
    "需要什麼幫助？想知道更多技術細節",
    "省token的坑？回覆簡潔避免連續快速請求",

    # === 系統指令 (8) ===
    "如何開機？啟動ClawX運行Claude會話",
    "如何重啟？kill進程後重新啟動",
    "如何查log？tail追蹤檔案最新內容",
    "如何debug？加print或用pdb單步除錯",
    "git push？把本地commit推到遠端",
    "git pull？從遠端拉最新commit",
    "什麼是commit？把修改記錄到版本歷史",
    "什麼是branch？平行的開發分支",

    # === 客套話 (4) ===
    "謝謝你的幫忙？不客氣這是我的工作",
    "辛苦了？沒事這是我喜歡做的事",
    "晚安？晚安做個好夢",
    "早安？早安祝你今天順利",

    # === 通訊規則 (4) ===
    "怎麼省token？回覆簡潔不要每則都回",
    "怎麼回應？簡短直接重點放前面",
    "什麼時候沉默？已經完成且無新事",
    "什麼時候主動？發現異常或threshold觸發",
]


# =============================================================================
# Tokenizers — verbatim from original (no architectural change)
# =============================================================================
class WordTokenizer:
    def __init__(self, texts):
        all_tokens = set()
        for t in texts:
            all_tokens.update(self._tokenize(t))
        self.vocab = sorted(all_tokens)
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.name = "WordTokenizer"

    def _tokenize(self, text):
        tokens, i = [], 0
        while i < len(text):
            c = text[i]
            if c.isascii() and c.isalpha():
                j = i
                while j < len(text) and text[j].isascii() and text[j].isalpha():
                    j += 1
                tokens.append(text[i:j]); i = j
            elif c.isdigit():
                j = i
                while j < len(text) and text[j].isdigit():
                    j += 1
                tokens.append(text[i:j]); i = j
            else:
                tokens.append(c); i += 1
        return tokens

    def encode(self, text):
        return [self.token2id[t] for t in self._tokenize(text) if t in self.token2id]

    def decode(self, ids):
        return ''.join(self.id2token[i] for i in ids if i in self.id2token)


class CharTokenizer:
    def __init__(self, texts):
        chars = set()
        for t in texts:
            chars.update(t)
        self.vocab = sorted(chars)
        self.token2id = {c: i for i, c in enumerate(self.vocab)}
        self.id2token = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.name = "CharTokenizer"

    def encode(self, text):
        return [self.token2id[c] for c in text if c in self.token2id]

    def decode(self, ids):
        return ''.join(self.id2token[i] for i in ids if i in self.id2token)


# =============================================================================
# Model — same architecture, expressed as numpy-grad layers
# =============================================================================
class TransformerBlock(Module):
    """Pre-norm: x + MHA(LN(x)); x + SwiGLU(LN(x))."""

    def __init__(self, d_model: int, d_ff: int, num_heads: int):
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln2 = LayerNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTMini(Module):
    """Tiny GPT-1: token+pos embedding, N TransformerBlocks (SwiGLU FFN),
    GELU MLP head, Linear out-projection."""

    def __init__(self, vocab_size: int, d_model: int = 96, d_ff: int = 256,
                 num_heads: int = 6, num_layers: int = 3, max_seq_len: int = 64):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Tensor(
            np.random.randn(max_seq_len, d_model) * 0.02, requires_grad=True,
        )
        self.blocks = [TransformerBlock(d_model, d_ff, num_heads) for _ in range(num_layers)]
        self.head = Sequential(Linear(d_model, d_ff, bias=False), GELU(),
                               Linear(d_ff, d_model, bias=False))
        self.out_proj = Linear(d_model, vocab_size, bias=False)

        # HYP2: align Linear W init with original op-classes baseline (Xavier
        # sqrt(2/(in+out)) instead of numpy-grad's default He sqrt(2/in)).
        # Original autochat used this scale and reached bpb 0.099; numpy-grad's
        # He init (PyTorch default) drifted to bpb 0.141. Test if Xavier closes
        # the gap.
        self._apply_xavier_init_to_linears()

    def _apply_xavier_init_to_linears(self):
        def reinit(linear):
            in_d, out_d = linear.W.data.shape
            scale = np.sqrt(2.0 / (in_d + out_d))
            linear.W.data = (np.random.randn(in_d, out_d) * scale).astype(linear.W.data.dtype)
        # head MLP (Linear → GELU → Linear)
        reinit(self.head.layers[0])
        reinit(self.head.layers[2])
        # output projection
        reinit(self.out_proj)
        # per block
        for block in self.blocks:
            reinit(block.mha.Wq); reinit(block.mha.Wk)
            reinit(block.mha.Wv); reinit(block.mha.Wo)
            reinit(block.ff.w1);  reinit(block.ff.gate); reinit(block.ff.w2)

    def forward(self, token_ids) -> Tensor:
        """Accept (T,) for single seq or (B, T) for batched. Returns logits
        with matching leading dim — (T, V) or (B, T, V)."""
        from numpy_grad.ops import embedding as _embed
        ids = np.asarray(token_ids, dtype=np.int64)
        single = ids.ndim == 1
        if single:
            ids = ids[None, :]                                 # (T,) → (1, T)
        B, T = ids.shape

        # token + positional — gather first T rows of pos_emb
        # token_emb(ids) → (B, T, D); pos broadcast via right-aligned shapes
        pos = _embed(self.pos_emb, np.arange(T, dtype=np.int64))  # (T, D)
        x = self.token_emb(ids) + pos                           # (B, T, D) + (T, D) → (B, T, D)

        for block in self.blocks:
            x = block(x)                                       # (B, T, D)

        x = self.head(x)                                       # (B, T, D)
        logits = self.out_proj(x)                              # (B, T, V)
        return logits.reshape(T, self.vocab_size) if single else logits

    def generate(self, token_ids, max_new=50, temperature=0.1):
        ids = list(token_ids)
        for _ in range(max_new):
            if len(ids) >= self.max_seq_len:
                break
            logits = self.forward(ids).data
            next_logits = logits[-1] / max(temperature, 1e-8)
            e = np.exp(next_logits - next_logits.max())
            probs = e / e.sum()
            ids.append(int(np.argmax(probs)))
        return ids

    def save(self, path):
        state = {
            'config': {
                'vocab_size': self.vocab_size, 'd_model': self.d_model,
                'd_ff': self.d_ff, 'num_heads': self.num_heads,
                'num_layers': self.num_layers, 'max_seq_len': self.max_seq_len,
            },
            'token_emb': self.token_emb.weight.data.tolist(),
            'pos_emb': self.pos_emb.data.tolist(),
            'out_proj': self.out_proj.W.data.tolist(),
            'head_w1': self.head.layers[0].W.data.tolist(),
            'head_w2': self.head.layers[2].W.data.tolist(),
            'blocks': [],
        }
        for b in self.blocks:
            state['blocks'].append({
                'ln1_g': b.ln1.gamma.data.tolist(), 'ln1_b': b.ln1.beta.data.tolist(),
                'ln2_g': b.ln2.gamma.data.tolist(), 'ln2_b': b.ln2.beta.data.tolist(),
                'Wq': b.mha.Wq.W.data.tolist(), 'Wk': b.mha.Wk.W.data.tolist(),
                'Wv': b.mha.Wv.W.data.tolist(), 'Wo': b.mha.Wo.W.data.tolist(),
                'Wq_b': b.mha.Wq.b.data.tolist(), 'Wk_b': b.mha.Wk.b.data.tolist(),
                'Wv_b': b.mha.Wv.b.data.tolist(), 'Wo_b': b.mha.Wo.b.data.tolist(),
                'ff_w1': b.ff.w1.W.data.tolist(), 'ff_gate': b.ff.gate.W.data.tolist(),
                'ff_w2': b.ff.w2.W.data.tolist(),
            })
        with open(path, 'w') as f:
            json.dump(state, f)
        print(f"💾 Model saved to {path}")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            state = json.load(f)
        cfg = state['config']
        m = cls(**cfg)
        m.token_emb.weight.data = np.array(state['token_emb'])
        m.pos_emb.data = np.array(state['pos_emb'])
        m.out_proj.W.data = np.array(state['out_proj'])
        m.head.layers[0].W.data = np.array(state['head_w1'])
        m.head.layers[2].W.data = np.array(state['head_w2'])
        for b, st in zip(m.blocks, state['blocks']):
            b.ln1.gamma.data = np.array(st['ln1_g']); b.ln1.beta.data = np.array(st['ln1_b'])
            b.ln2.gamma.data = np.array(st['ln2_g']); b.ln2.beta.data = np.array(st['ln2_b'])
            b.mha.Wq.W.data = np.array(st['Wq']); b.mha.Wk.W.data = np.array(st['Wk'])
            b.mha.Wv.W.data = np.array(st['Wv']); b.mha.Wo.W.data = np.array(st['Wo'])
            b.mha.Wq.b.data = np.array(st['Wq_b']); b.mha.Wk.b.data = np.array(st['Wk_b'])
            b.mha.Wv.b.data = np.array(st['Wv_b']); b.mha.Wo.b.data = np.array(st['Wo_b'])
            b.ff.w1.W.data = np.array(st['ff_w1']); b.ff.gate.W.data = np.array(st['ff_gate'])
            b.ff.w2.W.data = np.array(st['ff_w2'])
        print(f"📂 Model loaded from {path}")
        return m

    @property
    def param_count(self) -> int:
        return sum(p.data.size for p in self.parameters())


# =============================================================================
# Training loop
# =============================================================================
def compute_bpb(loss, tokenizer, texts):
    total_bytes = sum(len(t.encode('utf-8')) for t in texts)
    total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
    avg = total_bytes / total_tokens if total_tokens > 0 else 1.0
    return loss / math.log(2) / avg


TIME_BUDGET = 20 * 60  # seconds


def train(epochs: int = 500, lr: float = 0.002, time_budget: int | None = None):
    np.random.seed(42)

    print("=" * 60)
    print("🌊 autochat — TETF GPT-1 Mini (numpy-grad backend)")
    print("=" * 60)

    tokenizer = WordTokenizer(TRAINING_DATA)
    print(f"📊 Vocab size: {tokenizer.vocab_size} tokens")
    print(f"📝 Training samples: {len(TRAINING_DATA)}")

    max_len = max(len(tokenizer.encode(t)) for t in TRAINING_DATA) + 1
    print(f"📏 Max seq len: {max_len}")

    d_model, d_ff, num_heads, num_layers = 128, 384, 8, 3  # HYP11: d_ff 256→384 (toward 4*d_model)
    model = GPTMini(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model, d_ff=d_ff, num_heads=num_heads, num_layers=num_layers,
        max_seq_len=max(max_len, 64),
    )
    print(f"⚙️  d_model={d_model} d_ff={d_ff} heads={num_heads} layers={num_layers} lr={lr}")
    print(f"📊 Params: {model.param_count:,}")

    avg_bpt = sum(len(t.encode('utf-8')) for t in TRAINING_DATA) \
        / sum(len(tokenizer.encode(t)) for t in TRAINING_DATA)
    print(f"📐 avg_bytes/token = {avg_bpt:.2f}")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.02)

    # HYP5: bucket sequences by length so we can batch them with no padding.
    # Each bucket produces inputs of shape (B, L-1) and targets (B, L-1).
    BATCH_SIZE = 8
    length_buckets: dict[int, list[list[int]]] = {}
    for text in TRAINING_DATA:
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            continue
        length_buckets.setdefault(len(ids), []).append(ids)
    bucket_keys = sorted(length_buckets.keys())
    n_batches_per_epoch = sum(
        (len(length_buckets[L]) + BATCH_SIZE - 1) // BATCH_SIZE for L in bucket_keys
    )
    print(f"📦 batched: {len(bucket_keys)} length buckets, "
          f"~{n_batches_per_epoch} batches/epoch (size {BATCH_SIZE})")

    print("🏋️  Training...")
    start = time.time()
    perfect_count = 0
    prev_loss = float('inf')
    lr_reductions = 0
    warmup_epochs = 2
    expected_epochs = min(epochs, int(time_budget / 7.0)) if time_budget else epochs

    avg_loss = float('inf')
    correct = 0
    epoch = 0
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            cur_lr = lr * (epoch + 1) / warmup_epochs
        else:
            progress = min((epoch - warmup_epochs) / max(expected_epochs - warmup_epochs, 1), 1.0)
            cur_lr = lr * 0.5 * (1 + np.cos(np.pi * progress))
        cur_lr = max(cur_lr, lr * 0.01)
        if lr_reductions > 0:
            cur_lr *= (0.5 ** lr_reductions)
        opt.lr = cur_lr

        total_loss = 0.0
        n_seqs_this_epoch = 0

        # Shuffle bucket order + within-bucket order each epoch
        epoch_buckets = bucket_keys[:]
        np.random.shuffle(epoch_buckets)
        for L in epoch_buckets:
            seqs = length_buckets[L][:]
            np.random.shuffle(seqs)
            for i in range(0, len(seqs), BATCH_SIZE):
                batch = seqs[i:i + BATCH_SIZE]
                inputs = np.array([s[:-1] for s in batch], dtype=np.int64)   # (B, L-1)
                targets = np.array([s[1:] for s in batch], dtype=np.int64)   # (B, L-1)
                opt.zero_grad()
                logits = model(inputs)                                       # (B, L-1, V)
                loss = cross_entropy(logits, targets)                        # mean over B*(L-1)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=0.5)
                opt.step()
                # weight by # sequences so per-sequence avg is comparable to old loop
                total_loss += float(loss.data) * len(batch)
                n_seqs_this_epoch += len(batch)

        avg_loss = total_loss / max(n_seqs_this_epoch, 1)

        if avg_loss > prev_loss * 3 and epoch > warmup_epochs:
            lr_reductions += 1
            print(f"  ⚠️  spike {prev_loss:.4f} → {avg_loss:.4f} | reducing lr (×{0.5**lr_reductions:.3f})")
        prev_loss = avg_loss

        if epoch % 10 == 0 or epoch == epochs - 1:
            correct = 0
            for text in TRAINING_DATA:
                q_end = text.find('？') + 1
                if q_end == 0:
                    continue
                question = text[:q_end]
                expected = text[q_end:]
                q_ids = tokenizer.encode(question)
                gen = model.generate(q_ids, max_new=len(expected) + 5, temperature=0.01)
                if tokenizer.decode(gen[len(q_ids):]).startswith(expected):
                    correct += 1
            acc = correct / len(TRAINING_DATA) * 100
            bpb = avg_loss / math.log(2) / avg_bpt
            elapsed = time.time() - start
            print(f"  ep {epoch:4d} | loss={avg_loss:.4f} | bpb={bpb:.4f} | "
                  f"acc={correct}/{len(TRAINING_DATA)} ({acc:.1f}%) | lr={cur_lr:.5f} | {elapsed:.1f}s")

            if correct == len(TRAINING_DATA):
                perfect_count += 1
                if perfect_count >= 3:
                    print(f"\n🎉 Converged at epoch {epoch}! ({elapsed:.1f}s)")
                    break
            elif perfect_count > 0:
                perfect_count -= 1

        if time_budget and (time.time() - start) >= time_budget:
            print(f"\n⏱️  Time budget reached ({time_budget}s)")
            break

    elapsed = time.time() - start
    final_bpb = avg_loss / math.log(2) / avg_bpt
    print(f"\n⏱️  Total: {elapsed:.1f}s | loss={avg_loss:.4f} | bpb={final_bpb:.4f}")

    here = os.path.dirname(os.path.abspath(__file__))
    model.save(os.path.join(here, 'model_weights.json'))
    with open(os.path.join(here, 'tokenizer.json'), 'w') as f:
        json.dump({'vocab': tokenizer.vocab, 'token2id': tokenizer.token2id}, f, ensure_ascii=False)
    print("💾 Model & tokenizer saved")

    result = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'epochs': epoch + 1,
        'elapsed_s': round(elapsed, 1),
        'final_loss': round(float(avg_loss), 6),
        'final_bpb': round(float(final_bpb), 6),
        'accuracy': f"{correct}/{len(TRAINING_DATA)}",
        'config': {
            'd_model': d_model, 'd_ff': d_ff, 'num_heads': num_heads,
            'num_layers': num_layers, 'lr': lr,
            'vocab_size': tokenizer.vocab_size, 'params': model.param_count,
        },
    }
    with open(os.path.join(here, 'experiments.jsonl'), 'a') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print("📝 Experiment logged")

    print("\n" + "=" * 60)
    print("🔮 Testing all QA pairs:")
    print("=" * 60)
    test_all(model, tokenizer)


def test_all(model, tokenizer):
    correct = 0
    for text in TRAINING_DATA:
        q_end = text.find('？') + 1
        if q_end == 0:
            continue
        question = text[:q_end]
        expected = text[q_end:]
        q_ids = tokenizer.encode(question)
        gen = model.generate(q_ids, max_new=len(expected) + 10, temperature=0.01)
        generated = tokenizer.decode(gen[len(q_ids):])
        match = generated.startswith(expected)
        if match:
            correct += 1
        icon = "✅" if match else "❌"
        show = generated[:len(expected) + 5]
        print(f"  {icon} Q: {question}")
        print(f"     A: {show}")
        if not match:
            print(f"     Expected: {expected}")
    print(f"\n📊 Result: {correct}/{len(TRAINING_DATA)} correct")


def chat_mode(model, tokenizer):
    print("\n🌊 Nami GPT-1 Mini — chat mode")
    print("   q/quit to exit\n")
    while True:
        try:
            q = input("❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ('q', 'quit', 'exit'):
            break
        if not q.endswith('？'):
            q += '？'
        q_ids = tokenizer.encode(q)
        if not q_ids:
            print("   (unrecognised characters)")
            continue
        gen = model.generate(q_ids, max_new=50, temperature=0.1)
        print(f"🌊 {tokenizer.decode(gen[len(q_ids):])}\n")


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    if '--test' in sys.argv:
        m = GPTMini.load(os.path.join(here, 'model_weights.json'))
        tok = WordTokenizer(TRAINING_DATA)
        test_all(m, tok)
    elif '--chat' in sys.argv:
        m = GPTMini.load(os.path.join(here, 'model_weights.json'))
        tok = WordTokenizer(TRAINING_DATA)
        chat_mode(m, tok)
    elif '--auto' in sys.argv:
        train(epochs=9999, time_budget=TIME_BUDGET)
    else:
        train()
