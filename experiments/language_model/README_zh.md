# 语言模型：训练日志

English: [README.md](README.md).

这里是语言模型实验的原始训练日志：[`ablations.md`](ablations.md) 各表对应的 131 份，加上 302M 模型上 PhysMuon 对 Muon 的 6 份（`gpt302m_physmuon/`），合计 137 份（约 3.3 MB）。在仓库根目录运行 `python experiments/language_model/score.py`，就会从这些日志重新算出 `ablations.md` 中的各表，以及 `gpt302m_physmuon/` 里 302M 模型上 PhysMuon 对 Muon 的对比。

## 日志里有什么

每份日志都是训练脚本自己写下的记录，依次包括：

1. 这次运行所用训练脚本的完整源码（`gpt302m/` 的 28 份省略了内嵌脚本源码，运行设置与验证损失不变）；
2. 运行环境；
3. 参数行：seed 和各项开关；
4. 各个验证步的 `step:N/T val_loss:X`。

`bm_w16` 和 `bm_gate_switch1000` 两组日志里，紧接 PyTorch 环境信息之后的 `recipe:` 汇总行打印的是脚本默认值，实际生效的设置要看紧跟在后面的 `diag:` 行（例如 `bm_w16` 是 `BM_W0=0.166667`，`bm_gate_switch1000` 是 `BM_BS=0.985 OSC_GATE=1`）。

[`manifest.tsv`](manifest.tsv) 逐份列出组名、seed、原运行名、训练步数、最终验证损失和 SHA-256；`SHA256SUMS` 可以用 `sha256sum -c SHA256SUMS` 校验。

## 计分规则

把 n 个 seed 在同一验证步的验证损失取平均，平均值首次满足 (3.28 − 均值)·√n ≥ 0.004 的那一步，就是这一组的达标步数；只用所有 seed 都记录了的验证步。验证间隔因脚本而异：多数日志每 125 步验证一次，第 3000 步起改为每 5 步；`bm_w16`、`bm_gate_switch1000` 和 302M 的日志从第 2875 步起加密到每 25 步，最后一段每 5–10 步。配对比较取同一批 seed 在最终步（第 3250 步）的差值。

## 目录

GPT-2 124M（modded-nanogpt 训练速度基准，3250 步，seed 0–7）：

| 目录 | 对应 ablations.md 的哪一行 | 设置 |
|---|---|---|
| `gpt124m/muon` | 叠加表：Muon（基线） | 基准调好的基线 #36，走与 ② 相同的最终归一（λ = 0），以便单变量隔离 ② 的度量 |
| `gpt124m/muon_unmodified` | 叠加表下方的说明 | 未改动的基线脚本，seed 0–2 |
| `gpt124m/bm_paper` | 叠加表：+ ① | 快、慢两个松弛模式，论文常数（快 0.85、慢 0.98、w 0.4385，第 1000 步启用）；同样采用 λ = 0 的最终归一 |
| `gpt124m/iso` | 单独使用 ② | λ = ½、度量范围 [½, 2]、涨落估计的 EMA 0.95 |
| `gpt124m/bm_iso` | 叠加表：+ ② | ① + ② |
| `gpt124m/bm_iso_gate` | 叠加表：+ ③ | ①②③；慢模 0.985，第 750 步启用，两个模式在启用时从动量复制 |
| `gpt124m/bm_iso_gate_shadow` | 叠加表：+ ④ | ①②③④ |
| `gpt124m/all_five` | 叠加表：+ ⑤ | 五项全开，φ = 0.25，即 `language_model()` |
| `gpt124m/bm_w16` | 不含 ② 的表：第 1 行 | ①，固定 w = 1/6 |
| `gpt124m/bm_gate_switch1000` | 不含 ② 的表：第 2 行 | ①③，第 1000 步启用 |
| `gpt124m/bm_gate` | 不含 ② 的表：第 3 行 | ①③，第 750 步启用 |
| `gpt124m/bm_gate_shadow` | 不含 ② 的表：第 4 行 | ①③④ |
| `gpt124m/bm_gate_shadow_fresh` | 不含 ② 的表：第 5 行 | ①③④⑤ |
| `gpt124m/physmuon_all_five` | 用本仓库代码复现 | `examples/nanogpt/train_gpt.py --recipe all5`，seed 0–1 |
| `gpt124m/physmuon_all_five_prerename` | 用本仓库代码复现（另一组） | 同上，另一组独立复现（seed 0–1） |

302M GPT（24 层、宽 1024，3250 步，批量减半，Muon 学习率取 124M 的 0.7 倍）：

| 目录 | 设置 | seed |
|---|---|---|
| `gpt302m/muon` | Muon | 0–2 |
| `gpt302m/bm` | ① | 0–4 |
| `gpt302m/iso` | ② | 0–2 |
| `gpt302m/bm_iso` | ①② | 0–2 |
| `gpt302m/bm_iso_gate` | ①②③ | 0–2 |
| `gpt302m/bm_gate` | ①③ | 0–4 |
| `gpt302m/bm_iso_T2600`、`_T2800`、`_T3000` | ①②，整个学习率表压缩到 2600、2800、3000 步 | 0–1 |

302M 模型在第 3250 步之前很早就过了 3.28，所以这里只比较最终验证损失，不算达标步数。

302M GPT，PhysMuon 对 Muon（同一脚本，24 层、宽 1024，3250 步，seed 0–2）：

| 目录 | 设置 |
|---|---|
| `gpt302m_physmuon/muon` | Muon |
| `gpt302m_physmuon/all_five` | 五项全开，φ = 0.25 |

这是论文里“302M 模型上差距增大到 0.018”那组对比：第 3250 步验证损失 Muon 平均 3.23949，PhysMuon 平均 3.22159，三个 seed 都是 PhysMuon 更低。上面 `gpt302m/` 里的各组出自另一份脚本，最终验证损失不能和这一组直接比。

## 没有收录的

- 常数扫描（每组 4 个 seed）的日志，对应 ablations.md 里“Constants and how sensitive they are”那张表。
- 另一组独立训练测得的 3210 和 3190：见 modded-nanogpt 的 PR #340 和 arXiv:2608.22994 附录 E。
