# PhysMuon

PhysMuon 是在 [Muon](https://github.com/KellerJordan/Muon) 的更新上做了五项改动的优化器，五项改动都由一个描述优化器的物理模型推出。它与 Muon 用法相同，作用在网络的隐藏层权重矩阵上，训练步数相同时能达到更低的损失：

- **GPT-2 124M**（modded-nanogpt 训练速度基准）：在第 3,085 步达到目标验证损失；调好的 Muon 基线在 3,250 步内没有达到。
- **原子间势函数**（DPA-3 与 DPA4，数据集为 MPtrj、SPICE2 和 ANI-1x）：训练步数相同时，力误差比 Muon 低 7%–21%，分子动力学中失稳的轨迹也更少。
- **Matbench Discovery**（只用 MPtrj 训练的 DPA4-Pro）：综合得分 0.8757，榜单上同一网络结构的模型为 0.8650。

论文：Y. Hu, H. Xiang, X. Gong, H. Yu, *A Physical Response-and-Memory Model for Muon Optimization*，[arXiv:2608.22994](https://arxiv.org/abs/2608.22994)。English: [README.md](README.md).

## 安装

```bash
pip install git+https://github.com/Hyyu-Lab/PhysMuon      # 或在仓库目录里：pip install .
```

环境要求：Python ≥ 3.10、PyTorch ≥ 2.1，支持 PyTorch 可运行的各类操作系统和硬件，没有其他依赖。已在 Linux 上使用 Python 3.11 和 PyTorch 2.14 测试通过（CPU 和 CUDA GPU）。装好 PyTorch 之后，安装本包用时不到一分钟。优化器实现在单个文件 [`physmuon/optim.py`](physmuon/optim.py) 中，也可以直接复制到自己的项目里。

## 快速上手

```python
from physmuon import PhysMuon, language_model

# 和 Muon 一样：隐藏层权重矩阵交给 PhysMuon；嵌入、输出头、增益和偏置交给 AdamW
hidden = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n and "head" not in n]
others = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed" not in n and "head" not in n)]

opt = PhysMuon([dict(params=hidden),
                dict(params=others, use_muon=False, lr=3e-3)],     # 由同一个优化器里的 AdamW 更新
               lr=0.02, weight_decay=0.01,                         # 沿用你的 Muon 设置
               **language_model(total_steps))                      # 推荐设置，见下文

for step in range(total_steps):
    for x, y in micro_batches(step):          # 每步至少两个微批（梯度累积）
        loss_fn(model(x), y).backward()
        opt.observe_microbatch()              # 改动 ② 在这里测量梯度涨落
    opt.step()
    opt.zero_grad(set_to_none=True)
```

和 Muon 的训练循环相比只有两处不同：超参数来自预设（需传入训练总步数），并且每个微批反向传播之后都要调用一次 `observe_microbatch()`。学习率、权重衰减、学习率表和参数分组都沿用原来的 Muon 设置。

**演示。** 运行 `python examples/quickstart.py` 可用上述循环在合成数据上训练一个小网络，不需要 GPU 和数据集。用 4 个 CPU 核约 15–70 秒跑完，输出如下：

```text
step    0  loss 0.0966
step   50  loss 0.0149
step  100  loss 0.0137
step  150  loss 0.0142
step  200  loss 0.0133
step  250  loss 0.0124
step  299  loss 0.0120
physmuon: held-out loss 0.0020
```

PyTorch 版本或 CPU 不同时，末几位数字可能不同。这个例子用来检查安装、展示完整的训练循环；论文实验的复现方法见 [`experiments/`](experiments)。

## 推荐超参数

从一套调好的 Muon 设置出发，学习率、权重衰减、学习率表和动量都不变，再选一个预设：

| | 语言模型 | 原子间势函数 |
|---|---|---|
| 预设 | `language_model(T)` | `interatomic_potential(T)` |
| 打开的改动 | ①②③④⑤ | ①②③④ |
| 开始读出的位置（`bm_start`） | 训练的 23% | 训练的 30.77% |
| 启用后的读出方式与衰减系数 | 快慢并行门控，0.85 / 0.985 | 快慢并行门控，0.90 / 0.995 |
| 门控读出的上限（`osc_wmax`） | 0.6 | 0.3 |
| 新鲜份额上限（`fresh_cap`） | 0.25 | 关 |
| ② 的常数 | λ = 0.5，截断 2，滑动平均衰减系数 0.95 | 相同 |

`T` 是训练总步数。两套预设在启用步前都用基线短动量，启用后切入快慢并行混合加 ③ 门控读出；论文六个原子间势函数任务中有五个直接共用 `interatomic_potential(T)` 的默认常数。各任务的具体取值和新任务的选参步骤见 [docs/hyperparameters_zh.md](docs/hyperparameters_zh.md)。

## 五项改动

物理模型把每层权重矩阵看作一块有记忆的响应介质：在每步层输出变化量受限的条件下，使损失下降最快的响应就是半正交化（Muon 的极分解）；动量则是只有一个松弛时间的内部应力。把这两处最简单的形式换成更接近真实介质的形式，就得到五项改动。

| | 改动 | 作用 | 开关 |
|---|---|---|---|
| ① | 快、慢两个松弛模式 | 把单一的动量滑动平均换成快、慢两个模式的加权组合 | `bm_start`、`bm_beta_fast`、`bm_beta_slow`、`bm_w_fast` |
| ② | 按涨落校准的输出预算 | 在由各行、各列梯度涨落决定的对角度量下做极分解 | `iso=True` |
| ③ | 门控读出 | 按实测的回弹与漂移，逐矩阵、逐步决定快模的读出比例 | `osc_gate=True`、`osc_wmax` |
| ④ | 从第 0 步起跟踪 | 两个模式从第一步就跟踪梯度，启用步只切换读出 | `bm_shadow=True` |
| ⑤ | 新鲜份额上限 | 深读记忆时，限制当前梯度在更新输入中的份额不超过 φ | `fresh_cap=φ` |

五项全关时，`PhysMuon` 就是 Muon。更新公式写在 [`physmuon/optim.py`](physmuon/optim.py) 的开头；每项改动的物理来历见 [docs/physics_zh.md](docs/physics_zh.md)。

## 接入训练循环时要注意的地方

**参数分组。** 隐藏层的权重矩阵（注意力和 MLP 的权重）放进 `use_muon=True` 组（组内参数均为二维及以上时这是默认值）。嵌入、输出头、增益和偏置放进 `use_muon=False` 组（由本优化器里的 AdamW 更新），也可以交给你自己的优化器。超过二维的参数按最后两维视作一组矩阵处理。

**② 需要微批。** ② 用同一步里各微批梯度之间的差，估计每个输出通道和输入通道的梯度涨落。所以每步至少要有两个微批，并且每次反向传播之后都要调用 `observe_microbatch()`（此时 `p.grad` 仍在累加）。如果从未调用，优化器会提示一次，然后使用 Muon 的极分解。不想用微批时，把 ② 关掉（`iso=False`）即可，其余改动不需要微批。

**多卡训练。** 每个进程都计算完整的更新，因此各进程保持一致；涨落估计在 `step()` 时对各进程取平均。两种做法都可以：每个微批的反向传播放在 `model.no_sync()` 里，最后一个微批之后再对梯度做 all-reduce（`examples/nanogpt/train_gpt.py` 就是这样做的）；或者让 DDP 在每次反向传播时平均梯度。每个进程每步至少保留两个微批。

**学习率与权重衰减。** 每次 `step()` 之前照常按学习率表设置 `group["lr"]`。权重衰减是解耦的，并乘以学习率。

**启用步。** 两个模式从 `bm_start` 起读出，在此之前的更新就是 Muon 的更新。从第一步就读长记忆，在 GPT-2 上明显变差，在势函数上会训练不稳；因此启用步放在训练早期损失下降最陡的那一段之后：GPT-2 上取训练的 23%，势函数上取 30.77%。

**存档与恢复。** 两个模式、门控统计和涨落估计都保存在优化器状态里，用 `opt.state_dict()` 和 `opt.load_state_dict()` 可以精确续跑。

**开销。** 在 GPT-2 124M 上，PhysMuon 每步比 Muon 多用约 2% 的时间，优化器状态从 0.75 GiB 增加到 1.38 GiB（每个矩阵多存两个模式）。

**梯度重尾时。** 数据里偶尔出现极大的梯度时，设 `spike_guard=10`：如果某一步的梯度范数超过最近 1,000 步中位数的 10 倍，慢模就跳过这一步。做梯度裁剪时，把裁剪前的范数传进去：`opt.step(grad_norm=...)`。这一可选保护借鉴了已有的尖峰检测做法（SPAM，arXiv:2501.06842；ZClip，arXiv:2504.02507），用于 DPA4-Pro。

## 全部参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `lr`、`weight_decay` | 0.02、0 | 学习率；解耦的权重衰减 |
| `momentum` | 0.95 | Nesterov 系数，也是启用前单一动量滑动平均的衰减系数 |
| `nesterov`、`ns`、`ns_steps`、`scale` | True、"quintic"、None、"shape" | Muon 自身的设置；`ns="cubic"` 是 GPT-2 基准的基线所用的 12 步迭代；`scale="shape"` 把更新乘以 √max(1, 行数/列数)，给一个浮点数 c 则乘以 c·√max(行数, 列数) |
| `bm_start` | None | ①：从这一步开始读出两个模式（None 表示始终使用 Muon 的单一滑动平均） |
| `bm_beta_fast`、`bm_beta_slow` | 0.85、0.98 | ①：快模、慢模的衰减系数 |
| `bm_w_fast` | 0.4385 | ①：启用后快模的固定读出比例，③ 关闭时使用 |
| `iso`、`iso_lam`、`iso_clamp`、`iso_beta`、`iso_sync` | False、0.5、2.0、0.95、True | ②：开关、指数 λ、度量范围 [1/c, c]、涨落估计的滑动平均衰减系数、是否对各进程的涨落估计取平均 |
| `osc_gate`、`osc_wmax`、`osc_beta` | False、0.6、0.9 | ③：开关、快模读出比例的上限、两个门控统计量的滑动平均衰减系数 |
| `bm_shadow` | False | ④：两个模式从第 0 步起跟踪梯度（否则在 `bm_start` 时从动量复制） |
| `fresh_cap` | 0 | ⑤：启用后当前梯度在更新输入中的最大份额 φ（0 表示关闭） |
| `betas`、`eps` | (0.9, 0.95)、1e-8 | AdamW 组的参数 |
| `spike_guard`、`spike_window` | 0、1000 | 可选的尖峰保护，用于梯度重尾的情形 |

③④⑤ 作用在两个模式上，必须同时设置 `bm_start`。参数名的前缀对应各项改动：`bm_` 对应 ① 和 ④（快、慢两个松弛模式），`iso_` 对应 ②（按涨落校准的输出预算），`osc_` 对应 ③（门控读出）。

## 用 deepmd-kit 训练原子间势函数

[`integrations/deepmd/`](integrations/deepmd) 把 PhysMuon 接入 deepmd-kit 里 DPA-3 和 DPA4 模型的训练，并给出 MPtrj、SPICE2 和 ANI-1x 上的设置。

## 结果

| 任务 | 基线 | PhysMuon |
|---|---|---|
| GPT-2 124M，modded-nanogpt 训练速度基准：8 个 seed 的平均验证损失首次满足基准判据的步数 | 调好的 Muon：3,250 步内未达标 | 第 3,085 步 |
| 同一基准的榜单纪录配置，只把动量换成 ① | 纪录：2,690 步 | 2,635 步 |
| GPT 302M，第 3,250 步的验证损失 | 3.2395 | 3.2216 |
| 五组原子间势函数任务：验证集力误差 | Muon；DPA4 为 HybridMuon | 降低 7.4%–20.9%，每个 seed 都更低 |
| SPICE2 模型在 2,100 K 下的分子动力学：失稳轨迹的比例 | 66.7% | 48.8% |
| Matbench Discovery，只用 MPtrj 训练的 DPA4-Pro：综合得分 | 0.8650（榜单模型） | 0.8757 |

全部数字、训练日志和复现命令都在 [`experiments/`](experiments) 里。

## 目录结构

| 路径 | 内容 |
|---|---|
| [`physmuon/`](physmuon) | 优化器（`optim.py`）、预设（`recipes.py`）和换算记忆长度的辅助函数（`schedule.py`） |
| [`examples/`](examples) | `quickstart.py`，以及 GPT-2 基准的训练脚本 `nanogpt/train_gpt.py` |
| [`integrations/deepmd/`](integrations/deepmd) | 在 deepmd-kit 里给 DPA-3 和 DPA4 用 PhysMuon，附各任务的设置 |
| [`docs/`](docs) | 超参数说明；物理图像的中文概述 |
| [`experiments/`](experiments) | 论文的实验，语言模型、势函数、分子动力学、Matbench Discovery、只读探针各一个文件夹：结果、数据、重算论文数字的脚本和复现命令 |
| [`tests/`](tests) | 单元测试 |

## 测试

```bash
python tests/run_tests.py    # 或 pytest tests
```

测试在 CPU 上运行，检查以下三点：与产出 GPT-2 结果的训练脚本中的优化器逐位一致；在应当退化为 Muon 的情形下确实等于 Muon；从 state dict 可以精确续跑。

## 引用

```bibtex
@misc{hu2026physmuon,
  title         = {A Physical Response-and-Memory Model for Muon Optimization},
  author        = {Hu, Yinze and Xiang, Hongjun and Gong, Xingao and Yu, Hongyu},
  year          = {2026},
  eprint        = {2608.22994},
  archivePrefix = {arXiv}
}
```

## 许可与致谢

MIT。`examples/nanogpt/train_gpt.py` 和 `tests/reference_nanogpt.py` 改自 [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)（MIT，许可文本见 [`examples/nanogpt/LICENSE`](examples/nanogpt/LICENSE)）。PhysMuon 建立在 Muon（K. Jordan 等，2024）之上，原子间势函数部分使用 [deepmd-kit](https://github.com/deepmodeling/deepmd-kit)。
