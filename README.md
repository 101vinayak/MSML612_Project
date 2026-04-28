# MSML612 Project: Low-Rank MoE DistilBERT for SST-2

This project studies parameter-efficient compression of Transformer feed-forward network (FFN) layers for sequence classification. We modify DistilBERT on SST-2 using:

- Low-rank factorization of selected FFN expansion layers
- Mixture-of-Experts (MoE) routing over low-rank experts
- Knowledge distillation from the pretrained DistilBERT teacher
- Layer-wise sensitivity analysis
- Inference latency and throughput benchmarking

## Project Summary

The best low-rank model compresses selected FFN layers while improving efficiency:

| Model | Params (M) | Param Reduction (%) | Accuracy (%) | Latency (ms/sample) | Throughput (samples/sec) |
|---|---:|---:|---:|---:|---:|
| Baseline | 66.96 | 0.00 | 90.37 | 1.219 | 820.50 |
| Low-rank | 62.48 | 6.69 | 90.94 | 1.110 | 901.10 |
| MoE | 62.73 | 6.31 | 90.48 | 1.274 | 784.72 |
| MoE + KD | 62.73 | 6.31 | 91.06 | 1.275 | 784.31 |

Key finding:

- Low-rank compression gives the best accuracy-efficiency tradeoff.
- MoE + KD gives the highest accuracy but has higher inference latency due to routing overhead.
- Later Transformer layers are more compressible than earlier layers.

## Repo Layout

```text
.
├── src/
│   ├── data.py          # SST-2 loading and tokenization
│   ├── models.py        # Low-rank FFN, MoE FFN, DistilBERT layer replacement
│   └── utils.py         # Seeds, parameter counting, timing, JSON helpers
├── train.py             # Training baseline / low-rank / MoE / KD
├── eval.py              # Evaluate saved checkpoints
├── benchmark.py         # Latency, throughput, accuracy benchmarking
├── plot_results.py      # Final benchmark plots
├── plot_layer_sweep.py  # Layer sensitivity plots
├── requirements.txt
├── results/
└── figures/
````

## Setup

```bash
pip install -r requirements.txt
```

## Training Commands

Baseline:

```bash
python train.py --model_type baseline --epochs 3 --train_batch_size 16 --eval_batch_size 32 --max_length 128
```

Best low-rank model:

```bash
python train.py --model_type lowrank --epochs 3 --factorized_layers 4 5 --rank 32 --train_batch_size 16 --eval_batch_size 32 --max_length 128
```

MoE:

```bash
python train.py --model_type moe --epochs 3 --factorized_layers 4 5 --num_experts 4 --rank 16 --top_k 1 --train_batch_size 8 --eval_batch_size 16 --max_length 128
```

MoE + KD:

```bash
python train.py --model_type moe --epochs 3 --factorized_layers 4 5 --num_experts 4 --rank 16 --top_k 1 --use_distillation --alpha 0.8 --beta 0.2 --temperature 2.0 --train_batch_size 8 --eval_batch_size 16 --max_length 128
```

## Benchmarking

```bash
python benchmark.py --batch_size 32 --max_length 128
```

This generates:

```text
results/benchmark_results.json
```

## Plot Generation

```bash
python plot_results.py
python plot_layer_sweep.py
```

Generated figures include:

* `figures/accuracy_vs_params.png`
* `figures/accuracy_vs_latency.png`
* `figures/throughput_comparison.png`
* `figures/latency_comparison.png`
* `figures/params_comparison.png`
* `figures/layer_sensitivity_accuracy.png`
* `figures/layer_sensitivity_params.png`
* `figures/layer_sweep_accuracy_vs_params.png`

## Layer-wise Compression Analysis

| Layers Factorized  | # Layers | Params (M) | Accuracy (%) |
| ------------------ | -------: | ---------: | -----------: |
| [5]                |        1 |      64.72 |        90.83 |
| [4, 5]             |        2 |      62.48 |        90.94 |
| [3, 4, 5]          |        3 |      60.24 |        90.60 |
| [2, 3, 4, 5]       |        4 |      58.00 |        89.91 |
| [1, 2, 3, 4, 5]    |        5 |      55.76 |        87.96 |
| [0, 1, 2, 3, 4, 5] |        6 |      53.52 |        86.70 |

## Notes

* All reported accuracy values are on the SST-2 validation set.
* GLUE/SST-2 test labels are not publicly available, so validation is used for evaluation.
* Checkpoints are saved locally under `checkpoints/` but should not be pushed to GitHub.
