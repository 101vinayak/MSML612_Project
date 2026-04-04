# Low-Rank MoE DistilBERT for SST-2

Minimal, reproducible course-project repo for compressing DistilBERT feed-forward layers using low-rank factorization and mixture-of-experts routing on SST-2.

## Repo layout

- `src/data.py` - dataset loading and tokenization
- `src/models.py` - low-rank FFN, MoE FFN, and DistilBERT layer swapping
- `src/utils.py` - seeds, parameter counting, timing, save helpers
- `train.py` - train baseline / low-rank / MoE student, with optional distillation
- `eval.py` - evaluate checkpoints and print metrics

## Quick start (Colab)

```bash
pip install -r requirements.txt
python train.py --model_type baseline --epochs 1
python train.py --model_type lowrank --epochs 2 --factorized_layers 4 5 --rank 64
python train.py --model_type moe --epochs 2 --factorized_layers 4 5 --num_experts 4 --rank 32 --top_k 1 --use_distillation
python eval.py --checkpoint checkpoints/moe_last.pt --model_type moe --factorized_layers 4 5 --num_experts 4 --rank 32 --top_k 1
```

## Suggested interim-report experiments

1. Baseline DistilBERT on SST-2
2. Low-rank FFN replacement on final 2 layers
3. Low-rank MoE FFN replacement on final 2 layers
4. Optional distillation from the pretrained SST-2 teacher

## Notes

- Keep scope encoder-only for the report.
- Use SST-2 consistently.
- Save all outputs under `results/` and `checkpoints/`.
