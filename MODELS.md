# Model Index — LLMs-from-Scratch

Complete reference of all language model implementations in this repository.

## Large Language Models (LLMs)

| Model | Architecture | Params | Notebook | Status |
|-------|-------------|--------|----------|--------|
| GPT-2 | Transformer decoder | 117M | `notebooks/gpt2/` | ✅ |
| LLaMA-2 | Grouped-query attention | 7B | `notebooks/llama2/` | ✅ |
| LLaMA-3 | GQA + RoPE improvements | 8B | `notebooks/llama3/` | ✅ |
| DeepSeek-R1 | Mixture of Experts | — | `notebooks/deepseek/` | ✅ |

## Small Language Models (SLMs)

| Model | Params | Technique | Notebook | Status |
|-------|--------|-----------|----------|--------|
| Phi-inspired | 1.3B | Textbooks-are-all-you-need | `notebooks/phi/` | ✅ |
| TinyLLaMA | 1.1B | Continuous pre-training | `notebooks/tinyllama/` | ✅ |

## Vision Language Models (VLMs)

| Model | Architecture | Notebook | Status |
|-------|-------------|----------|--------|
| PaliGemma | SigLIP + Gemma | `notebooks/paligemma/` | ✅ |
| Vision Transformer (ViT) | Patch embedding + Transformer | `notebooks/vit/` | ✅ |

## Techniques Covered

- ✅ Transformer architecture (attention, MLP, LayerNorm)
- ✅ Rotary Position Embeddings (RoPE)
- ✅ Grouped-Query Attention (GQA)
- ✅ LoRA / QLoRA fine-tuning
- ✅ RLHF / DPO alignment
- ✅ Speculative decoding
- ✅ Flash Attention
- ✅ Mixture of Experts (MoE)
