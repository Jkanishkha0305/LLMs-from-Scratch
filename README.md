# 🔨 Building LLMs from Scratch 🚀  
A curated collection of Large Language Models(LLMs), Small Language Models(SLM), Visiona Language Models(VLM) implemented from scratch for **Learning, experimentation, and innovation** across **Text, Vision, and Multimodal** domains.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/transformers/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org/)
![Lightning AI](https://img.shields.io/badge/Lightning%20AI-For%20GPU%20Resources-pink)
[![Weights & Biases](https://img.shields.io/badge/W%26B-Experiment%20Tracking-fcc42d?logo=wandb&logoColor=000)](https://wandb.ai/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo%20UI-FF4B4B?logo=gradio)](https://gradio.app/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-red) 

---

## 🧠 About This Repository

This repository documents the journey of building powerful Transformer based models from the ground up using **PyTorch** and other modern ML tools. It includes:

- Implementation of state-of-the-art LLMs, VLMs and diffusion models  
- End-to-end training pipelines including data loading, tokenization, optimization, and evaluation
- Hands-on experimentation with architectures like LLaMA, GPT, BERT, ViT, Diffusion, and multimodal models.
- Focus on core internals like Multi-Head & Grouped Query Attention, Rotary/ALiBi embeddings, BPE tokenization, KV-caching, weight tying, Top-k/p sampling, RMSNorm, SwiGLU, CLIP/SigLip, and MoE blocks
---

## 🧩 Models Implemented

| **Model Name** | **Type** | **Description** | **Code Repository** |
|--------|----------|------------------|----------------------|
| **PaliGemma**      | Vision-Language Model | Vision-language model enabling joint image and text reasoning.  | [View Code](https://github.com/Jkanishkha0305/LLMs-from-Scratch/tree/main/PaliGemma) |
| **LLaMA-1 (2.3M)** | Text Generation       | Tiny LLaMA-style decoder optimized for efficient text generation. | [View Code](https://github.com/Jkanishkha0305/LLMs-from-Scratch/tree/main/LLaMA-1-2.3M) |
| **GPT (29M)**      | Text Generation       | GPT-style transformer trained for next-token prediction.        | [View Code](https://github.com/Jkanishkha0305/LLMs-from-Scratch/tree/main/GPT-29M) |
| **LLaMA-3**        | Text Generation       | Enhanced LLaMA architecture with memory and performance tweaks. | [View Code](https://github.com/Jkanishkha0305/LLMs-from-Scratch/tree/main/LLaMA-3) |
| **Stable Diffusion** | Diffusion Model     | Latent diffusion model for high-quality image synthesis.        | 🚧 Coming Soon |
| **ViT**            | Vision Model          | Transformer architecture for image classification from patches. | 🚧 Coming Soon |

---

## 📚 Techniques & Concepts Covered

### 🧠 Transformer Architecture (LLaMA-Inspired)
<details>
<summary>Click to expand</summary>

- **Decoder-Only Transformers**: Implemented a causal Transformer architecture focusing on autoregressive text generation, inspired by LLaMA 1/2/3.
- **Multi-Head Attention Mechanism**: Built from scratch, including projection of queries, keys, and values, followed by scaled dot-product attention and concatenation.
- **Rotary Positional Embeddings (RoPE)**: Injected relative position information into attention using RoPE, enabling better generalization for long sequences.
- **RMSNorm Pre-Normalization**: Employed Root Mean Square Layer Normalization as a lightweight, scale-invariant alternative to LayerNorm.
- **SwiGLU Activation Function**: Integrated Switch Gated Linear Unit (SwiGLU) as an efficient and expressive feedforward activation function.
- **Weight Tying**: Shared input/output embeddings to reduce parameters.
- **KV-Cache**: Efficient decoding using cached keys and values for autoregressive generation.

---
</details>

### 🧩 Tokenization & Data Pipeline
<details>
<summary>Click to expand</summary>

- **Byte Pair Encoding (BPE)**: Custom BPE tokenizer implemented from scratch to compress vocabulary while maintaining text fidelity.
- **Text Preprocessing**: Cleaned, tokenized, and encoded large text corpora for training, preserving sequence context.
- **Embedding Layer**: Token embeddings initialized and trained with tied weights for input/output layers to reduce parameter count.
- **Dataset Loader**: Built custom iterable and batched dataset loaders for efficiency during model training and validation.

---
</details>

### 🧪 Training & Evaluation Pipeline
<details>
<summary>Click to expand</summary>

- **Loss Function**: Cross-Entropy Loss for autoregressive next-token prediction.
- **Evaluation Strategy**: Included perplexity, loss trend tracking, and sampled text generations for qualitative analysis.
- **Checkpointing**: Periodic model state saving and evaluation to prevent overfitting and enable model restoration.
- **Sampling Techniques**: Top-k / Top-p (nucleus) sampling implemented for creative yet controlled text generation.

---
</details>

### 🧬 Advanced Model Components (LLaMA 4 + DeepSeek Inspired)
<details>
<summary>Click to expand</summary>

- **Mixture of Experts (MoE)**: 
  - Introduced MoE layers within the feedforward block, enabling sparse activation of expert networks per token.
  - Used a learned router with Top-K gating and load balancing loss.
  - Shared expert implementation for baseline generalization across all inputs.

- **Gated MLPs with SiLU/Swish**: MLP experts activated with smooth nonlinearities for better gradient flow and expressiveness.
- **Grouped Query Attention (planned)**: Discussed GQA as an optimization in LLaMA3/4, with planned implementation for performance boost.

---
</details>

### 🧠 Reinforcement Learning from Human Feedback (RLHF) – R1 Training Concepts
<details>
<summary>Click to expand</summary>

- **Policy & Reward Models**: Simulated a reward model to score outputs on accuracy, structure, and reasoning.
- **GRPO Algorithm**: Applied a simplified version of Generalized Reinforcement Policy Optimization for model alignment.
- **Custom Rewards**:
  - Accuracy-based scoring
  - Format consistency
  - Multi-step reasoning chains
  - Repetition penalties
  - Cosine similarity rewards for embeddings
- **Few-shot Prompting with Long CoT**: Integrated multi-turn reasoning examples to guide response quality and completeness.

---
</details>

### 🖼️ Multimodal Capabilities (PaliGemma, SigLip)
<details>
<summary>Click to expand</summary>

- **PaliGemma Vision-Language Model**:
  - ViT Encoder + Gemma Decoder pipeline for image captioning.
  - Vision features projected with linear layer before token-level decoding.
  - Rotary Positional Embeddings across modalities.
  - RMSNorm in both encoder and decoder.
  - Top-P sampling for VQA generation.

- **SigLip Architecture**:
  - Contrastive Learning with Image-Text Pairs.
  - Vision Transformer backbone with Image Patch Tokenization.
  - Cosine similarity loss with learnable temperature.
  - Separate encoder + MLP for text and vision.

---
</details>



---

## ⚙️ Tech Stack

| Tool/Library | Purpose |
|--------------|---------|
| **Python** | Core programming language |
| **PyTorch** | Deep learning framework |
| **NumPy** | Numerical computing |
| **Hugging Face Datasets** | Tokenized datasets |
| **Matplotlib / Seaborn** | Visualization |
| **Weights & Biases** | Experiment tracking |
| **Jupyter Notebooks** | Prototyping and debugging |

---


## 🤝 Contributing

Have ideas? Fixes? New models?  
You're welcome to fork this repository and open a pull request.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

> Made with ❤️ by [@Jkanishkha0305](https://github.com/Jkanishkha0305)