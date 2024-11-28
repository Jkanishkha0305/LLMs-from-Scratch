# PaliGemma Vision Language Model

A fully coded implementation of a **Multimodal (Vision) Language Model** from scratch using **Python** and **PyTorch**. This project follows a detailed tutorial and explains all underlying concepts while building the **PaliGemma Vision Language Model** step-by-step.

---

## Overview

The **PaliGemma Vision Language Model** is a multimodal system capable of generating textual descriptions or understanding inputs by processing both visual (images) and textual (prompts) data. This project focuses on explaining and implementing foundational concepts like Transformer architecture, Vision Transformer, and advanced techniques such as contrastive learning, attention mechanisms, and efficient inference strategies.

---

## Key Features and Concepts

This project covers the following topics in-depth:
- **Transformer Model Architecture**:
  - Embeddings, Positional Encoding
  - Multi-Head Attention (MHA) with Grouped Query Attention
  - Feed Forward Layer, Logits, and Softmax
- **Vision Transformer**:
  - Coding a visual encoder using SigLip and contrastive learning
- **Contrastive Learning**:
  - Concepts of CLIP and SigLip
  - Numerical stability of Softmax and Cross-Entropy Loss
- **Normalization Layers**:
  - Batch Normalization, Layer Normalization, RMS Normalization
- **KV-Cache**:
  - Prefilling and token generation for efficient decoding
- **Attention Mechanisms**:
  - Causal and non-causal masks
  - Rotary Positional Embedding
- **Sampling Strategies**:
  - Top-P (nucleus) Sampling
  - Temperature-based generation
- **Weight Tying** for parameter sharing across layers
## Key Coding Concepts

Below are the key coding concepts and milestones implemented in this project:

| Concept                                     |
|---------------------------------------------|
| Contrastive Learning and CLIP               |
| Numerical stability of the Softmax          |
| SigLip                                       |
| Why a Contrastive Vision Encoder?           |
| Vision Transformer                          |
| Coding SigLip                               |
| Batch Normalization, Layer Normalization    |
| Coding SigLip (Encoder)                     |
| Coding SigLip (FFN)                         |
| Multi-Head Attention (Coding + Explanation) |
| Coding SigLip                               |
| PaliGemma Architecture review               |
| PaliGemma input processor                   |
| Coding Gemma                                |
| Weight tying                                |
| Coding Gemma                                |
| KV-Cache (Explanation)                      |
| Coding Gemma                                |
| Image features projection                   |
| Coding Gemma                                |
| RMS Normalization                           |
| Gemma Decoder Layer                         |
| Gemma FFN (MLP)                             |
| Multi-Head Attention (Coding)               |
| Grouped Query Attention                     |
| Multi-Head Attention (Coding)               |
| KV-Cache (Coding)                           |
| Multi-Head Attention (Coding)               |
| Rotary Positional Embedding                 |
| Inference code                              |
| Top-P Sampling                              |
| Inference code                              |

---

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.x or higher
- CUDA (optional, for GPU acceleration)
- Additional Python libraries: 
  - `torch`, `fire`, `Pillow`

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/paligemma-vlm.git
   cd paligemma-vlm
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained weights:

	•	From Hugging Face
	•	Save them to a folder (e.g., ~/projects/paligemma-weights/).

### Usage

#### Running Inference

Use the provided script to run inference on the model:
	1.	Update the inference.sh file with your paths for the model, image, and parameters.
	2.	Run the script:

```bash
  ./inference.sh
```

Example Output

Input:
	•	Prompt: “this building is”
	•	Image: A picture of the Eiffel Tower.

Output:

this building is an iconic French landmark known as the Eiffel Tower.

### Folder Structure
```
paligemma-vlm/
├── inference.py         # Main inference script
├── processing_paligemma.py # Input processing pipeline
├── modeling_gemma.py    # Model architecture implementation
├── utils.py             # Utility functions
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
└── inference.sh         # Shell script for inference
```

## Literature for Better Understanding

Below are some key resources and papers to deepen your understanding of the concepts and technologies used in this project:

### Vision Transformers (ViT)
- **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**  
  _A foundational paper introducing Vision Transformers (ViT) for computer vision tasks._
  [Link to Paper](https://arxiv.org/abs/2010.11929)

### SigLIP
- **"Learning Transferable Visual Models From Natural Language Supervision" (CLIP)**  
  _This paper by OpenAI introduces CLIP, which serves as the foundation for SigLIP's contrastive learning approach._
  [Link to Paper](https://arxiv.org/abs/2303.15343)

### Gemma
- **"Gemma: A Modular and Scalable Approach to Multimodal Learning"**  
  _Gemma is a modular approach enabling efficient multimodal learning._  
  [Link to Paper](https://arxiv.org/abs/2403.08295)

### Pali-3 and PaliGemma
- **"Pali-3: Scaling Multimodal Models for Enhanced Vision-Language Understanding"**  
  _Pali-3 provides the foundational architecture expanded upon in PaliGemma._  
  [Link to Paper](https://arxiv.org/abs/2310.09199)  
  [PaliGemma Github](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md)

### Related Foundational Papers
- **"Attention Is All You Need"**  
  _This paper introduces the Transformer architecture, the backbone of modern deep learning models like ViT._  
  [Link to Paper](ttps://arxiv.org/abs/1706.03762)

- **"Rotary Position Embedding"**  
  _Introduces rotary positional embeddings (RoPE), an efficient method to encode sequence information in Transformers._  
  [Link to Paper](https://arxiv.org/abs/2104.09864v5)

- **"Root Mean Square Layer Normalization"**  
  _A variant of normalization techniques providing stable training and improved performance._  
  [Link to Paper](https://arxiv.org/abs/1910.07467)

- **"Gaussian Error Linear Units" (GELU)**  
  _A popular activation function commonly used in modern deep learning architectures._  
  [Link to Paper](https://arxiv.org/abs/1606.08415v5)

### Contributions

Contributions are welcome! Feel free to fork this repository, create issues, or submit pull requests to enhance this project.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgements

This project was built using the Video tutorial [[tutorial](https://youtu.be/vAmKB7iPkWw?si=SakH0BmprOGUhxDj)] by Umar Jamil and concepts from recent advances in multimodal learning, including CLIP and Vision Transformers.

