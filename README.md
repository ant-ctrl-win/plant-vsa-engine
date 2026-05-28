# 🌐 General-Purpose Neuro-Symbolic Inference Engine using VSA

![Java](https://img.shields.io/badge/Java-11%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-0.5.0-orange)
![Edge Ready](https://img.shields.io/badge/Edge%20Ready-Yes-brightgreen)

An open-source implementation of a **domain-agnostic Neuro-Symbolic Inference Engine**. This architecture bridges the gap between deep learning perception and hyperdimensional logic, enabling **ultra-fast, few-shot continual learning on resource-constrained edge devices** without backpropagation.

---

## 🎯 Core Philosophy & Paradigm Shift
Traditional Edge AI relies on heavy, fully connected neural network heads followed by computationally expensive Softmax layers. This architecture replaces the entire classification head with a **Vector Symbolic Architecture (VSA / Hyperdimensional Computing)** framework.

### Key Innovations:
* **Zero-Cloud Continual Learning:** New classes can be learned directly on the field (e.g., on a smartphone or microcontroller) by performing simple vector additions, skipping the need for backpropagation or GPU retraining.
* **Explainable Confidence via Z-Scores:** Instead of "hallucinating" predictions on unseen or out-of-distribution data, the engine measures actual geometric distances, safely filtering anomalies that fall below a 3-Sigma confidence threshold.

---

## 🧠 Part 1: The Vector Processing Pipeline
The core engine is entirely domain-agnostic. It processes raw embeddings from any upstream encoder through a strict mathematical pipeline:

1. **Continuous Random Projection:** A fixed, deterministic pseudo-random matrix maps the initial embedding (e.g., `float[1280]`) into a holographic `10,000`-dimensional space. L2 Normalization is applied immediately to unify vector magnitudes onto a unit hypersphere.
2. **Dimension-Wise Z-Score Standardization:** The projected global distribution is centered around the origin. This mathematical step is crucial: it translates the dense cloud of vectors into a sferical arrangement, forcing unrelated vectors to become **quasi-orthogonal** (an expected angular distance of 90°).
3. **Continuous Bundling (Late Binarization):** Instead of binarizing vectors individually (which acts as a destructive low-pass filter on correlated features), the system aggregates training samples via element-wise addition in the continuous domain.
4. **Sign Crystallization:** The accumulated continuous bundle is finally passed through a sign threshold function (`sum > 0 -> 1, else 0`) and bit-packed into an array of `long` primitive types, generating a pristine **Binary Spatter Code (BSC)** archetype.

---

## 📊 Part 2: Evaluation & The Metrics Journey
To evaluate the mathematical validity of this framework, we documented a two-step ablation study tracking how the symbolic layer adapts to different encoder qualities:

### Test A: Untrained/Generic CNN Encoder
* **Setup:** A standard MobileNetV2 pre-trained exclusively on generic ImageNet objects (cars, cats, chairs), possessing zero intrinsic knowledge of agricultural features.
* **VSA Accuracy:** **~91.25%**
* **Insight:** Even with a sub-optimal, noisy embedding space, the combination of Z-Score centering and high-dimensional bundling successfully extracts structural patterns, achieving high classification accuracy with zero domain training on the CNN.

### Test B: Domain Fine-Tuned CNN Encoder
* **Setup:** The exact same MobileNetV2 encoder, but fine-tuned on the domain specific textures.
* **VSA Accuracy:** **98.77%**
* **Inference Time (VSA Logic):** **< 10 ms**
* **Memory Footprint:** **~165 KB**
* **Insight:** When paired with high-quality latent spaces, the VSA engine crystallizes perfect boundaries. The target class emerges with massive confidence (**> 25-Sigma**), while alternative classes remain strictly orthogonal (near 0-Sigma).

---

## 🌿 Part 3: Application Example (PlantVillage Use-Case)
As a concrete evaluation environment, the engine was deployed to diagnose plant diseases using the comprehensive PlantVillage dataset.
* 🔗 **Dataset Source:** [PlantVillage on Kaggle (by Mohit Singh)](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)

In this specific use-case, the system processed thousands of grape leaf images across 4 distinct classes (*Healthy*, *Black Rot*, *Esca*, *Leaf Blight*), distilling the visual experience of 3,251 images into a static 5 KB binary file (`vsa_archetypes.bin`) and a 160 KB statistical anchor (`vsa_stats.bin`).

---

## 💻 How To Run

### Prerequisites
* Java 11 or higher
* Maven

### 1. Build the Project
```bash
mvn clean compile