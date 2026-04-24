# Devanagari-OCR-using-Sequence-Modele
A deep learning-based Optical Character Recognition (OCR) system for handwritten Devanagari script, built using a CNN + Bidirectional LSTM + CTC Loss pipeline.

🚀 Overview

This project tackles the problem of recognizing handwritten Devanagari words from images without explicit character segmentation.
Unlike traditional OCR systems, this model:
- Works end-to-end
- Requires no character-level annotation
- Handles variable-length words
- Uses sequence modeling + CTC alignment

🎯 Motivation

Over 600M+ people use Devanagari script
Large amount of undigitized handwritten data
Existing OCR systems (e.g., Tesseract, Google Vision):
Not script-specific
Perform poorly on handwriting
Cloud OCR APIs:
Expensive
Not accessible for research

👉 This project builds a free, accurate, script-specific OCR model

📊 Dataset

IIIT-HW-Dev Dataset

🧠 95,000+ word images
✍️ Handwritten Devanagari
📂 Word-level annotations

| Split      | Samples |
| ---------- | ------- |
| Train      | 69,853  |
| Validation | 12,708  |
| Test       | 12,869  |

Format:
HindiSeg/train/1/234/7.jpg   दवाओं

⚙️ Pipeline
```
Dataset → Preprocessing → Vocabulary → Model → Training → Evaluation
```

Steps:
- Preprocessing
- Grayscale conversion
- Resize to (32 × 128)
- Normalize
- Augmentation (Train Only)
- Gaussian Blur
- Color Jitter
- Rotation
- Perspective Warp
- Sharpness adjustment
- Vocabulary
- 94 characters,
  Includes matras, digits, punctuation
- CTC blank token (index 0)

🧠 Model Architecture (CRNN)
```
Input Image (B,1,32,128)
        ↓
Input Image (B,1,32,128)
        ↓
CNN Backbone
        ↓
Feature Map (B,512,1,32)
        ↓
Sequence Conversion → (32,B,512)
        ↓
BiLSTM (2 layers, 256 hidden)
        ↓
Linear Layer → (32,B,94)
        ↓
     CTC Loss
```
🔍 CNN Details

| Block | Channels | Pooling | Output Size |
| ----- | -------- | ------- | ----------- |
| 1     | 64       | 2×2     | 16×64       |
| 2     | 128      | 2×2     | 8×32        |
| 3     | 256      | (2×1)   | 4×32        |
| 4     | 512      | (2×1)   | 2×32        |
| 5     | 512      | (2×1)   | 1×32        |

👉 Height collapses → converts image into sequence

🔁 BiLSTM
- Input: 512-dim features
- Hidden: 256 per direction
- Output: 512 per timestep
- Layers: 2
- Dropout: 0.3

🔤 Output Layer
- Linear: 512 → 94 (vocab size)
- Produces character probabilities per timestep

🔗 CTC Loss
CTC enables training without alignment.

Example:
```
Raw:   र र - - ा ा म म
Decode: राम
```
Handles:
- Repeated characters
- Blank tokens
- Eliminates need for segmentation

📈 Results

🔥 Final Performance
| Metric            | Value                    |
| ----------------- | ------------------------ |
| CER (Greedy)      | **11.1%**                |
| CER (Beam Search) | **~9%**                  |
| WER               | **38.0%**                |
| Exact Match       | **62.0%** (7980 / 12869) |

📉 Validation CER Curve
```
Epoch:  1   → 30  
CER:   0.85 → 0.11
```
Trend:
- Rapid early learning
- Stabilization after ~20 epochs
- Best model at epoch 23

📊 Prediction Quality Breakdown

| Category         | Percentage |
| ---------------- | ---------- |
| Perfect (0% CER) | 38.2%      |
| Good (<10%)      | 28.5%      |
| Fair (<30%)      | 16.3%      |
| Poor (<60%)      | 9.4%       |
| Wrong (>60%)     | 7.6%       |

⚡ Training Details

- Epochs: 30–50
- Optimizer: Adam
- LR: 1e-3
- Loss: CTC
- Hardware: RTX 3050 Ti
- Training Time: ~3 hours

🔍 Inference

- Greedy Decoding
- Fast
- Lower accuracy
- Beam Search (width=10)
- Improves CER (~9%)
- Better word-level predictions
```
GT:   अनाथों
PRED: अनाथों

GT:   देखना
PRED: देरनना

GT:   ऊर्ध्वगामी
PRED: उर्वगामी
```
📁 Project Structure
- ├── dataset.py
- ├── model.py
- ├── train.py
- ├── test.py
- ├── vocab.py
- ├── checkpoints/
- ├── HindiSeg/
- ├── train.txt / val.txt / test.txt

▶️ How to Run
1. Install dependencies
   ```
   pip install torch torchvision pillow
   ```
2. Train
   ```
   python train.py
   ```
3. Test
   ```
   python test.py
   ```
🧠 Key Learnings
- Sequence modeling is essential for OCR
- CNN extracts spatial features
- BiLSTM captures context
- CTC removes need for alignment
- Beam search improves decoding

🚀 Future Work
- Transformer-based OCR (TrOCR-style)
- Lexicon-guided decoding
- Line-level OCR
- Synthetic data generation
- Mobile deployment (ONNX / TFLite)

🏁 Conclusion

A lightweight (~4.75M params), efficient OCR model trained from scratch that:

- Achieves strong performance on handwritten Devanagari
- Outperforms generic OCR systems in this domain
- Is fully open and extensible

📜 Quote

"A small but dedicated model, trained right, outperforms a generic one trained on everything."
