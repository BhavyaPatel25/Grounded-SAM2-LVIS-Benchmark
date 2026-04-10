# Grounded-SAM 2: Zero-Shot Open-Vocabulary Segmentation on LVIS

This repository benchmarks **open-vocabulary zero-shot segmentation** against traditional closed-set approaches on the **LVIS dataset** (1,200+ categories). The core contribution is a hybrid pipeline that pairs **Grounding DINO** (text-guided detection) with **Segment Anything Model 2** (SAM 2) for mask generation — and demonstrates that it **outperforms fine-tuned models on rare categories without any training on LVIS**.

---

## 🧠 Motivation

Standard object detectors like Mask R-CNN are limited to a fixed set of training classes. When deployed on datasets with a long-tail distribution (many rare categories), they fail on any class not seen during training. Open-vocabulary models eliminate this constraint by using natural language to describe what to detect, enabling zero-shot generalisation to arbitrary categories.

---

## 🏗️ Pipeline Architecture

```
Category Names (from LVIS annotations)
         │
         ▼
   Grounding DINO  ──▶  Bounding Boxes + Confidence Scores
                                │
                                ▼
                           SAM 2  ──▶  Instance Segmentation Masks
                                              │
                                              ▼
                                     RLE Encoding  ──▶  LVIS Evaluation (mAP)
```

---

## 📂 Repository Structure

```
Grounded-SAM2-LVIS-Benchmark/
│
├── Dataset Exploration/        # LVIS analysis & 16k subset creation
│   ├── LVIS_Dataset_Exploration.ipynb   # Full LVIS v1 statistics & visualisation
│   ├── Making Dataset Smaller.ipynb     # Balanced 16k subset sampling pipeline
│   └── Exploring Smaller Dataset.ipynb  # QA validation of the 16k subset
│
├── Inferred_RCNN/              # Baseline 1 — COCO-pretrained Mask R-CNN (no fine-tuning)
│   ├── RCNN_Evaluation.ipynb   # Inference + LVIS evaluation
│   ├── rcnn_results.json       # Raw predictions
│   └── final_rcnn_results.csv  # Summary metrics
│
├── Fine_Tuned_RCNN/            # Baseline 2 — Mask R-CNN fine-tuned on LVIS 16k
│   ├── RCNN Model Fine Tuning.ipynb  # Training (5 epochs) + evaluation
│   ├── mask_rcnn_final.pth     # Saved model weights
│   └── fine_tune_evaluation_results.csv  # Summary metrics
│
└── Grounded_SAM2/              # Main contribution — Zero-shot pipeline
    ├── hybrid_model.py         # GroundedSAM2 class (Grounding DINO + SAM 2)
    ├── main.py                 # Full 16k benchmark driver
    ├── evaluation_metrics.py   # RLE encoding & official LVIS evaluation
    ├── run_eval.py             # Post-inference report generation
    ├── download_ckpts.sh       # Checkpoint download script
    ├── requirements.txt        # Python dependencies
    ├── lvis_results.json       # Raw LVIS-format predictions (~140k detections)
    ├── final_project_metrics.csv       # Executive summary metrics
    └── full_benchmark_results.csv      # Instance-level detection log
```

---

## 📊 Results Comparison

| Model | Type | mAP (Overall) | mAP (Rare) | mAP (Common) | mAP (Frequent) | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| COCO-Pretrained Mask R-CNN | Closed-Set | ~0.0% | 0.0% | 0.0% | 0.0% | 57.1 ms |
| Fine-Tuned Mask R-CNN | Closed-Set | 30.1% | 0.0% | 38.8% | 51.4% | 39.7 ms |
| **Grounded-SAM 2** | **Open-Vocabulary** | **18.3%** | **18.6%** | **16.9%** | **19.7%** | **173.7 ms** |

### Key Takeaways

- **COCO-pretrained Mask R-CNN** achieves ~0% mAP — its 80 COCO classes do not cover LVIS's 1,200+ categories.
- **Fine-tuned Mask R-CNN** reaches 30.1% overall but **completely fails on rare categories (0.0%)** because they lack training examples.
- **Grounded-SAM 2** scores 18.3% overall with **18.6% on rare categories** — demonstrating strong zero-shot transfer across the entire long-tail distribution.

The fine-tuned baseline wins on frequent classes (51.4% vs 19.7%), but its inability to handle rare categories is a fundamental limitation of the closed-set paradigm.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r Grounded_SAM2/requirements.txt
```

### 2. Download model checkpoints

```bash
cd Grounded_SAM2
bash download_ckpts.sh
```

### 3. Run the benchmark

```bash
python Grounded_SAM2/main.py
```

### 4. Generate evaluation reports

```bash
python Grounded_SAM2/run_eval.py
```

### Requirements

- Python 3.10+
- CUDA-enabled GPU (benchmarked on RTX 4090)
- LVIS 16k dataset at `/workspace/lvis_16k/lvis_16k_dataset/`
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) cloned to `/workspace/Grounded-SAM-2/`

---

## 🎓 Academic Context

* **Institution:** Concordia University
* **Course:** COMP 6341 — Computer Vision
* **Date:** April 2026

---
*Developed as part of the Master of Applied Computer Science (MACS) program at Concordia University.*
