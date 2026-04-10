# Grounded-SAM 2 — Inference & Evaluation Pipeline

This folder contains the core zero-shot segmentation pipeline that combines **Grounding DINO** (open-vocabulary detection) with **SAM 2** (high-fidelity mask generation) and evaluates results against the **LVIS 16k** ground truth.

---

## Pipeline Overview

```
Text Prompt ──▶ Grounding DINO ──▶ Bounding Boxes ──▶ SAM 2 ──▶ Segmentation Masks
                                                                        │
                                                              RLE Encoding
                                                                        │
                                                              LVIS Evaluation
```

1. **Dynamic Prompt Engineering** — Per-image category names are pulled from the LVIS annotations and concatenated into a text prompt (capped at 15 categories / 512 BERT tokens).
2. **Grounding DINO Detection** — The text prompt is matched against the image to produce bounding boxes with confidence scores.
3. **SAM 2 Segmentation** — Detected boxes are fed as spatial prompts to SAM 2, which generates pixel-level masks.
4. **RLE Encoding & LVIS Evaluation** — Masks are compressed to Run-Length Encoding and scored with the official LVIS API (mAP, mAP_rare, etc.).

---

## File Descriptions

| File | Purpose |
| :--- | :--- |
| `hybrid_model.py` | `GroundedSAM2` class — loads Grounding DINO + SAM 2 checkpoints and exposes `run_inference()` |
| `main.py` | Runs the full benchmark over all LVIS 16k images, saves predictions to `lvis_results.json` |
| `evaluation_metrics.py` | `mask_to_rle()` and `evaluate_lvis()` — RLE conversion and official LVIS metric computation |
| `run_eval.py` | Post-inference report generator — produces `final_project_metrics.csv` and `full_benchmark_results.csv` |
| `download_ckpts.sh` | Downloads SAM 2 Hiera-Large and Grounding DINO SwinT-OGC checkpoints |
| `requirements.txt` | Python dependencies (PyTorch, pycocotools, lvis, transformers, etc.) |
| `lvis_results.json` | Raw LVIS-formatted predictions (output of `main.py`) |
| `final_project_metrics.csv` | Executive summary — mAP, mAP_rare, latency |
| `full_benchmark_results.csv` | Instance-level detection log (~140k rows) |

---

## Results

| Metric | Value | Context |
| :--- | :--- | :--- |
| **Overall mAP** | 18.3% | IoU = 0.50:0.95 |
| **mAP (Rare)** | 18.6% | Zero-shot success on unseen categories |
| **mAP (Common)** | 16.9% | Common categories |
| **mAP (Frequent)** | 19.7% | Frequent categories |
| **AP75** | 19.4% | Geometric / mask accuracy |
| **Avg Latency** | 173.7 ms | Per-image on RTX 4090 |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model checkpoints

```bash
bash download_ckpts.sh
```

### 3. Run the benchmark

```bash
python main.py
```

### 4. Generate evaluation reports

```bash
python run_eval.py
```

---

## Requirements

- Python 3.10+
- CUDA-enabled GPU (benchmarked on RTX 4090)
- LVIS 16k dataset placed at `/workspace/lvis_16k/lvis_16k_dataset/`
- Grounded-SAM-2 repository cloned to `/workspace/Grounded-SAM-2/`

---

## Author

**Mahfuzzur Rahman**
COMP 6341 — Computer Vision, Concordia University (April 2026)
