# Fine-Tuned Mask R-CNN

This folder contains the code and artifacts for fine-tuning a **Mask R-CNN (ResNet-50 + FPN)** model on a 16K-image subset of the LVIS dataset for instance segmentation.

---

## Files

| File | Description |
| :--- | :--- |
| `RCNN Model Fine Tuning.ipynb` | Full training, inference, and evaluation notebook |
| `mask_rcnn_final.pth` | Saved model weights after training |
| `fine_tune_evaluation_results.csv` | Final evaluation metrics |

---

## Notebook: RCNN Model Fine Tuning.ipynb

The notebook is organized into three parts:

### Part 1: Training

1. **Dataset Extraction & Exploration** — Extracts `lvis_16k_dataset.zip`, verifies images and annotation files.
2. **Load & Inspect Annotations** — Uses `pycocotools` to load COCO-format annotations; visualizes random samples with bounding boxes, labels, and segmentation masks.
3. **Custom PyTorch Dataset** — Implements `CustomCocoDataset` that loads images with bounding boxes, category labels, and instance masks. Wraps in a `DataLoader` with batch size 4.
4. **Model Architecture** — Loads a pretrained Mask R-CNN backbone and replaces the box predictor and mask predictor heads to match the number of LVIS classes.
5. **Training Loop** — Trains for 5 epochs using the Adam optimizer (lr=1e-4) with per-batch loss tracking via `tqdm`.
6. **Save Model** — Exports trained weights to `mask_rcnn_final.pth`.

### Part 2: Inference

1. **Load Trained Model** — Rebuilds the model architecture and loads saved weights.
2. **Preprocess Test Image** — Reads a test image, normalizes to `[0, 1]`, converts to `(C, H, W)` tensor.
3. **Run Inference** — Generates predictions and filters by a 0.5 confidence threshold.
4. **Visualize Predictions** — Renders bounding boxes, category names, confidence scores, and instance masks.

### Part 3: Evaluation

1. **Mean IoU (mIoU)** — Compares predicted instance masks against ground-truth masks over 50 images.
2. **Rare Class Performance** — Evaluates detection accuracy on rare categories (< 100 annotations).
3. **Full Metrics Computation** — Computes mAP by frequency group (rare/common/frequent), mIoU, and inference latency. Saves results to CSV.

---

## Evaluation Results

| Metric | Value | Context |
| :--- | :--- | :--- |
| Overall mAP | 0.301 | IoU=0.5:0.95 |
| mAP (Rare) | 0.0 | Zero-Shot Success |
| mAP (Common) | 0.388 | Common Categories |
| mAP (Frequent) | 0.514 | Frequent Categories |
| mIoU Proxy (AP75) | 0.283 | Geometric Accuracy |
| Avg Latency | 39.7 ms | RTX 4090 |
| Baseline Type | Mask R-CNN | Closed-Set |

> **Key takeaway:** The fine-tuned Mask R-CNN performs well on frequent/common categories but scores **0.0 mAP on rare classes**, highlighting the limitation of closed-set models on long-tail distributions.

---

## Key Libraries

`torch`, `torchvision`, `pycocotools`, `cv2`, `matplotlib`, `numpy`, `pandas`
