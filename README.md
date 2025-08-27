[![Huggingface Space](https://img.shields.io/badge/Demo%20-yellow.svg)](https:///)

<div align="center">
    <img src="/images/Banner.png" alt="Banner" />

</div>



## Setup

### 1) Clone

```bash
git clone git@github.com:ManarEyad7/kaust-surv-analysis.git
cd kaust-surv-analysis
```

### 2) Conda env

```bash
conda create --name kaust-surv python=3.10 -y
conda activate kaust-surv
```

### 3) Python deps

```bash
pip install -r requirements.txt
```

### 4) ffmpeg

```bash
conda install -c conda-forge ffmpeg -y
```

### 5) RAFT Weights

Download the pretrained RAFT models and place them under RAFT/models:

```bash

cd RAFT
unzip models.zip -d models
```

## Running the Pipeline

**Run in order:** 1) Segmentation → 2) Captioning → 3) Merge

### 1) Video Segmentation (`motion_based_clip_segmentation_adaptive.py`)

Cuts a long video into motion-dense clips (RAFT).

**Example**

```bash
python LoVR/motion_based_clip_segmentation_adaptive.py \
  --model RAFT/models/raft-things.pth \
  --video /path/to/your/video.mp4 \
  --out_dir /path/to/output/dir \
  --iters 12 \
  --max_side 640 \
  --savgol_window 31 --savgol_poly 3 \
  --thr z --k_hi 2.0 --k_lo 1.0 \
  --min_on 24 --min_off 24 --max_hole 6 \
  --seg_k 1.0 --min_active_ratio 0.20 \
  --merge_gap_sec 4.0 \
  --export_clips
# (Optional if FPS metadata is wrong) add: --force_fps --fps 24
```

---

### 2) Caption Generation (`caption_generator.py`)

Generates captions for clips from step 1.

**Example (chunked)**

```bash
# Replace <> with your actual paths
python LoVR/caption_generator.py \
  --model-path Qwen/Qwen2-VL-7B-Instruct \
  --video-folder </path/to/clips> \
  --jsonl-file </path/to/index.jsonl> \
  --result-file </path/to/output/captions.jsonl> \
  --num-chunks 1 \
  --chunk-idx 0 \
  --batch-size 4 \
  --debug
```

---

### 3) Merge Caption Results (`caption_merger.py`)

Merges per-chunk caption JSONLs into one final file.

**Key flags:**
`--cap-file` final merged JSONL • `--result-file` input chunks (repeat or glob) • `--num-workers`

**Example**

```bash
python LoVR/caption_merger.py \
  --cap-file ${BASE}/results/captions_merged.jsonl \
  --result-file "${BASE}/results/captions_chunk_*.jsonl" \
  --num-workers 8
```

---
## References

[1] Q. Cai et al., “LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts,” *arXiv:2505.13928*, 2025.  
[2] Y. Wu et al., “Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges,” *arXiv:2309.13925*, 2023.  
[3] Z. Teed and J. Deng, “RAFT: Recurrent All-Pairs Field Transforms for Optical Flow,” *arXiv:2003.12039*, 2020.

### Notes

* Replace sample paths with your actual data locations.
* Outputs from step 1 include `motion.csv`, `segments.json`, `segments_sec.json`, `plot.png`, and optional `clips/*.mp4`.
