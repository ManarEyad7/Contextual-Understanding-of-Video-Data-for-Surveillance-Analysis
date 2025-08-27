

# KAUST Surveillance Analysis Project

Tools for motion-based segmentation and captioning of surveillance videos (RAFT-based flow + action-focused captions).

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

### 1) Video Segmentation (`motion_based_clip_segmentation.py`)

Cuts a long video into motion-dense clips (RAFT).

**Key flags:**
`--model` RAFT weights • `--video` input video • `--out_dir` outputs •
`--iters` flow iters • `--max_side` resize • `--savgol_window/savgol_poly` smoothing •
`--thr z|quantile` with `--k_hi/--k_lo` or `--q_hi/--q_lo` •
`--min_on/--min_off/--max_hole` durations (frames) •
`--seg_k/--seg_q --min_active_ratio` quality filter •
`--merge_gap_sec` merge nearby segments (sec) • `--export_clips` save MP4s

**Example**

```bash
python LoVR/motion_based_clip_segmentation.py \
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

**Key flags:**
`--model-path` model weights • `--video-folder` clips dir • `--jsonl-file` clip index •
`--result-file` output JSONL • `--batch-size` • `--num-chunks` • `--chunk-idx` (0-based)

**Example (chunked)**

```bash
export CKPT=/path/to/model_weights
export BASE=/path/to/workdir
CHUNKS=8
IDX=0
LOG_FILE=${BASE}/logs/output_${IDX}.log

python caption_generator.py \
  --model-path ${CKPT} \
  --video-folder ${BASE}/clips \
  --jsonl-file ${BASE}/index.jsonl \
  --result-file ${BASE}/results/captions_chunk_${IDX}.jsonl \
  --batch-size 16 \
  --num-chunks ${CHUNKS} \
  --chunk-idx ${IDX} \
  > "$LOG_FILE" 2>&1 &
```

---

### 3) Merge Caption Results (`caption_merger.py`)

Merges per-chunk caption JSONLs into one final file.

**Key flags:**
`--cap-file` final merged JSONL • `--result-file` input chunks (repeat or glob) • `--num-workers`

**Example**

```bash
export BASE=/path/to/workdir

python caption_merger.py \
  --cap-file ${BASE}/results/captions_merged.jsonl \
  --result-file "${BASE}/results/captions_chunk_*.jsonl" \
  --num-workers 8
```

---

### Notes

* Replace sample paths with your actual data locations.
* Outputs from step 1 include `motion.csv`, `segments.json`, `segments_sec.json`, `plot.png`, and optional `clips/*.mp4`.
