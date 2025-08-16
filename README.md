# KAUST Surveillance Analysis Project

This repository contains code for motion-based segmentation and video analysis using various models, including RAFT, for surveillance analysis. The goal of this project is to process and segment videos for motion analysis.

## Steps to Clone and Set Up

### 1. Clone the Repository

To get started with the project, clone the repository using the following command:

```bash
git clone git@github.com:ManarEyad7/kaust-surv-analysis.git
cd kaust-surv-analysis
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies. You can create a virtual environment using **Conda** or **virtualenv**.

#### Using Conda:

```bash
conda create --name kaust-surv python=3.10
conda activate kaust-surv
```

#### Using `virtualenv`:

```bash
python3 -m venv kaust-surv
source kaust-surv/bin/activate  # On Linux/macOS
kaust-surv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

Once the environment is activated, install the required dependencies. If you have a `requirements.txt` file, use the following command:

```bash
pip install -r requirements.txt
```

### 4. Running the Code

To run the motion-based segmentation script, use the following command:

```bash
python LoVR/motion_based_clip_segmentation.py \
  --model models/raft-things.pth \
  --video /path/to/your/video.mp4 \
  --out_dir /path/to/output/dir \
  --iters 8 \
  --max_side 512 \
  --savgol_window 31 \
  --savgol_poly 3 \
  --z_k 2.0 \
  --min_gap 12 \
  --min_seg_len 24 \
  --export_clips
```
