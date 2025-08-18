

# KAUST Surveillance Analysis Project

This repository contains code for motion-based segmentation and video analysis using various models, including RAFT, for surveillance analysis. The goal of this project is to process and segment videos for motion analysis.

## Steps to Clone and Set Up

### 1. Clone the Repository

To get started with the project, clone the repository using the following command:

```bash
git clone git@github.com:ManarEyad7/kaust-surv-analysis.git
cd kaust-surv-analysis
```

### 2. Create a Conda Virtual Environment

It's recommended to use a Conda virtual environment to manage dependencies. You can create and activate the environment using the following commands:

```bash
conda create --name kaust-surv python=3.10
conda activate kaust-surv
```

### 3. Install Python Dependencies

Once the environment is activated, install the required Python dependencies from the `requirements.txt` file by running:

```bash
pip install -r requirements.txt
```

### 4. Install ffmpeg

This project requires `ffmpeg` for video processing. You can install `ffmpeg` in your Conda environment by running:

```bash
conda install -c conda-forge ffmpeg
```

This will install the necessary dependencies for video processing.

### 5. Download Pretrained Models

Navigate to the RAFT directory and download the pretrained models:

```bash
cd RAFT
./RAFT/models.zip
```

### 6. Running the Code

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

---

### Additional Notes

* **Installing Dependencies**: It's crucial to install `ffmpeg` using Conda as part of setting up the environment.
* **Video Files**: Replace `/path/to/your/video.mp4` with the actual path to your video file.
* **Output Directory**: Set the output directory path to where you want the segmentation results to be saved.

---

This README structure ensures users understand the steps to set up the environment, install `ffmpeg`, and run the code smoothly. Let me know if you need more details!
