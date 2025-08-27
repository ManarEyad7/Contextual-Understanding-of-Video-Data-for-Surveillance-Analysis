import sys, os, argparse
sys.path.append('core')  # RAFT repo layout: core/raft.py, core/utils/*
import cv2
import numpy as np
import torch
from tqdm import tqdm

from core.raft import RAFT
from core.utils.utils import InputPadder
from core.utils import flow_viz  # has flow_to_image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def _resize_keep_ar(frame, max_side=None):
    """Resize keeping aspect ratio so that max(H,W) == max_side (if provided)."""
    if max_side is None or max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return frame
    scale = max_side / float(m)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def _to_tensor_bgr_uint8(frame_bgr):
    # BGR uint8 -> torch float tensor [1,3,H,W]
    t = torch.from_numpy(frame_bgr).permute(2, 0, 1).float()[None]
    return t.to(DEVICE)

def ensure_writer(path, fps, size_wh):
    """Create cv2.VideoWriter with mp4 fallback to avi."""
    w, h = size_wh
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if writer.isOpened():
        return writer, path
    base, _ = os.path.splitext(path)
    path2 = base + ".avi"
    fourcc2 = cv2.VideoWriter_fourcc(*"MJPG")
    writer2 = cv2.VideoWriter(path2, fourcc2, fps, (w, h))
    assert writer2.isOpened(), "Failed to open VideoWriter. Try different codec/path."
    return writer2, path2

@torch.no_grad()
def run_on_video(args):
    # ----- Load model -----
    model = torch.nn.DataParallel(RAFT(args))
    ckpt = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model = model.module.to(DEVICE).eval()

    # ----- Open video -----
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open video: {args.video}"

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = args.fps if args.fps > 0 else (src_fps if src_fps and src_fps > 0 else 24)

    # Read first frame
    ok, prev_bgr = cap.read()
    assert ok, "Empty video / failed to read the first frame."
    prev_bgr = _resize_keep_ar(prev_bgr, args.max_side)

    # Prepare writer after we know processed size
    writer, out_path = ensure_writer(args.out_video, out_fps, (prev_bgr.shape[1], prev_bgr.shape[0]))
    print(f"Writing to: {out_path} @ {out_fps:.2f} fps")

    # Stride (process every Nth frame pair)
    stride = max(1, args.stride)

    # Progress bar setup (estimated total pairs if available)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # may return 0 for some codecs
    total_pairs_est = (frame_count // stride - 1) if frame_count > 0 else None
    pbar = tqdm(total=total_pairs_est, desc="Processing frame pairs", unit="pair", leave=True)

    total_pairs = 0

    try:
        while True:
            # Optionally skip frames for speed
            for _ in range(stride - 1):
                if not cap.grab():
                    prev_bgr = None
                    break
            if prev_bgr is None:
                break

            ok, cur_bgr = cap.read()
            if not ok:
                break
            cur_bgr = _resize_keep_ar(cur_bgr, args.max_side)

            # Keep sizes equal (guard against decoder hiccups)
            if cur_bgr.shape != prev_bgr.shape:
                cur_bgr = cv2.resize(cur_bgr, (prev_bgr.shape[1], prev_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Tensors
            im1 = _to_tensor_bgr_uint8(prev_bgr)
            im2 = _to_tensor_bgr_uint8(cur_bgr)

            # Pad to be divisible by 8 (RAFT expects this)
            padder = InputPadder(im1.shape)
            im1p, im2p = padder.pad(im1, im2)

            # Inference
            flow_low, flow_up = model(im1p, im2p, iters=args.iters, test_mode=True)

            # Visualize flow on first frame of the pair
            f = flow_up[0].permute(1, 2, 0).detach().cpu().numpy()  # [H,W,2]
            flow_rgb = flow_viz.flow_to_image(f)                    # uint8 RGB
            overlay = 0.5 * prev_bgr.astype(np.float32) + 0.5 * flow_rgb[:, :, ::-1].astype(np.float32)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            writer.write(overlay)  # OpenCV expects BGR
            total_pairs += 1
            pbar.update(1)

            # Slide window
            prev_bgr = cur_bgr

        if args.flush_last and prev_bgr is not None:
            writer.write(prev_bgr)

    finally:
        pbar.close()
        cap.release()
        writer.release()
        print(f"Done. Flow pairs processed: {total_pairs}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("RAFT video demo (with tqdm progress bar)")
    ap.add_argument("--model", required=True, help="Path to RAFT checkpoint (e.g., models/raft-things.pth)")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--out_video", default="demo_out.mp4", help="Output video path")
    ap.add_argument("--iters", type=int, default=20, help="RAFT iterations per pair")
    ap.add_argument("--max_side", type=int, default=0, help="Resize so max(H,W)=max_side; 0=keep original")
    ap.add_argument("--fps", type=float, default=0, help="Output FPS; 0=use source FPS or 24 if unknown")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame pair (speed/less smooth)")
    ap.add_argument("--flush_last", action="store_true", help="Write last frame again to end the video")
    # RAFT flags (forwarded to RAFT __init__)
    ap.add_argument("--small", action="store_true", help="Use small model (needs small checkpoint)")
    ap.add_argument("--mixed_precision", action="store_true", help="Use torch.amp autocast in RAFT")
    ap.add_argument("--alternate_corr", action="store_true", help="Efficient correlation implementation")
    args = ap.parse_args()

    run_on_video(args)
