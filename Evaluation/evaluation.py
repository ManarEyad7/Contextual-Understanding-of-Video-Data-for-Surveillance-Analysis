import json
import numpy as np

FPS = 24
TIoU_THRESHOLD = 0.5  # tIoU threshold 

# --- GT JSON reorder ---
def reorder_gt_json(gt_json, video_key):
    """Sort GT timestamps & sentences by start time."""
    if "sentences" in gt_json[video_key]:
        timestamps = gt_json[video_key]['timestamps']
        sentences = gt_json[video_key]['sentences']
        combined = sorted(zip(timestamps, sentences), key=lambda x: x[0][0])
        sorted_timestamps, sorted_sentences = zip(*combined)
        gt_json[video_key]['timestamps'] = list(sorted_timestamps)
        gt_json[video_key]['sentences'] = list(sorted_sentences)
    else:
        gt_json[video_key]['timestamps'] = sorted(gt_json[video_key]['timestamps'], key=lambda x: x[0])
    return gt_json

# --- Frame conversion based on FPS ---
def seconds_to_frames(segment_seconds):
    start_sec, end_sec = segment_seconds
    return (int(round(start_sec * FPS)), int(round(end_sec * FPS)))


# --- tIoU functions ---
def compute_tIoU(seg1, seg2):
    """Compute temporal IoU between two segments [start, end]."""
    inter_start = max(seg1[0], seg2[0])
    inter_end = min(seg1[1], seg2[1])
    intersection = max(0, inter_end - inter_start)
    union = max(seg1[1], seg2[1]) - min(seg1[0], seg2[0])
    return intersection / union if union > 0 else 0

def evaluate_segments_tIoU(gt_segments, pred_segments, threshold=TIoU_THRESHOLD):
    """Evaluate Precision, Recall, F1 at a given tIoU threshold."""
    matched_gt = set()
    matched_pred = set()
    
    for i, pred in enumerate(pred_segments):
        for j, gt in enumerate(gt_segments):
            if j in matched_gt:  # GT already matched
                continue
            if compute_tIoU(pred, gt) >= threshold:
                matched_pred.add(i)
                matched_gt.add(j)
                break
    
    TP = len(matched_pred)
    FP = len(pred_segments) - TP
    FN = len(gt_segments) - TP

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

# --- JSON loaders ---
def load_gt_segments_from_json(gt_json, video_key):
    return [seconds_to_frames(ts) for ts in gt_json[video_key]['timestamps']]

def load_pred_segments_from_json(pred_json, video_key):
    return [seconds_to_frames(seg) for seg in pred_json[video_key]["segments"]]

# --- Main function ---
def main(gt_path, pred_path, video_key):
    with open(gt_path) as f: gt_json = json.load(f)
    with open(pred_path) as f: pred_json = json.load(f)
    gt_json = reorder_gt_json(gt_json, video_key)

    gt_segments = load_gt_segments_from_json(gt_json, video_key)
    pred_segments = load_pred_segments_from_json(pred_json, video_key)
    video_duration = gt_json[video_key].get("duration", None)

    # tIoU threshold evaluation
    tIoU_precision, tIoU_recall, tIoU_f1 = evaluate_segments_tIoU(gt_segments, pred_segments)

    # Print results
    print(f"--- tIoU metrics (threshold={TIoU_THRESHOLD}) ---")
    print(f"Precision: {tIoU_precision:.4f}  Recall: {tIoU_recall:.4f}  F1: {tIoU_f1:.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate video segmentation.')
    parser.add_argument('--gt', required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--video', required=True)
    args = parser.parse_args()
    main(args.gt, args.pred, args.video)
