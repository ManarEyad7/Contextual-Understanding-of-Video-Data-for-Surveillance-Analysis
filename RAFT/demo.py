import sys
sys.path.append('core')

import argparse
import os
import cv2
import os
frame_idx = 0

def save_frame(img_flo):
    global frame_idx
    os.makedirs('demo_out', exist_ok=True)
    cv2.imwrite(f'demo_out/{frame_idx:06d}.png', img_flo[:, :, [2,1,0]].astype('uint8'))
    frame_idx += 1

import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    """Write visualization frames straight to a video file."""
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    from core.utils import flow_viz
    flo = flow_viz.flow_to_image(flo)

    import cv2
    import numpy as np

    img_flo = 0.5 * img + 0.5 * flo
    img_flo = np.clip(img_flo, 0, 255).astype(np.uint8)

    if not hasattr(viz, "writer"):
        out_path = getattr(args, "out_video", "demo_out.mp4")
        fps = getattr(args, "fps", 24)
        h, w = img_flo.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        viz.writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not viz.writer.isOpened():
            out_path = out_path.rsplit(".", 1)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            viz.writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            assert viz.writer.isOpened(), "Failed to open VideoWriter. Try a different path/codec."
    viz.writer.write(img_flo[:, :, ::-1])

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)
    # Release writer if created
    if hasattr(viz, 'writer'):
        viz.writer.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--out_video', type=str, default='demo_out.mp4')
    parser.add_argument('--fps', type=int, default=24)
    args = parser.parse_args()

    demo(args)