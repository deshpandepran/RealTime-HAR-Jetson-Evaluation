"""
Live / offline demo of the RT-HARE pipeline.

Pipeline per clip of K=6 frames:
    resize (256x344) + normalise -> IMFE (motion, 2048-d)
                                 -> ResNet-50 on the centre frame (RGB, 2048-d)
    concat -> feature buffer -> LSTR -> (optional) hybrid post-processing -> label

Run from the repository root (extensions/live_demo.py):

    python extensions/live_demo.py                       # webcam 0
    python extensions/live_demo.py --video clip.mp4      # a video file
    python extensions/live_demo.py --video clip.mp4 --no-display --max-clips 20

Without checkpoints the weights are random, so predictions are meaningless;
the run only checks that the pipeline works and measures latency.
"""
import argparse
import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.preprocess import VideoPreprocessor  # noqa: E402
from models.imfe import IMFE  # noqa: E402
from models.lstr import LongShortTermTransformer  # noqa: E402

try:
    from models.post_processing import HybridPostProcessor  # noqa: E402
except Exception:  # keep the demo usable if the module changes
    HybridPostProcessor = None


def build_rgb_extractor(device):
    """ResNet-50 with the classifier removed -> 2048-d feature."""
    import torchvision

    try:
        net = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
    except Exception as e:  # no internet / old torchvision
        print(f"[warn] could not load ImageNet weights ({e}); using random weights")
        net = torchvision.models.resnet50(weights=None)
    net.fc = torch.nn.Identity()
    return net.eval().to(device)


def load_ckpt(model, path, name):
    if not path:
        print(f"[warn] no {name} checkpoint given -> random weights")
        return
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[info] loaded {name} from {path} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def predict_label(logits_seq, post, fps_clip):
    """logits_seq: (T, C) tensor -> (label_index, confidence) for the newest step."""
    probs = torch.softmax(logits_seq[-1], dim=-1)
    label = int(probs.argmax())
    if post is not None:
        try:
            smoothed = post(logits_seq.cpu().numpy())
            label = int(np.asarray(smoothed)[-1])
        except Exception as e:
            print(f"[warn] post-processing failed once ({e}); using raw argmax")
            return label, float(probs[label])
    return label, float(probs[label])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--video", type=str, default=None, help="video file instead of webcam")
    ap.add_argument("--imfe-ckpt", type=str, default=None)
    ap.add_argument("--lstr-ckpt", type=str, default=None)
    ap.add_argument("--labels", type=str, default=None, help="text file, one class name per line")
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--clip-length", type=int, default=6)
    ap.add_argument("--buffer", type=int, default=80, help="feature buffer length (clips)")
    ap.add_argument("--rgb-dim", type=int, default=2048)
    ap.add_argument("--no-post", action="store_true", help="disable hybrid post-processing")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-clips", type=int, default=0, help="stop after N clips (0 = no limit)")
    ap.add_argument("--latency-json", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = args.clip_length
    print(f"[info] device: {device}")

    if args.labels:
        with open(args.labels) as f:
            names = [l.strip() for l in f if l.strip()]
    elif args.num_classes == 2:
        names = ["normal", "anomaly"]
    else:
        names = [f"class_{i}" for i in range(args.num_classes)]

    imfe = IMFE(clip_length=K).eval().to(device)
    rgb_net = build_rgb_extractor(device)
    lstr = LongShortTermTransformer(
        in_dim=2048 + args.rgb_dim, num_classes=args.num_classes, causal=True
    ).eval().to(device)
    load_ckpt(imfe, args.imfe_ckpt, "IMFE")
    load_ckpt(lstr, args.lstr_ckpt, "LSTR")

    pre = VideoPreprocessor()
    src = args.video if args.video else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {src}")
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if cam_fps < 1 or cam_fps > 240:
        cam_fps = 30.0
    deadline_ms = K * 1000.0 / cam_fps
    post = None
    if HybridPostProcessor is not None and not args.no_post:
        post = HybridPostProcessor(fps=max(1, int(round(cam_fps / K))))
    print(f"[info] source fps ~{cam_fps:.1f}, deadline per clip = {deadline_ms:.0f} ms")

    feats = deque(maxlen=args.buffer)
    clip_frames = []
    latencies = []
    shown = "waiting..."
    last_lat = 0.0
    n_clips = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        clip_frames.append(pre.resize_frame(rgb))

        if len(clip_frames) == K:
            sync(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                clip = pre.process_clip(clip_frames)            # (K, 3, H, W)
                x = clip.unsqueeze(0).to(device)                # (1, K, 3, H, W)
                motion = imfe(x)                                # (1, 2048)
                rgb_feat = rgb_net(x[:, K // 2])                # (1, rgb_dim)
                feats.append(torch.cat([motion, rgb_feat], dim=1).squeeze(0))
                seq = torch.stack(list(feats)).unsqueeze(0)     # (1, T, in_dim)
                logits = lstr(seq)[0]                           # (T, C)
                label, conf = predict_label(logits, post, cam_fps / K)
            sync(device)
            last_lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(last_lat)
            shown = f"{names[label] if label < len(names) else label} ({conf:.2f})"
            clip_frames = []
            n_clips += 1
            print(f"clip {n_clips:4d}  {shown:<24s} {last_lat:7.1f} ms")
            if args.max_clips and n_clips >= args.max_clips:
                break

        if not args.no_display:
            vis = frame_bgr.copy()
            color = (0, 200, 0) if last_lat <= deadline_ms else (0, 0, 255)
            cv2.putText(vis, f"Action: {shown}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis, f"Latency: {last_lat:.0f} ms (deadline {deadline_ms:.0f} ms)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("RT-HARE live demo (press q to quit)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if latencies:
        lat = np.array(latencies)
        miss = float((lat > deadline_ms).mean()) * 100
        print("\n=== latency summary (end-to-end per clip, incl. RGB + LSTR) ===")
        print(f"clips: {len(lat)}  mean: {lat.mean():.1f} ms  std: {lat.std():.1f} ms  "
              f"p95: {np.percentile(lat, 95):.1f} ms  deadline miss: {miss:.1f}%")
        if args.latency_json:
            with open(args.latency_json, "w") as f:
                json.dump({"device": str(device), "deadline_ms": deadline_ms,
                           "latencies_ms": latencies}, f, indent=2)
            print(f"saved {args.latency_json}")


if __name__ == "__main__":
    main()
