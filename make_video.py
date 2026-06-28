#!/usr/bin/env python3
import argparse, glob, os
import cv2

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  default="")
    p.add_argument("--out",        default="")
    p.add_argument("--fps",        type=int, default=24)
    p.add_argument("--pattern",    default="*.png", help=" (*.png, frame*.png)")
    args = p.parse_args()

    frames = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not frames:
        raise SystemExit(f"No images found under {args.input_dir}/{args.pattern}")


    sample = cv2.imread(frames[0], cv2.IMREAD_COLOR)
    if sample is None:
        raise SystemExit(f"Failed to read: {frames[0]}")
    h, w = sample.shape[:2]


    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
    if not vw.isOpened():
        raise SystemExit("Failed to open VideoWriter. Try a different filename or codec.")

    for i, fp in enumerate(frames, 1):
        img = cv2.imread(fp, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            print(f"[skip] cannot read {fp}")
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(img)
        if i % 100 == 0:
            print(f"{i}/{len(frames)}")

    vw.release()
    print(f"Saved video: {args.out}")

if __name__ == "__main__":
    main()
