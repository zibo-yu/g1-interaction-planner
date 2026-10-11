from __future__ import annotations

import argparse
from pathlib import Path

from .model import build_g1_model
from .motion import MotionClip
from .playback import play_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a MotionClip .npz on G1.")
    parser.add_argument("clip", type=Path)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_g1_model(add_hand_target=True)
    clip = MotionClip.load(args.clip)
    play_clip(model, clip, speed=args.speed, loop=args.loop)


if __name__ == "__main__":
    main()
