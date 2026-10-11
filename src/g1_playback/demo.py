from __future__ import annotations

import argparse
from pathlib import Path

from .model import build_g1_model
from .playback import play_clip
from .trajectories import reach_clip, stand_clip, wave_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unitree G1 MuJoCo kinematic playback demo."
    )
    parser.add_argument(
        "--mode",
        choices=["stand", "wave", "reach"],
        default="reach",
        help="Built-in reference motion.",
    )
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--save-npz",
        type=Path,
        default=None,
        help="Optionally save the generated reference clip before playback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading Unitree G1 from MuJoCo Menagerie...")
    model = build_g1_model(add_hand_target=True)
    print(f"Loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}")

    if args.mode == "stand":
        clip = stand_clip(model, duration=args.duration, fps=args.fps)
    elif args.mode == "wave":
        clip = wave_clip(model, duration=args.duration, fps=args.fps)
    else:
        clip = reach_clip(model, duration=args.duration, fps=args.fps)

    if args.save_npz is not None:
        clip.save(args.save_npz)
        print(f"Saved reference motion to {args.save_npz}")

    print(
        f"Playing {clip.nframes} frames at {clip.fps:.1f} fps "
        f"({clip.duration:.2f} s reference duration)."
    )
    print("This is KINEMATIC reference playback: no tracker/dynamics are being tested.")
    play_clip(model, clip, speed=args.speed, loop=args.loop)


if __name__ == "__main__":
    main()
