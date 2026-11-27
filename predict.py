"""Load the submitted model and predict the gyroscope state at one time."""

import argparse
import json
import math
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="输入一个时刻，预测重力陀螺的状态。")
    parser.add_argument("t", nargs="?", type=float, help="时间，单位为秒。也可运行后输入。")
    args = parser.parse_args()

    model_path = Path(__file__).resolve().with_name("gyro_pinn.pt")
    extra_files = {"metadata.json": ""}
    try:
        with model_path.open("rb") as model_file:
            model = torch.jit.load(model_file, map_location="cpu", _extra_files=extra_files)
    except FileNotFoundError:
        parser.error("未找到 gyro_pinn.pt，请将它与 predict.py 放在同一文件夹。")
    metadata = json.loads(extra_files["metadata.json"])
    start, end = metadata["time_range_s"]

    t = args.t
    if t is None:
        try:
            t = float(input(f"请输入时间 t（{start:g} 到 {end:g} 秒）："))
        except (ValueError, EOFError):
            parser.error("请输入一个数字，例如 0.037。")
    if not math.isfinite(t) or not start <= t <= end:
        parser.error(f"时间 t 必须在 [{start:g}, {end:g}] 秒内。")

    model.eval()
    with torch.inference_mode():
        time = torch.tensor([[t]], dtype=torch.float64)
        rotation, omega = model(time)
        lever = torch.tensor(metadata["lever_body_m"], dtype=rotation.dtype)
        position = rotation[0] @ lever

    print(f"t = {t!r} s")
    print("姿态矩阵 R（物体坐标 → 空间坐标）")
    for row in rotation[0].tolist():
        print("  [" + ", ".join(f"{value: .9f}" for value in row) + "]")
    print("物体坐标系角速度 ω（rad/s）", omega[0].tolist())
    print("空间坐标系质心位置 r（m）", position.tolist())


if __name__ == "__main__":
    main()
