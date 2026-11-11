"""Compare PT and ONNX inference on the same local video."""

import argparse
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from mypredict1 import ROOT, device_name, execution_provider, prepare_runtime


@dataclass
class Measurement:
    frames: int
    objects: int
    mean_ms: float
    backend: str


def benchmark(weights, args):
    from ultralytics import YOLO

    model = YOLO(str(weights.resolve()))
    results = model.predict(
        source=str(args.video.resolve()),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        rect=False,
        quantize=32,
        stream=True,
        save=False,
        verbose=False,
    )
    frames = objects = 0
    inference_ms = 0.0
    backend = ""
    with closing(results):
        for result in results:
            if frames == 0:
                backend = (
                    execution_provider(model, args.device)
                    if weights.suffix.lower() == ".onnx"
                    else f"PyTorch ({model.predictor.device})"
                )
                shape = tuple(model.predictor.imgsz)
                if shape != (args.imgsz, args.imgsz):
                    raise ValueError(f"{weights.name} 使用输入尺寸 {shape}，请指定与模型一致的 --imgsz")
            if result.boxes is None:
                raise ValueError("基准对比需要输出检测框的模型，例如装甲板姿态模型")
            frames += 1
            objects += len(result.boxes)
            if frames > args.warmup:
                inference_ms += result.speed["inference"]
    timed_frames = frames - args.warmup
    if timed_frames <= 0:
        raise ValueError(f"{weights.name} 只读取到 {frames} 帧，请检查视频或减小 --warmup")
    return Measurement(frames, objects, inference_ms / timed_frames, backend)


def main():
    parser = argparse.ArgumentParser(description="使用同一个视频比较 PT / ONNX 的推理耗时和目标数")
    parser.add_argument("video", type=Path, help="用于对比的本地视频文件")
    parser.add_argument("--pt", type=Path, required=True, help="源 .pt 权重")
    parser.add_argument("--onnx", type=Path, default=ROOT / "best.onnx", help="对应的 ONNX 模型")
    parser.add_argument("--device", type=device_name, default="cpu", help="cpu 或 CUDA GPU 编号，例如 0")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸，须与静态 ONNX 模型一致")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--warmup", type=int, default=5, help="每个模型跳过的计时帧数")
    args = parser.parse_args()
    for path, suffix in ((args.pt, ".pt"), (args.onnx, ".onnx")):
        if path.suffix.lower() != suffix or not path.is_file():
            parser.error(f"找不到 {suffix} 模型：{path}")
    if not args.video.is_file():
        parser.error(f"找不到输入视频：{args.video}")
    if args.imgsz <= 0 or args.warmup < 0 or not (0 <= args.conf <= 1 and 0 <= args.iou <= 1):
        parser.error("输入尺寸必须大于 0，预热帧数必须非负，阈值必须在 0 到 1 之间")

    from ultralytics.data.utils import VID_FORMATS

    if args.video.suffix.lower().lstrip(".") not in VID_FORMATS:
        parser.error(f"输入文件须为 Ultralytics 支持的视频格式：{args.video}")
    try:
        prepare_runtime(args.device)
        pt = benchmark(args.pt, args)
        onnx = benchmark(args.onnx, args)
        if pt.frames != onnx.frames:
            raise ValueError(f"两个模型处理的帧数不同（PT {pt.frames}，ONNX {onnx.frames}）")
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    print()
    print(f"视频 {args.video.name} · 输入 {args.imgsz}×{args.imgsz} · conf={args.conf} · iou={args.iou}")
    print(f"每个模型跳过前 {args.warmup} 帧的计时，统计 {pt.frames - args.warmup} 帧。")
    print()
    print("| 格式 | 后端 | 文件大小 (MiB) | 帧数 | 目标数 | 平均推理耗时 (ms/帧) |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for weights, measurement in ((args.pt, pt), (args.onnx, onnx)):
        size = weights.stat().st_size / (1024 ** 2)
        print(
            f"| {weights.suffix} | {measurement.backend} | {size:.2f} | {measurement.frames} | "
            f"{measurement.objects} | {measurement.mean_ms:.2f} |"
        )


if __name__ == "__main__":
    main()
