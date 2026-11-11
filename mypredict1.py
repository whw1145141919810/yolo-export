"""Run ONNX Runtime inference on local media."""

import argparse
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def device_name(value):
    if value != "cpu" and not (value.isascii() and value.isdecimal()):
        raise argparse.ArgumentTypeError("设备须为 cpu 或 CUDA GPU 编号，例如 0")
    return value


def prepare_runtime(device):
    import onnxruntime

    if device != "cpu":
        if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            raise RuntimeError("ONNX Runtime 缺少 CUDAExecutionProvider，请按 README 安装 GPU 运行时")
        onnxruntime.preload_dlls()


def execution_provider(model, device):
    provider = model.predictor.model.session.get_providers()[0]
    if device != "cpu" and provider != "CUDAExecutionProvider":
        raise RuntimeError(f"请求 CUDA，但 ONNX Runtime 使用了 {provider}。请检查 CUDA 和 cuDNN")
    return provider


def main():
    parser = argparse.ArgumentParser(description="用 ONNX 模型对本地视频、图片或目录做推理")
    parser.add_argument("source", type=Path, help="输入视频、图片或目录")
    parser.add_argument("--model", type=Path, default=ROOT / "best.onnx", help="ONNX 模型")
    parser.add_argument("--device", type=device_name, default="cpu", help="cpu 或 CUDA GPU 编号，例如 0")
    parser.add_argument("--imgsz", type=int, help="动态模型的推理输入尺寸，默认 640")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "onnx", help="输出目录")
    parser.add_argument("--show", action="store_true", help="实时显示推理结果")
    args = parser.parse_args()
    if not args.source.exists():
        parser.error(f"输入路径不存在：{args.source}")
    if args.model.suffix.lower() != ".onnx" or not args.model.is_file():
        parser.error(f"找不到 .onnx 模型：{args.model}")
    if (args.imgsz is not None and args.imgsz <= 0) or not 0 <= args.conf <= 1:
        parser.error("--imgsz 必须大于 0，--conf 必须在 0 到 1 之间")

    from ultralytics import YOLO

    try:
        prepare_runtime(args.device)
        model = YOLO(str(args.model.resolve()))
        size = {"imgsz": args.imgsz} if args.imgsz is not None else {}
        results = model.predict(
            source=str(args.source.resolve()),
            device=args.device,
            conf=args.conf,
            stream=True,
            save=True,
            show=args.show,
            project=str(args.project.resolve()),
            name="predict",
            verbose=False,
            **size,
        )
        with closing(results):
            if next(results, None) is None:
                parser.exit(1, "输入中没有可处理的图像或视频帧。\n")
            provider = execution_provider(model, args.device)
            frames = 1 + sum(1 for _ in results)
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")
    print(f"已处理 {frames} 张图像或视频帧（{provider}）。结果目录：{model.predictor.save_dir}")


if __name__ == "__main__":
    main()
