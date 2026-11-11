"""Export a local PyTorch checkpoint to ONNX."""

import argparse
from os import link
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="将本地 YOLO 权重导出为 ONNX")
    parser.add_argument("--model", type=Path, default=ROOT / "best.pt", help="源 .pt 权重")
    parser.add_argument("--output", type=Path, help="输出文件，默认 runs/export/<权重名称>.onnx")
    parser.add_argument("--imgsz", type=int, default=640, help="导出时的输入尺寸")
    parser.add_argument("--opset", type=int, help="ONNX opset，默认由 Ultralytics 选择")
    parser.add_argument("--dynamic", action="store_true", help="导出动态形状")
    parser.add_argument("--no-simplify", action="store_true", help="跳过 ONNX 图简化")
    parser.add_argument("--overwrite", action="store_true", help="允许替换已有的输出文件")
    args = parser.parse_args()
    if args.model.suffix.lower() != ".pt" or not args.model.is_file():
        parser.error(f"找不到 .pt 权重：{args.model}")
    if args.imgsz <= 0 or (args.opset is not None and args.opset <= 0):
        parser.error("--imgsz 和 --opset 必须大于 0")
    output = args.output or ROOT / "runs" / "export" / args.model.with_suffix(".onnx").name
    if output.suffix.lower() != ".onnx":
        parser.error("--output 必须以 .onnx 结尾")
    output = output.absolute()
    if output.exists() and not args.overwrite:
        parser.error(f"输出已存在：{output}，请指定新路径或使用 --overwrite")

    import onnx
    from ultralytics import YOLO

    output.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics writes beside the checkpoint. Stage both files in the destination filesystem.
    with TemporaryDirectory(dir=output.parent) as temporary:
        checkpoint = Path(temporary) / args.model.name
        copyfile(args.model, checkpoint)
        exported = Path(
            YOLO(str(checkpoint)).export(
                format="onnx",
                imgsz=args.imgsz,
                opset=args.opset,
                dynamic=args.dynamic,
                simplify=not args.no_simplify,
                device="cpu",
            )
        )
        onnx.checker.check_model(str(exported))
        if args.overwrite:
            exported.replace(output)
        else:
            try:
                link(exported, output)
            except FileExistsError:
                parser.error(f"输出已存在：{output}，请指定新路径或使用 --overwrite")
    print(f"ONNX 导出与图校验完成：{output}")


if __name__ == "__main__":
    main()

