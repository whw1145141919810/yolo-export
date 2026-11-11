# RoboMaster 装甲板模型 ONNX 导出与对比

将 YOLO26n-pose 的 PyTorch 权重导出为 ONNX，用同一段视频比较两个模型的推理耗时和检测数量。仓库提供可直接运行的 [best.onnx](best.onnx)，其输入为 FP32、`1 × 3 × 640 × 640`，可检测 B1、B2、B3、R1、R2、R3 六类装甲板，每个装甲板输出四个关键点。

## 安装与试跑

从仓库根目录执行。以下示例适用于 Windows PowerShell，原项目环境使用 Python 3.13.15。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe mypredict1.py "videos\armor.mp4"
```

将 `videos\armor.mp4` 换成自己的视频、图片或目录。默认用 CPU 和仓库自带的 `best.onnx`。脚本逐帧保存推理结果，结束时打印实际的 ONNX Runtime 后端和输出目录（默认在 `runs/onnx/predict*/`）。Ultralytics 导出的静态模型按模型元数据设置输入尺寸，动态输入模型的默认尺寸为 640，可用 `--imgsz` 调整。

## 导出自己的模型

原始 `.pt` 权重由社团提供，获取和放置方式见 [yolo](https://github.com/whw1145141919810/yolo)。下面的命令假设两个仓库在同一父目录下，且权重在 `yolo/weights/best.pt`。

```powershell
.\.venv\Scripts\python.exe demo.py --model "..\yolo\weights\best.pt"
.\.venv\Scripts\python.exe mypredict1.py "videos\armor.mp4" --model "runs\export\best.onnx"
```

默认以 CPU 导出 640×640 的静态图，完成 ONNX 图校验后写入 `runs/export/`。可通过 `--output` 指定路径，或通过 `--dynamic` 导出动态输入。仅当明确传入 `--overwrite` 时才允许覆盖已有文件。

## 用同一视频复跑对比

```powershell
.\.venv\Scripts\python.exe benchmark.py "videos\armor.mp4" --pt "..\yolo\weights\best.pt" --onnx "runs\export\best.onnx"
```

脚本以同一个视频、设备、输入尺寸和阈值运行两个模型，输出可复制的 Markdown 表格。默认跳过每个模型前 5 帧的计时，并统计全视频目标数。平均耗时取自 Ultralytics 的推理阶段，视频解码、预处理和后处理各自单独计时。目标数是逐帧累加的检测框数。静态模型使用 640 以外的输入尺寸时，传入匹配的 `--imgsz`。

### 使用 NVIDIA GPU

完成上面的安装后，在本仓库的虚拟环境中安装对应 CUDA 版本的 PyTorch，并切换到 GPU 版 ONNX Runtime。两个 Runtime 安装包共享同一个 Python 模块，虚拟环境中应保留对应设备的版本。以下命令使用原项目环境的 CUDA 13.0 构建。

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip uninstall -y onnxruntime
.\.venv\Scripts\python.exe -m pip install onnxruntime-gpu==1.31.0
.\.venv\Scripts\python.exe mypredict1.py "videos\armor.mp4" --device 0
.\.venv\Scripts\python.exe benchmark.py "videos\armor.mp4" --pt "..\yolo\weights\best.pt" --device 0
```

### GPU 复测

在 RTX 5060 Laptop GPU 上用原始演示视频和仓库自带的 `best.onnx` 复测。PyTorch 使用 2.14.1+cu130，ONNX Runtime 使用 1.31.0 CUDA 后端。两个模型各处理 399 帧，平均耗时按跳过前 5 帧后的 394 帧计算。

| 指标 | PyTorch `.pt` | ONNX `.onnx` |
| --- | ---: | ---: |
| 平均推理耗时 | 9.17 ms/帧 | 6.04 ms/帧 |
| 累计检测目标数 | 630 | 630 |

## 原始实验记录

以下是历史记录，采用当时脚本的全流程计时。当前 `benchmark.py` 只统计推理阶段，两者的计时口径不同。

| 指标 | PyTorch `.pt` | ONNX `.onnx` |
| --- | ---: | ---: |
| 模型文件大小 | 5.62 MiB | 9.72 MiB |
| 同一次实验的平均耗时 | 21.76 ms/帧 | 17.99 ms/帧 |
| 同一次实验的累计检测目标数 | 625 | 630 |
| 另一份记录的平均耗时 | 22.83 ms/帧（43.80 FPS） | 19.29 ms/帧（51.83 FPS） |

| PyTorch 推理结果 | ONNX 推理结果 |
| --- | --- |
| <img width="420" alt="PyTorch 原始权重的推理画面" src="https://github.com/user-attachments/assets/b3cd6c35-522a-429d-afd5-ca7ae0c5e932" /> | <img width="420" alt="ONNX 导出模型的推理画面" src="https://github.com/user-attachments/assets/9923450a-5d9e-4e8c-915f-3c31d0f0e86b" /> |
