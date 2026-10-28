-yolo模型版本：yolo26n
-导出模型格式：onnx
-运行环境与依赖库：见yolo/requests.txt
-原始<img width="1294" height="1094" alt="QQ20261028-180448" src="https://github.com/user-attachments/assets/b3cd6c35-522a-429d-afd5-ca7ae0c5e932" />
-导出<img width="1294" height="1094" alt="QQ20261028-180510" src="https://github.com/user-attachments/assets/9923450a-5d9e-4e8c-915f-3c31d0f0e86b" />
-对比：
原始 .pt  - 耗时: 21.76 ms, 检测数: 625
导出 .onnx - 耗时: 17.99 ms, 检测数: 630
| 模型格式 | .pt | .onnx |
| 模型文件大小 | 5.62 MB | 9.72 MB |
| 推理耗时 | 22.83 ms/帧 (FPS: 43.80) | 19.29 ms/帧 (FPS: 51.83) |


