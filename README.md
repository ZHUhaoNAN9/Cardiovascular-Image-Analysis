# CardioVision YOLO + MedSAM

> 心血管影像病变检测与分割研究项目  
> Research Project for Cardiovascular Lesion Detection and Segmentation

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Detection-111111)
![MedSAM](https://img.shields.io/badge/MedSAM-Detection-111111)
![FastAPI](https://img.shields.io/badge/FastAPI-Inference%20API-009688?logo=fastapi&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Demo-F97316?logo=gradio&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)

## 中文介绍

`CardioVision YOLO + MedSAM` 是一个面向心血管影像智能分析的研究型项目，目标是对病变区域进行自动检测与分割。项目围绕 `数据集构建 -> 模型训练 -> 融合推理 -> Web 演示` 四个环节展开，结合 `YOLO` 的检测能力和 `MedSAM` 的医学图像分割能力，形成了一套较完整的实验与演示流程。

当前仓库同时包含主流程脚本、实验配置、Notebook、报告生成脚本和部分中间产物。如果你准备把它上传到 GitHub，建议保留核心脚本，把草稿和生成物尽量移出主仓库。

### 项目亮点

- 支持从手术视频与 CVAT 标注到训练数据集的完整预处理流程
- 支持 YOLO 检测训练、MedSAM 微调训练与融合推理
- 提供 YOLO + MedSAM 两阶段推理与后处理逻辑
- 提供 `FastAPI + Gradio` 的本地演示方案
- 包含多种数据集切分与防数据泄露辅助脚本

### 功能概览

#### 1. 数据集构建

项目提供从原始视频和 CVAT XML 标注到训练集的完整处理链路，主要包括：

- 视频抽帧
- XML 标注诊断
- YOLO 标签与分割掩膜生成
- 图像与标签一致性清洗
- 训练集 / 验证集划分与 `data.yaml` 生成

#### 2. 病变检测与分割

项目以心血管影像中的典型病变为目标，当前脚本里主要围绕以下类别展开：

- `calcification`
- `fibre`
- `lipid`

部分数据处理脚本中还保留了：

- `damage`

该类别更多出现在早期数据构建与实验阶段，不一定属于最终三分类主流程。

#### 3. 融合推理

项目核心推理链路为：

- 第一步，YOLO 负责病变检测
- 第二步，将检测框作为提示输入 MedSAM
- 第三步，通过扩框、连通域筛选、Box Clipping 等策略优化分割结果

#### 4. 本地可视化演示

项目提供：

- `api.py`：FastAPI 推理接口
- `ui_gradio.py`：Gradio 可视化前端

适合用于课程展示、毕业设计答辩或本地 Demo 演示。

### 技术栈

- Python
- PyTorch
- Ultralytics YOLO
- MedSAM / Segment Anything
- OpenCV
- FastAPI
- Gradio
- NumPy
- Pillow
- Pandas
- Matplotlib

### 项目结构

```text
1A_Cardio_Codes/
├── create_dataset/             # 视频抽帧、XML 诊断、标签与掩膜生成、数据整理
├── pre_dataset/                # 早期数据裁剪与标签辅助脚本
├── COCO2DAtaset/               # COCO 到 Mask 的转换辅助脚本
├── api.py                      # FastAPI 推理后端
├── ui_gradio.py                # Gradio 可视化前端
├── Yolo_train.py               # YOLO 检测训练
├── MedSAM_train.py             # MedSAM 微调训练
├── Yolo+MedSAM_train.py        # 融合推理评估主脚本
├── Chunked_82_Dataset.py       # 分块切分，减少连续帧泄露
├── generate_charts.py          # 评估图表生成
├── Mask_Image.py               # 掩膜可视化
├── Config1.py ~ Config4.py     # 融合配置实验脚本
├── *.ipynb                     # 调参与实验记录
└── Readme.md
```

### 快速开始

#### 环境要求

- Python 3.10+
- 可用的 PyTorch 环境
- 如果要运行前端演示，建议本地具备完整桌面 / 浏览器环境
- 如果要进行训练或推理，需自行准备模型权重文件

#### 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch ultralytics opencv-python numpy pillow fastapi uvicorn gradio pandas matplotlib pyyaml python-docx requests
```

如需运行 MedSAM 相关代码，还需要额外准备 `segment-anything` 及相关模型环境。

#### 数据处理流程

建议按以下顺序运行：

```bash
python create_dataset/Video2imgs.py
python create_dataset/check.py
python create_dataset/XML2Mask.py
python create_dataset/Clean_Data.py
python create_dataset/organize_dataset.py
```

注意：这些脚本当前普遍使用硬编码路径，运行前请先修改脚本中的本地路径配置。

#### 模型训练

```bash
python Yolo_train.py
python MedSAM_train.py
python Yolo+MedSAM_train.py
```

训练前请确认：

- 数据集路径已改为你的本地路径
- `data.yaml` 中类别顺序与标签 ID 对齐
- YOLO、SAM、MedSAM 权重路径已正确配置

#### 启动演示系统

先启动后端：

```bash
uvicorn api:app --host 0.0.0.0 --port 8004
```

再启动前端：

```bash
python ui_gradio.py
```

默认前端会请求：

```text
http://127.0.0.1:8004/predict_json
```


### 当前已知问题

- 许多脚本仍使用 Windows 或 Linux 本地绝对路径，尚不可直接跨环境运行
- 仓库内同时存在主流程脚本与实验 Notebook，结构略混杂
- 目前缺少统一的 `requirements.txt`
- 目前缺少统一的配置文件，路径与参数分散在各脚本内部
- `Readme.md` 之外的文档命名和放置方式仍带有明显实验阶段痕迹

### 后续可优化方向

- 增加统一的 `requirements.txt`
- 增加 `.gitignore`
- 把路径参数收敛到统一配置文件
- 清理重复 Notebook 与旧实验脚本
- 增加推理结果截图、流程图与示例输入输出
- 将 `Readme.md` 统一改为 `README.md`


---

## English

`CardioVision YOLO + MedSAM` is a research-oriented project for cardiovascular image analysis. It covers the full workflow from dataset preparation to model training, fused inference, and local web demo. The main idea is to combine `YOLO` for lesion detection and `MedSAM` for medical image segmentation in a practical experimental pipeline.

The repository currently contains core scripts, experiment configurations, notebooks, report-generation files, and temporary artifacts. If you plan to publish it on GitHub, it is recommended to keep the core pipeline and move drafts or generated files elsewhere.

### Highlights

- End-to-end preprocessing pipeline from raw videos and CVAT annotations to training datasets
- Supports YOLO detection training, MedSAM fine-tuning, and fused inference
- Includes post-processing logic such as box padding, component filtering, and clipping
- Provides a local `FastAPI + Gradio` demo workflow
- Includes multiple dataset split utilities for stricter evaluation and leakage control

### Features

#### 1. Dataset Preparation

The repository includes scripts for:

- video frame extraction
- CVAT XML diagnosis
- YOLO label and mask generation
- image / label consistency cleaning
- train / validation split and `data.yaml` generation

#### 2. Lesion Detection and Segmentation

The main lesion categories used in the current pipeline are:

- `calcification`
- `fibre`
- `lipid`

Some earlier preprocessing scripts also contain:

- `damage`

which appears to belong more to earlier experiments than the final three-class pipeline.

#### 3. Fused Inference

The core inference workflow is:

- YOLO detects candidate lesion boxes
- MedSAM uses those boxes as prompts
- post-processing refines the segmentation masks

#### 4. Local Demo

The project includes:

- `api.py` as the FastAPI backend
- `ui_gradio.py` as the Gradio frontend

This makes the repository suitable for demos, course projects, and research presentations.

### Tech Stack

- Python
- PyTorch
- Ultralytics YOLO
- MedSAM / Segment Anything
- OpenCV
- FastAPI
- Gradio
- NumPy
- Pillow
- Pandas
- Matplotlib

### Project Structure

```text
1A_Cardio_Codes/
├── create_dataset/
├── pre_dataset/
├── COCO2DAtaset/
├── api.py
├── ui_gradio.py
├── Yolo_train.py
├── MedSAM_train.py
├── Yolo+MedSAM_train.py
├── Chunked_82_Dataset.py
├── generate_charts.py
├── Mask_Image.py
├── Config1.py ~ Config4.py
├── *.ipynb
└── Readme.md
```

### Quick Start

#### Requirements

- Python 3.10+
- A working PyTorch environment
- A desktop / browser environment for running the demo UI
- Prepared checkpoint files for training or inference

#### Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch ultralytics opencv-python numpy pillow fastapi uvicorn gradio pandas matplotlib pyyaml python-docx requests
```

For MedSAM-related scripts, you will also need the `segment-anything` environment and compatible checkpoints.

#### Preprocessing Workflow

```bash
python create_dataset/Video2imgs.py
python create_dataset/check.py
python create_dataset/XML2Mask.py
python create_dataset/Clean_Data.py
python create_dataset/organize_dataset.py
```

Before running them, update the hard-coded local paths inside the scripts.

#### Training

```bash
python Yolo_train.py
python MedSAM_train.py
python Yolo+MedSAM_train.py
```

Before training, make sure:

- dataset paths are updated
- class names match label IDs
- YOLO, SAM, and MedSAM checkpoint paths are correct

#### Run the Demo

Start the backend first:

```bash
uvicorn api:app --host 0.0.0.0 --port 8004
```

Then launch the frontend:

```bash
python ui_gradio.py
```

The frontend expects:

```text
http://127.0.0.1:8004/predict_json
```

### Recommended Files to Keep for GitHub

Recommended core files:

- `create_dataset/`
- `api.py`
- `ui_gradio.py`
- `Yolo_train.py`
- `MedSAM_train.py`
- `Yolo+MedSAM_train.py`
- `merge_datasets.py`
- `Chunked_82_Dataset.py`
- `generate_kfold_dataset.py`
- `generate_charts.py`
- `Mask_Image.py`
- `Config1.py` to `Config4.py`

More like drafts, logs, or generated artifacts:

- `*.ipynb`
- `__pycache__/`
- `charts_output/`
- `nul`
- sample images in the repository root
- final report `.docx`
- the `.rtf` usage note


### Known Issues

- many scripts still use hard-coded local Windows or Linux paths
- the repository mixes main scripts with experiment notebooks
- there is no unified `requirements.txt` yet
- parameters and paths are scattered across multiple scripts
- documentation and file naming still reflect a research-stage workspace

### Future Improvements

- add a proper `requirements.txt`
- add a `.gitignore`
- centralize path and parameter configuration
- clean duplicated notebooks and outdated experiment scripts
- add screenshots, pipeline diagrams, and example outputs
- rename `Readme.md` to `README.md`

