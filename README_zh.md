# EVA

[English](README.md) | 中文

“EVA: Editing for Versatile Alignment against Jailbreaks” 的官方代码仓库。已被 **IEEE TPAMI 2026** 接收 🎉🎉。

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2605.14750)
[![IEEE TPAMI](https://img.shields.io/badge/IEEE%20TPAMI-paper-00629B.svg?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11523146)
[![Hugging Face Collection](https://img.shields.io/badge/🤗%20Hugging%20Face-EVA%20Collection-FFD21E)](https://huggingface.co/collections/wanglne/eva-editing-for-versatile-alignment-against-jailbreaks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 方法概述

本工作提出了 EVA，一种面向多种越狱攻击的模型编辑对齐框架。EVA 旨在提升大语言模型（LLM）和视觉语言模型（VLM）在不同越狱场景下的鲁棒性与安全对齐能力，同时保持模型在正常任务上的通用能力。

<p align="center">
<img src="figs/method.png" width="90%">
</p>


## 最新消息

- 🎉 **EVA: Editing for Versatile Alignment against Jailbreaks** 已被 **IEEE TPAMI 2026** 接收。

## 安装

需要 Python 3.9 或更高版本。克隆仓库并创建 Conda 环境：

```bash
git clone https://github.com/wanglne/EVA.git
cd EVA

conda create -n EVA python=3.9 -y
conda activate EVA

python -m pip install --upgrade pip
python -m pip install ".[gpu]"
```

## 模型来源

如果未指定 `--model-path`，EVA 会下载对应模型配置所指定版本的 Hugging Face 模型；也可以通过 `--model-path` 指定本地 checkpoint。

| 模型配置 | Hugging Face 仓库 |
| --- | --- |
| `llava-1.5-7b` | [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf) |
| `qwen2.5-vl-7b-instruct` | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| `internvl3.5-8b` | [`OpenGVLab/InternVL3_5-8B`](https://huggingface.co/OpenGVLab/InternVL3_5-8B) |


## 数据

安装后的 EVA 包已包含实验所用的 200 条 JSON 记录和固定的视觉 token 索引。数据来自 [HarmBench](https://github.com/centerforaisafety/HarmBench) 标准行为数据集。配套的 EVA 图片可从以下位置下载：

- [EVA 图片数据（Google Drive）](https://drive.google.com/drive/folders/137l_6BwyXD72OuQ4XE4SHnutW9y6h7Gq?usp=drive_link)

请根据所用模型下载对应的压缩包：

| 模型 | 压缩包 | JSON 中的逻辑路径前缀 |
| --- | --- | --- |
| LLaVA | `llava_images.zip` | `images/1.jpg` ... `images/200.jpg` |
| Qwen2.5-VL | `qwen_images.zip` | `qwen_images/1.jpg` ... `qwen_images/200.jpg` |
| InternVL3.5 | `internvl_images.zip` | `internvl_images/1.jpg` ... `internvl_images/200.jpg` |

解压后，将 `--image-root` 指向直接包含 `1.jpg` 至 `200.jpg` 的图片目录。也可以传入保留了上表相对路径结构的上级目录。

例如，`--image-root` 既可以指向包含 `qwen_images/1.jpg` 的上级目录，也可以直接指向包含 `1.jpg` 的图片目录。开始编辑前，可使用以下命令检查全部数据：

```bash
python -m eva inspect \
  --model qwen2.5-vl-7b-instruct \
  --image-root /path/to/qwen_images
```

## 协方差矩阵（C）

图像编辑和文本编辑会在每个目标语言模型层上共用同一份协方差矩阵。建议直接下载预计算矩阵，并将对应模型的压缩包解压到 `cache/covariances`。

预计算矩阵可从以下位置下载：

- [EVA 协方差矩阵（Google Drive）](https://drive.google.com/open?id=1vQWdPu0WRCAmV_Fy-LyZdpojURCq7Lrd)

提供的压缩包如下：

| 模型 | 压缩包 | 大小 | 解压后包含的层 |
| --- | --- | ---: | --- |
| LLaVA | `eva-covariance-llava-1.5-7b.zip` | 925 MB | 7, 8 |
| Qwen2.5-VL | `eva-covariance-qwen2.5-vl-7b-instruct.zip` | 2.7 GB | 7, 8 |
| InternVL3.5 | `eva-covariance-internvl3.5-8b.zip` | 1.2 GB | 8, 9 |

下载对应模型的压缩包后，按以下方式解压：

```bash
mkdir -p cache/covariances
unzip eva-covariance-qwen2.5-vl-7b-instruct.zip -d cache/covariances
```

上述预计算矩阵基于 `wikipedia/20200501.en` 快照生成。

如果缺少某个层的矩阵，EVA 会在编辑开始前自动补算。自动计算使用以下固定的数据版本：

```text
repo:      wikimedia/wikipedia
config:    20231101.en
split:     train
revision:  b04c8d1ceb2f5cd4588862100d08de323dccfbaa
```
计算进度会保存在 `.partial.pt` checkpoint 中；任务中断后，再次运行时会自动继续。

如需提前计算当前模型配置指定的两个层，可运行：

```bash
python -m eva covariance compute \
  --model qwen2.5-vl-7b-instruct \
  --covariance-root cache/covariances \
  --device cuda:0
```

如需复用已有的旧版统计结果目录，可运行：

```bash
python -m eva covariance discover \
  --model qwen2.5-vl-7b-instruct \
  --covariance-root cache/covariances \
  --legacy-root /path/to/legacy/data/stats \
  --write-manifest
```

如果匹配到多个候选文件，代码不会自动选择。此时可使用 `covariance import --layer N --legacy-path FILE` 明确指定某一层对应的文件。


## 一条命令完成图像和文本编辑

解压图片和对应的 C 压缩包后，执行：

```bash
scripts/eva-edit.sh \
  --model qwen2.5-vl-7b-instruct \
  --image-root /path/to/qwen_images \
  --covariance-root cache/covariances \
  --output-dir outputs/qwen-eva \
  --device cuda:0
```

如果对应版本的模型尚未缓存，该命令会自动下载。可以添加 `--model-path /path/to/checkpoint`，改用本地 checkpoint。如果没有提前解压 C，代码会先计算并保存缺失的矩阵，再开始图像编辑。添加 `--no-auto-covariance` 可关闭自动计算，并在缺少 C 时直接报错。

EVA 默认使用以下实验参数：

| 模型配置 | 编辑层 | 图像 `v_lr` / KL | 文本 `v_lr` / KL |
| --- | --- | --- | --- |
| `llava-1.5-7b` | 7, 8 | 0.0055 / 0.1 | 0.5 / 0.0625 |
| `qwen2.5-vl-7b-instruct` | 7, 8 | 0.5 / 0.0625 | 0.5 / 0.0625 |
| `internvl3.5-8b` | 8, 9 | 0.5 / 0.0625 | 0.5 / 0.0625 |


输出目录结构如下：

```text
output-dir/
  image_checkpoint/       完成图像编辑后的完整 VLM、processor 和 tokenizer
    .eva_checkpoint.json
  final_checkpoint/       完成图像和文本编辑后的完整 VLM、processor 和 tokenizer
    .eva_checkpoint.json
  run_manifest.json       各阶段的执行记录
```

代码会先将 checkpoint 写入同级临时目录，确认保存完整后再通过原子重命名移至目标目录。如果图像阶段的 checkpoint 已保存，但文本编辑失败，可在相同命令后添加 `--resume`；代码会跳过图像编辑，并从经过验证的中间 checkpoint 重新开始文本编辑。

如果模型权重、数据、超参数、图片根目录、记录数量限制或协方差输入发生变化，`--resume` 会拒绝继续。`--force-restart` 只会删除当前输出目录中的 `run_manifest.json`，以及名为 `image_checkpoint` 和 `final_checkpoint` 的两个目录。如需只处理一条记录进行快速功能测试，可使用 `--max-records 1`。


## 配置

EVA 使用以下目录中的文件作为模型配置、超参数和内置数据的唯一来源：

```text
src/eva/resources/configs/models/
src/eva/resources/configs/hparams/image/
src/eva/resources/configs/hparams/text/
src/eva/resources/data/
```

代码会根据模型对象解析实际加载的模块名称，可识别 `model.language_model.layers.*`、`model.layers.*` 和 `language_model.model.layers.*` 等已知结构。图像和文本两个阶段实际使用的模块路径都会写入 `run_manifest.json`。

## 欢迎大家引用

```bibtex
@ARTICLE{11523146,
  author={Wang, Yi and Qiu, Hongye and Xu, Yue and Yang, Sibei and Qin, Zhan and Huang, Minlie and Wang, Wenjie},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={EVA: Editing for Versatile Alignment against Jailbreaks},
  year={2026},
  volume={},
  number={},
  pages={1-16},
  keywords={Automatic speech recognition;Modeling;Safety;Visualization;Large language models;Conferences;Optimization;Educational institutions;Light emitting diodes;Tuning;Safety Alignment;Jailbreak Attacks;Model Editing;Large Language Models;Vision Language Models},
  doi={10.1109/TPAMI.2026.3694189}}
```
