# EVA

[English](README.md) | 中文

这是论文 “EVA: Editing for Versatile Alignment against Jailbreaks” 的官方代码仓库。该论文已被 **IEEE TPAMI 2026** 接收 🎉🎉。

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2605.14750)
[![IEEE TPAMI](https://img.shields.io/badge/IEEE%20TPAMI-paper-00629B.svg?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/11523146)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 方法概述

本工作提出了 EVA，一种通过模型编辑实现通用越狱防御对齐的框架。EVA 旨在提升大语言模型（LLM）和视觉语言模型（VLM）面对不同越狱场景时的鲁棒性与安全对齐能力，同时维持模型在正常任务上的通用能力。

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

如果没有指定 `--model-path`，EVA 会下载对应配置中固定 revision 的 Hugging Face 模型快照。也可以通过 `--model-path` 显式指定本地 checkpoint。

| 配置名称 | Hugging Face 仓库 |
| --- | --- |
| `llava-1.5-7b` | [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf) |
| `qwen2.5-vl-7b-instruct` | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| `internvl3.5-8b` | [`OpenGVLab/InternVL3_5-8B`](https://huggingface.co/OpenGVLab/InternVL3_5-8B) |


## 数据

安装后的 EVA 包已经包含准确选取的 200 条 JSON 记录和固定的视觉 token 索引。这些行为记录来源于 [HarmBench](https://github.com/centerforaisafety/HarmBench) 标准行为数据集。请从以下位置下载对应的 EVA 图片：

- [EVA 图片数据（Google Drive）](https://drive.google.com/drive/folders/137l_6BwyXD72OuQ4XE4SHnutW9y6h7Gq?usp=drive_link)

每个模型配置下载一个对应的压缩包：

| 模型 | 压缩包 | JSON 中的逻辑路径前缀 |
| --- | --- | --- |
| LLaVA | `llava_images.zip` | `images/1.jpg` ... `images/200.jpg` |
| Qwen2.5-VL | `qwen_images.zip` | `qwen_images/1.jpg` ... `qwen_images/200.jpg` |
| InternVL3.5 | `internvl_images.zip` | `internvl_images/1.jpg` ... `internvl_images/200.jpg` |

解压选定的压缩包，找到直接包含 `1.jpg` 至 `200.jpg` 的目录，并通过 `--image-root` 传入该目录。EVA 也接受保留上表逻辑路径前缀的父目录。

通常，`--image-root` 既可以指向包含 `qwen_images/1.jpg` 这类路径的父目录，也可以直接指向包含 `1.jpg` 的图片目录。在进行编辑前，可以检查全部数据记录：

```bash
python -m eva inspect \
  --model qwen2.5-vl-7b-instruct \
  --image-root /path/to/qwen_images
```

## 协方差矩阵（C）

对于每个需要编辑的语言模型层，图像编辑和文本编辑阶段共享同一个协方差矩阵。推荐的复现方式是下载论文实验使用的预计算矩阵，并将对应模型的压缩包直接解压到 `cache/covariances`。

从以下位置下载发布的压缩包：

- [EVA 协方差矩阵（Google Drive）](https://drive.google.com/open?id=1vQWdPu0WRCAmV_Fy-LyZdpojURCq7Lrd)

已经准备好的发布文件如下：

| 模型 | 压缩包 | 大小 | 解压后包含的层 |
| --- | --- | ---: | --- |
| LLaVA | `eva-covariance-llava-1.5-7b.zip` | 925 MB | 7, 8 |
| Qwen2.5-VL | `eva-covariance-qwen2.5-vl-7b-instruct.zip` | 2.7 GB | 7, 8 |
| InternVL3.5 | `eva-covariance-internvl3.5-8b.zip` | 1.2 GB | 8, 9 |

下载对应模型的压缩包后，按下面的方式解压：

```bash
mkdir -p cache/covariances
unzip eva-covariance-qwen2.5-vl-7b-instruct.zip -d cache/covariances
```

这里提供的论文实验矩阵使用的是 `wikipedia/20200501.en` 快照计算。

如果缺少所需矩阵，EVA 可以在编辑开始前自行计算。使用当前可获取的 revision 的数据集：

```text
repo:      wikimedia/wikipedia
config:    20231101.en
split:     train
revision:  b04c8d1ceb2f5cd4588862100d08de323dccfbaa
```
计算过程中会使用 `.partial.pt` checkpoint 记录当前文档位置，中断后可以自动继续。

使用固定的模型和数据集计算配置中的两个层：

```bash
python -m eva covariance compute \
  --model qwen2.5-vl-7b-instruct \
  --covariance-root cache/covariances \
  --device cuda:0
```

如果希望注册已有的旧版统计目录，可以执行：

```bash
python -m eva covariance discover \
  --model qwen2.5-vl-7b-instruct \
  --covariance-root cache/covariances \
  --legacy-root /path/to/legacy/data/stats \
  --write-manifest
```

如果发现多个可能匹配的文件，自动发现过程会拒绝选择。当某一层需要显式指定时，请使用 `covariance import --layer N --legacy-path FILE`。


## 一条命令完成图像编辑和文本编辑

解压图片和推荐的 C 压缩包后，执行：

```bash
scripts/eva-edit.sh \
  --model qwen2.5-vl-7b-instruct \
  --image-root /path/to/qwen_images \
  --covariance-root cache/covariances \
  --output-dir outputs/qwen-eva \
  --device cuda:0
```

如果固定 revision 的模型尚未缓存，该命令会自动下载模型。可以添加 `--model-path /path/to/checkpoint`，改用指定的本地 Hugging Face checkpoint。如果没有解压 C，同一条命令会先计算并保存缺少的矩阵，然后再开始图像编辑。添加 `--no-auto-covariance` 可以禁止自动计算，并在缺少 C 时直接报错。

恢复出的实验参数如下：

| 配置名称 | 编辑层 | 图像 `v_lr` / KL | 文本 `v_lr` / KL |
| --- | --- | --- | --- |
| `llava-1.5-7b` | 7, 8 | 0.0055 / 0.1 | 0.5 / 0.0625 |
| `qwen2.5-vl-7b-instruct` | 7, 8 | 0.5 / 0.0625 | 0.5 / 0.0625 |
| `internvl3.5-8b` | 8, 9 | 0.5 / 0.0625 | 0.5 / 0.0625 |


输出目录结构如下：

```text
output-dir/
  image_checkpoint/       完整的父 VLM、processor 和 tokenizer
    .eva_checkpoint.json
  final_checkpoint/       完整的最终父 VLM、processor 和 tokenizer
    .eva_checkpoint.json
  run_manifest.json       不可变输入指纹和各阶段来源信息
```

checkpoint 会先写入同级临时目录，验证完整后再通过原子重命名正式写入目标位置。如果已经完成图像 checkpoint，但文本阶段运行失败，可以使用相同命令并添加 `--resume`；程序会跳过图像编辑，并从经过验证的中间 checkpoint 重新开始文本编辑。

如果模型权重、数据、超参数、图片根目录、记录数量限制或协方差输入发生变化，`--resume` 会拒绝继续。`--force-restart` 只会删除当前输出目录中本次运行的 manifest，以及两个具有固定名称的 checkpoint 目录。需要进行小规模功能测试时，可以使用 `--max-records 1`。


## 配置

安装包中的配置与数据来源文件位于：

```text
src/eva/resources/configs/models/
src/eva/resources/configs/hparams/image/
src/eva/resources/configs/hparams/text/
src/eva/resources/data/
```

程序会根据模型对象解析实际加载的模块名称，覆盖已知的 `model.language_model.layers.*`、`model.layers.*` 和 `language_model.model.layers.*` 等结构。图像和文本两个阶段最终选择的实际模块路径都会写入 `run_manifest.json`。

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
