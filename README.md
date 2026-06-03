# EVA

This is the official repository for "EVA: Editing for Versatile Alignment against Jailbreaks" **(Accepted by IEEE TPAMI 2026)** 🎉🎉.

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2605.14750)
[![IEEE TPAMI](https://img.shields.io/badge/IEEE%20TPAMI-paper-00629B.svg?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/11523146)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

In this work, we propose EVA, an editing-based framework for versatile alignment against jailbreak attacks. EVA aims to improve the robustness and safety alignment of LLMs and VLMs under diverse jailbreak scenarios, while maintaining their general capabilities on benign tasks.

<p align="center">
<img src="figs/method.png" width="90%">
</p>


## News

- 🎉 **EVA: Editing for Versatile Alignment against Jailbreaks** has been accepted by **IEEE TPAMI 2026**.
- 🚧 The code for the **VLM part** is currently being organized and will be released soon.
- 🚧 For the **LLM part**, please first refer to our previous repository: [DELMAN](https://github.com/wanglne/DELMAN).

## Code Release Status

### LLM Part

The LLM-related implementation of EVA is closely related to our previous work **DELMAN**. Before the EVA code is fully released, you may refer to the DELMAN repository for the LLM model editing pipeline:

- [DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing](https://github.com/wanglne/DELMAN)

### VLM Part

The VLM-related code is currently being organized and will be released soon.

## Data

The images used by the three HarmBench image data files in `data/` (`HarmBench_images_llava.json`, `HarmBench_images_qwen.json`, and `HarmBench_images_internvl.json`) are available in this Google Drive folder:

- [Image data](https://drive.google.com/drive/folders/137l_6BwyXD72OuQ4XE4SHnutW9y6h7Gq?usp=drive_link)

## TODO

- [x] Release LLM-related code
- [x] Release the EVA paper link
- [ ] Release VLM-related code

## Citation
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