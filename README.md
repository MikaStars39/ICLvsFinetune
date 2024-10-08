# Deeper Insights Without Updates: The Power of In-Context Learning Over Fine-Tuning
This repository contains the code for this paper:
> **Deeper Insights Without Updates: The Power of In-Context Learning Over Fine-Tuning**\
> Qingyu Yin  Xuzheng He  Luoao Deng  Chak Tou Leong Fan Wang  Yanzhao Yan  Xiaoyu Shen  Qiang Zhang\
> Paper: [https://arxiv.org/abs/2410.04691v1](https://arxiv.org/abs/2410.04691v1)
## 🔗 Quick Links
- [Deeper Insights Without Updates: The Power of In-Context Learning Over Fine-Tuning](#deeper-insights-without-updates:-the-power-of-in-context-learning-over-finetuning)
  - [TODO List](#todo-list)
  - [Abstract](#-abstract)
  - [Install Requirements](#install-requirements)
  - [Training scripts](#training-scripts)
  - [Evaluation](#evaluation)
  - [Citation](#citation)
## TODO List
- [x] Release the training scripts.
- [x] Release the evaluation scripts.
- [ ] Upload the dataset.
## Abstract
Fine-tuning and in-context learning (ICL) are two prevalent methods in imbuing large language models with task-specific knowledge. It is commonly believed that fine-tuning can surpass ICL given sufficient training samples as it allows the model to adjust its internal parameters based on the data. However, this paper presents a counterintuitive finding: For tasks with implicit patterns, ICL captures these patterns significantly better than fine-tuning. We developed several datasets featuring implicit patterns, such as sequences determining answers through parity or identifying reducible terms in calculations. We then evaluated the models’ understanding of these patterns under both fine-tuning and ICL across models ranging from 0.5B to 7B parameters. The results indicate that models employing ICL can quickly grasp deep patterns and significantly improve accuracy. In contrast, fine-tuning, despite utilizing thousands of times more training samples than ICL, achieved only limited improvements. We also proposed circuit shift theory from a mechanistic interpretability’s view to explain why ICL wins.
## Install Requirements
The following steps will guide you through the installation process.

First, create a Python virtual environment using e.g. Conda:
```shell
conda create -n iclvsft python=3.10
conda activate iclvsft
```

Next, install the package dependencies as follows:

```shell
pip install -r requirements.txt
```

You will also need Flash Attention 2 installed (This is not neccesary, only for training acceleration), which can be done by running:

```shell
python -m pip install flash-attn --no-build-isolation
```
## Training scripts
```shell
python finetune.py
```
## Evaluation
```shell
python test.py
```
## Citation 
Please cite our paper if you find the repo helpful in your work:

```bibtex
@misc{yin2024deeperinsightsupdatespower,
      title={Deeper Insights Without Updates: The Power of In-Context Learning Over Fine-Tuning}, 
      author={Qingyu Yin and Xuzheng He and Luoao Deng and Chak Tou Leong and Fan Wang and Yanzhao Yan and Xiaoyu Shen and Qiang Zhang},
      year={2024},
      eprint={2410.04691},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.04691}, 
}
```
