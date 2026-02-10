# PVI:


<p align="center">
  <a href="">
    <img alt="Static Badge" src="https://img.shields.io/badge/Home-PVI-blue?style=flat&link=https%3A%2F%2Fprincetonvisualai.github.io%2FPVI%2F">
  </a>
  <a href="https://huggingface.co/datasets/AkiyasuDMK/PVI_Dataset">
  <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PVI__Dataset-blue?color=ffc107&logoColor=white&style=flat">
</a>
</p>
</p>
<hr>



## Install Enviroment
Install the original enviroment here https://github.com/haotian-liu/LLaVA

## Dataset Download
LLaVA-665K dataset is available on [Download Link.](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)

VisionFlan-186k dataset is available on [https://vision-flan.github.io/#download](https://vision-flan.github.io/#download)

Then follow the original repo "LLaVA" to download the image data.
# 🚀 Pipeline
## Step1: Calculate PVI value 

```
bash ./scripts/cal_pvi.sh
```
## Step2: Task Distribution
Optimize the distribution of tasks based on calculated PVI values to obtain the final subset.
```
python ./scripts/task_distribution_ab.py
```
After task distribution We obtain the selected data for fine-tuning

# Model Fine-tuning
Ensure you have adjusted the data_path in the script to your PVI-selected data
```
bash ./LLaVA/scripts/v1_5/finetune_lora.sh
```

## Evaluation
Please follow the [original LLaVA page](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#evaluation) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate models.
# Citation
@article{pvi2024,
  title={Your Paper Title},
  author={Your Name and others},
  journal={arXiv},
  year={2024}
}

