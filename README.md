# PVI:
<a href="https://huggingface.co/datasets/AkiyasuDMK/PVI_Dataset">
  <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PVI__Dataset-blue?color=ffc107&logoColor=white&style=flat">
</a>


## Install Enviroment
Install the original enviroment here https://github.com/haotian-liu/LLaVA

## Dataset Download
LLaVA-665K dataset is available on [Download Link.](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)

VisionFlan-186k dataset is available on [https://vision-flan.github.io/#download](https://vision-flan.github.io/#download)

Then follow the original repo "LLaVA" to download the image data.

## Step1: Calculate PVI value 

```
bash ./scripts/cal_pvi.sh
```
## Step2: Implement task distribution
```
python ./scripts/task_distribution_ab.py
```
After task distribution We obtain the selected data 
