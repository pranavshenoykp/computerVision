# CS 6476 Project 6: Semantic Segmentation -- New!

To start coding locally, run:
```
jupyter notebook proj6_local.ipynb
```

To start training, run:
```
jupyter notebook proj6_colab.ipynb
```
You must use Colab for training and inference -- training will be way too slow on a CPU. It shouldn't take more than 15 minutes to train a basic model on Colab, and in 1-2 hours you can get a very solid model.

## 5 steps for Model Development

1. Start with just ResNet, without any dilation, end with 7x7
2. Now add in data augmentation
3. Now add in dilation
4. Now add in the PPM
5. Try adding in auxiliary loss



## Visualizing Data or Downloading Pre-trained Weights Locally

**You will not need to download the pretrained model locally, nor the full dataset locally, for any of the unit tests.**  However, if you'd like to play around locally with the data or pretrained model, you can run:

To download the pretrained model, run:
```bash
wget -O resnet50_v2.pth --no-check-certificate "https://drive.google.com/uc?export=download&id=1w5pRmLJXvmQQA5PtCbHhZc_uC4o0YbmA"
```
Now, move it to the following location:
```bash
mkdir proj6_code/initmodel
mv resnet50_v2.pth proj6_code/initmodel/
```

To download the dataset, run:
```bash
cd Camvid
unzip camvid_semseg11.zip
cd ..
./download_dataset.sh Camvid
```
Downloading the dataset may take about 1 minute over wget.

With an Imagenet-pretrained PSPNet-50 backbone, you can get >60% val mIoU on the Camvid dataset in 15 minutes in Colab at 240p resolution.


Expected mIoU on Val Split for:

Simple Segmentation Net Baseline, 50 Epochs | Simple Segmentation Net Baseline, 100 epochs
:-: | :-:
0.48 | 0.56


PSPNet, 50 epochs | PSPNet, 100 epochs
:-: |  :-:
0.60 | 0.65

