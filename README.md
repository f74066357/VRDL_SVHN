# Street View House Numbers detection

## Dataset:
SVHN dataset  
https://drive.google.com/drive/folders/1aRWnNvirWHXXXpPPfcWlHQuzGJdXagoc?usp=sharing
 - 33402 training images
 - 13068 testing images 

## Dataset Structure
```
data
└── VOCdevkit
    └── VOC2007
        ├── Annotations
        ├── JPEGImages
        └── ImageSets
            └── Main
                ├── test.txt
                ├── train.txt
                └── val.txt
```

## Environment
```
conda create -n open-mmlab python=3.7 -y 
conda activate open-mmlab
conda install pytorch=1.3 cudatoolkit=10.1 torchvision -c pytorch
pip install mmcv-full
```
-- clone this repository -- 
```
cd VRDL_SVHN
pip install -r requirements/build.txt
pip install -v -e .
```
## Train 
```
python tools/train.py ./configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py
```

## Inference
```
python demo/mytest.py
```
