import asyncio
from argparse import ArgumentParser
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import os
import json

img_path = 'data/VOCdevkit/VOC2007/JPEGImages/' # path saves images to be tested
img_list = os.listdir(img_path)
img_list = sorted(img_list, key=lambda x: int(os.path.splitext(x)[0]))
file_len = len(img_list)
imgnames = []
for i in range(file_len):
  filename = img_list[i][:-4]+ '.png'
  imgnames.append(filename)
# print(imgnames[:3])

config = 'configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py'
checkpoint = 'work_dirs/1109_2/latest.pth'
device = 'cuda:0'

# load model
model = init_detector(config, checkpoint, device)

# for i in range (len(imgnames)):

data = []

for x in range(int(len(imgnames)/2)):
  print(x*2)
  result = inference_detector(model, img_path+imgnames[x*2]) #return an array list
  for i in range (10): 
    if(len(result[i])!=0):
        # print(result[i])
        for j in range(len(result[i])):
          # print('class:',i)
          # print('score:'+str(result[i][j][4]))
          # print('bbox :',result[i][j][:4])
          if(result[i][j][4]>0.3):
            score = float(result[i][j][4])
            bbox=[float(result[i][j][0]),float(result[i][j][1]),float(result[i][j][2]-result[i][j][0]),float(result[i][j][3]-result[i][j][1])]
            image_id=int(imgnames[x*2][:-4])
            category_id=int(i)
            k = {}
            k=({
                'image_id': image_id,
                'score': score,
                'category_id': category_id,
                'bbox': bbox
            })
            data.append(k)
  # show_result_pyplot(model,img_path+imgnames[x], result,score_thr=0.3)

with open('answer.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)


# category_id = i (10->0)
# score = result[i][4]
# bbox = [result[i][0],result[i][1],result[i][2]-result[i][0],result[i][3]-result[i][1]]