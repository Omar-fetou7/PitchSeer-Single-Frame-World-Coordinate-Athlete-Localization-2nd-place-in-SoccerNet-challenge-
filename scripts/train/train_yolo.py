from ultralytics import YOLO
import os

# Load a model
#you have to configure your data the same as in ulrealytics docs for pose estimation

model = YOLO("yolo26m-pose")
model.train(data='data.yaml',epochs=100,batch=8,imgsz=1280,save_period=25,
            cos_lr=True,lr0=0.0001,lrf=0.01,warmup_epochs=5,mosaic=1,close_mosaic=10,degrees=5,shear=1,fliplr=0,optimizer='SGD',device=0,workers=8)