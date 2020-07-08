from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Create your views here.
from keras.models import load_model
from keras.preprocessing import image
import json
import tensorflow as tf
from tensorflow import Graph
from tensorflow import Session

import numpy as np
# import cv2


img_height,img_width=218,178
with open('./models/imagenet_class_index.json','r') as f:
    labelInfo=f.read()

labelInfo=json.loads(labelInfo)

model_graph=Graph()
with model_graph.as_default():
    tf_session = tf.Session()
    with tf_session.as_default():
        model=load_model('./models/weights.best.inc.male.h5')


def index(request):
    return render(request,'index.html')

def predictgender(request):
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    print(filePathName)
    testimage='.'+filePathName
    img=image.load_img(testimage,target_size=(img_height,img_width))
    x=image.img_to_array(img)
    x=x/255
    x=x.reshape(1,img_height,img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)
    predictedLabel=labelInfo[str(np.argmax(predi[0]))]
    context={'filePathName':filePathName,'predictedLabel':predictedLabel[1]}
    return render(request,'output.html',context)

# def predictgender(request):
#     fileObj=request.FILES['filePath']
#     fs=FileSystemStorage()
#     filePathName=fs.save(fileObj.name,fileObj)
#     filePathName=fs.url(filePathName)
#     im = cv2.imread(filePathName)
#     im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (178, 218)).astype(np.float32) / 255.0
#     im = np.expand_dims(im, axis =0)
#
#         # # prediction
#     with model_graph.as_default():
#         with tf_session.as_default():
#             result = model.predict(im)
#     prediction = labelInfo[str(np.argmax(result))]
#     context={'filePathName':filePathName,'prediction':prediction}
#     return render(request,'index.html',context)
