import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import  Flatten, Activation, Convolution2D,Input,Conv2DTranspose,GlobalAveragePooling2D,UpSampling2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D,Dense,Multiply,Concatenate,ReLU,BatchNormalization,Lambda,Reshape
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.models import Model
import os
import random
from tensorflow.python.ops import array_ops
import sys
from Preprocessing import pre_processing,myMaskRCNNConfig,get_rcnn
from sklearn.utils import class_weight
import math
import keras.backend as K
import json
import time
from tensorflow import keras

def create_model():
    inputs=Input(shape=(224, 224, 3))
    layerc1in=BatchNormalization()(inputs)
    mulayerc1=Convolution2D(filters=1,kernel_size=(19,19),strides=(2, 2),activation='linear',padding='same')(layerc1in)
    sigmalayerc1=Convolution2D(filters=1,kernel_size=(19,19),strides=(2, 2),activation='linear',padding='same')(layerc1in)
    layerc1 = Lambda(sample_z)([mulayerc1, sigmalayerc1])
    layerc1=Reshape((112,112,1))(layerc1)
    layerc1=Convolution2D(filters=32,kernel_size=(19,19),strides=(1, 1),activation='relu',padding='same')(layerc1)
    layer1= Conv2DTranspose(16, (19,19), strides=(2, 2),padding='same',activation='relu') (layerc1)
    #layer1=UpSampling2D((2,2))(layerc1)
    outputs1=Convolution2D(filters=8,kernel_size=(19,19),strides=(1, 1),activation='relu',padding='same')(layer1)
    outputs=Convolution2D(filters=2,kernel_size=(19,19),strides=(1, 1),activation='sigmoid',padding='same')(outputs1)
    model=Model([inputs],[outputs])
    return model

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(1,tf.shape(mu)[1],tf.shape(mu)[2],tf.shape(mu)[3]), mean=0, stddev=1)
    return mu + K.exp(log_sigma / 2) * eps

kmodel = keras.models.load_model('./preweight/multiclass_weight.h5')

dic = {0 : 'CER', 
       1 : 'DER13', 
       2 : 'DER579', 
       3 : 'GER', 
       4 : 'Healthy', 
       5 : 'Other'}

def DRWAP_rate(im_path,seg_model,rate_model,print_DRWAP=True):
    ratio=np.array([[0,0],[1,0.1],[3,0.25],[5,0.5],[7,0.75],[9,1]])
    img=cv2.imread(im_path)
    images,prediction,ims_crop,n_cluster,sortrois=pre_processing(im_path,seg_model)  #error !!
    print("type ->>>>")
    print(type(images))
    print("type ->>>>")
    print(type(prediction))
    print("type ->>>>")
    print(type(ims_crop))
    print("type ->>>>")
    print(type(n_cluster))
    print("type ->>>>")
    print(type(sortrois))

    rates=[]
    for image in ims_crop:
        im2ar = np.asarray(image)
        im2ar = cv2.resize(im2ar, (224,224))
        im2ar = im2ar/255.
        im2ar = im2ar.reshape((1, im2ar.shape[0], im2ar.shape[1], im2ar.shape[2]))
        if "DER" not in (dic[np.argmax(kmodel.predict(im2ar),axis=1)[0]]):
            rates.append(dic[np.argmax(kmodel.predict(im2ar),axis=1)[0]])
        else:
            rate,heighest=rating_by_ear(image,rate_model,print_rate=False)
            rates.append(rate)
    rates=np.array(rates)
    
    if(len(rates)==0) :
        DRWAP=None
    else :
        (unique, counts) = np.unique(rates, return_counts=True)

        print("<---- list of uniqe ---->")
        print(unique)
        print(len(unique))
        print(unique.dtype.type)
        if(np.str_==unique.dtype.type) :
            if not (unique[len(unique)-1].isnumeric()) :
                unique = np.delete(unique, len(unique)-1)
                counts = np.delete(counts, len(unique)-1)
            print(len(unique))
            if not (unique[len(unique)-1].isnumeric()) :
                unique = np.delete(unique, len(unique)-1)
                counts = np.delete(counts, len(unique)-1)
            print(len(unique))

    
        # test = np.concatenate((unique,counts),0)
        # print(test)
        # # for data in unique :
        # #     print(counts[joe])
        # #     print(np.argwhere(unique==data))
        # #     print(data.isnumeric())
        # #     if not (data.isnumeric()): ##/////////////// exception Healthy
        # #         unique = np.delete(unique, np.argwhere(unique==data))
        # #         counts = np.delete(counts, joe)
        # #     joe+=1
        # # for isnum in range(len(unique)) :
        # #     print(isnum)
        # for isnum in range(len(unique)):
        #     print("isnum = {i} : {k}".format(i=isnum,k=unique[isnum]))
        #     if(len(unique)-1==isnum) :
        #         print(unique[isnum])
        #     if not (unique[isnum].isnumeric()): ##/////////////// exception Healthy
        #         unique = np.delete(unique, isnum)
        #         counts = np.delete(counts, isnum)
        unique = unique.astype(int)
        print(unique)
        print(counts)
        ratio=np.array([x[1] for x in ratio if x[0] in unique ])
        DRWAP=np.sum(counts.T*ratio)/rates.shape[0]*100

    if print_DRWAP:
        print('rate each an ear',rates,'DRWAP of this group',str(DRWAP)+'%')
    return rates,DRWAP,sortrois

def rating_by_ear(img,model,print_rate=True):
    h_img,w_img=img.shape[0:2]
    resize_shape=model.layers[0].get_output_at(0).get_shape().as_list()[1:3]
    img=cv2.resize(img/255.,(resize_shape[0],resize_shape[1]))
    pre=model.predict(img.reshape(1,resize_shape[0],resize_shape[1],3))
    pre=np.argmax(pre[0],axis=2).astype('float32')
    pre[np.sum(img,axis=2)==0]=0
    pre=cv2.resize(pre,(w_img,h_img))
    kernel = np.ones((149,149),np.float32)/(149**2) #Kernelfilter
    dst = cv2.filter2D(pre,-1,kernel)
    dst[dst<0.5]=0
    percentder=np.mean(dst,axis=1)
    percentder[percentder<0.5]=0
    heighest=np.where(percentder>=0.5)[0]
    if heighest.shape[0]!=0:
        heighest=(np.min(heighest)+1)/percentder.shape[0]*100

        heighest=100-heighest
    else:
        heighest=0
    
    if heighest<=10:
        heighest+=1
        rate=1
    elif heighest <= 25:
        rate =3
    elif heighest <= 50:
        rate=5
    elif heighest <=75:
        rate=7
    elif heighest <= 100:
        rate=9
        
    if print_rate: #ไม่แสดงเปลี่ยน print_rate==False
        print(rate,heighest)
    
    return rate,heighest

def rbg (impath):
    #Rate by group
    model = create_model()
    model.load_weights('./preweight/rating_weight-0042.h5')
    model_dir=''
    rcnn_weight='./preweight/rcnn_weight.h5'
    config = myMaskRCNNConfig()
    #impath=r"D:\My Drive\AI\Datasets\corn_disease\corn_group\3.jpg"
    filename, file_extension = os.path.splitext(os.path.basename(impath))
    rates_r,DRWAP_r,sortrois_r=DRWAP_rate(impath,get_rcnn(model_dir,rcnn_weight,config),model)
    return rates_r,DRWAP_r,sortrois_r,impath,filename

def export_json(impath):
    rates_r,DRWAP_r,sortrois_r,impath,filename = rbg(impath)
    # Data to be written
    js ={
        "extID": "",
        "plotID": "",
        "trialID": "",
        "barcode": "",
        "cornGroupID": "",
        "earDetail": [],
        "predicted": []}
    js["cornGroupID"]=os.path.split(impath)[1].split(".")[0]
    if(type(sortrois_r) != int) :
        print("!!! this is if !!!!!!!!!!!")
        for i in range(sortrois_r.shape[0]):
            js["earDetail"].append({"ID":str(i+1),"position":{"xlt":str(sortrois_r[i,0]),"ylt":str(sortrois_r[i,1]),"xrb":str(sortrois_r[i,2]),"yrb":str(sortrois_r[i,3])},"rate":str(rates_r[i])})
        js["predicted"].append({"DRWAP":str(DRWAP_r)})
        dercount = 0
        for j in range(len(rates_r)):
            if rates_r[j].isnumeric():
                dercount+=1
        DLERP_r=dercount/len(rates_r)*100
        js["predicted"].append({"DLERP":str(DLERP_r)})
        js["predicted"].append({"#totalEar":str(len(rates_r))})
        js["predicted"].append({"#DERear":str(dercount)})

    # Serializing json
    json_object = json.dumps(js, indent = 4)

    filename, file_extension = os.path.splitext(os.path.basename(impath))

    # Writing to sample.json
    with open('./json/'+filename+".json", "w") as outfile:
        outfile.write(json_object)

export_json(r"C:\\Corn\\CORNGROUPS\\115.jpg")
#export_json(sys.argv[1])
## 86, 87, 89, 115, 154, 250, 254, 259, 278, 287, 330 