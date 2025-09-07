import os
import cv2
import numpy as np
from keras.models import load_model
import logging

def load_trained_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'model not found:{model_path}')
    
    model=load_model(model_path)
    logging.info(f'Model loaded from {model_path}')
    return model

def predict_image(model,img_path,img_size,index_to_label,class_names=None):
    img=cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Image not found:{img_path}')
    img=cv2.resize(img,(img_size,img_size))
    img=img.astype('float32')/255.0
    img=np.expand_dims(img,axis=0)

    pred=model.predict(img)
    class_idx=int(np.argmax(pred))
    confidence=float(np.max(pred))*100.0

    label=index_to_label.get(class_idx,str(class_idx))
    class_name=None
    if class_names and class_idx<len(class_name):
        class_name=class_name[class_idx]

    return {
        'label_index':class_idx,
        'label':label,
        'class_name':class_name,
        'confidence':confidence
    }