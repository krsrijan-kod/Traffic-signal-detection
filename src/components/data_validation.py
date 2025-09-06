import logging
import numpy as np 

def validate_images(images,labels):

    logging.info('validating images and labels')
    if len(images)!=len(labels):
        raise ValueError('No images found')
    
    if len(images)==0:
        raise ValueError('No images found')
    
    ##checking shape 
    for i,img in enumerate(images):
        if img is None:
            raise ValueError(f'Image at index{i} is none')
        if len(img.shape)!=3 or img.shape[2]!=3:
            raise ValueError(f'Image at index{i} doen not have 3 channels: shape{img.shape}')
        if np.isnan(img).any():
            raise ValueError(f'Image at index{i} contains NaN values')
        
    logging.info('Validation passed')
    return True