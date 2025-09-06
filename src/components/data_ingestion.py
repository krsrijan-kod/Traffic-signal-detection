import os
import pandas as pd
import cv2
import numpy as np
import logging




class DataIngestion:
    def __init__(self, config):
        self.dataset_path = config['data_ingestion']['dataset_path']
        self.labels_file = config['data_ingestion']['labels_file']
        self.image_size = config['data_ingestion']['image_size']


def load_data(self):
    logging.info('Starting data ingestion')
    labels_path = os.path.join(self.dataset_path, self.labels_file)
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f'Labels file not found: {labels_path}')


    df = pd.read_csv(labels_path)
    images = []
    labels = []


    for idx, row in df.iterrows():
        img_path = os.path.join(self.dataset_path, str(row['Filename']))
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError('cv2.imread returned None')
            img = cv2.resize(img, (self.image_size, self.image_size))
            images.append(img)
            labels.append(row['ClassId'])
        except Exception as e:
            logging.error(f'Failed to load {img_path}: {e}')


    images = np.array(images)
    labels = np.array(labels)


    logging.info(f'Ingested {len(images)} images')
    return images, labels