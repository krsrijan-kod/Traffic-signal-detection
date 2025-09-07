from keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
import logging
import numpy as np

class Datapreprocessing:
    def __init__(self,image_size,test_size=0.2,random_state=42):
        self.image_size=image_size
        self.test_size=test_size
        self.random_state=random_state

    def preprocess(self,images,labels):
        logging.info('starting preprocessing')

        images=images.astype('float32')/255.0

        unique_labels=sorted(list(set(labels)))
        labels_to_index={label: idx for idx,label in enumerate(unique_labels)}
        indexed_labels=np.array([labels_to_index[l] for l in labels])

        num_classes=len(unique_labels)
        y=to_categorical(indexed_labels,num_classes)

        X_train,X_test,y_train,y_test=train_test_split(images,y,test_size=self.test_size,random_state=self.random_state)

        logging.info(f'preprocessing completed. Train size:{len(X_train)},test size:{len(X_test)}')
        return X_train, X_test, y_train, y_test, num_classes, labels_to_index