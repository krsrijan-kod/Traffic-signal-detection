from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.callbacks import ModelCheckpoint,EarlyStopping
import logging 
import os

class ModelTrainer:
    def __init__(self,config):
        self.model_path=config['training']['model_path']
        self.epochs=config['training']['epochs']
        self.batch_size=config['training']['batch_size']

    def buil_model(self,num_classes,img_size):
        logging.info('Building model')
        model=Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        return model
    
    def train(self,model,X_train,y_train,X_val,y_val):
        logging.info('Starting training')
        create_dir=os.path.dirname(self.model_path)
        if create_dir and not os.path.exists(create_dir):
            os.makedirs(create_dir,exist_ok=True)

        checkpoint=ModelCheckpoint(self.model_path,monitor='val_accuracy',save_best_only=True,verbose=1)
        early_stop=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True,verbose=1)

        history=model.fit(
            X_train,y_train,
            validation_data=(X_val,y_val),
            batch_Size=self.batch_size,
            epochs=self.epochs,
            callbacks=[checkpoint,early_stop]        
        )

        logging.info('Training completed')
        return history