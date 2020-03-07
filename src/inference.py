import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
import pandas as pd
import os
import pickle
import joblib
import numpy as np
import albumentations
from PIL import Image
from glob import glob

class ResnetModel50:
    def __init__(self):
        self.model=self.load_model()
        

    def load_model(self):
        
        base_model=ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(137,236,3), pooling=None, classes=1000)
        base_model.trainable=False
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        grapheme_root = layers.Dense(168, activation = 'softmax', name = 'root')(x)
        vowel_diacritic = layers.Dense(11, activation = 'softmax', name = 'vowel')(x)
        consonant_diacritic = layers.Dense(7, activation = 'softmax', name = 'consonant')(x)

        model = Model(inputs=base_model.input,outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])
        # for layer in base_model.layers:
        #     layer.trainable = False
        # model.compile(optimizer='adam', loss = {'root' : 'categorical_crossentropy', 
        #             'vowel' : 'categorical_crossentropy', 
        #             'consonant': 'categorical_crossentropy'},
        #             loss_weights = {'root' : 0.333,
        #                     'vowel' : 0.333,
        #                     'consonant': 0.333},
        #             metrics={'root' : 'accuracy', 
        #             'vowel' : 'accuracy', 
        #             'consonant': 'accuracy'})
        # print(model.summary())
        
        return model



class TestDataGenerator(keras.utils.Sequence):
    def __init__(self,X, batch_size=64,shuffle=False):
        self.X=X
        # print(df.shape)
        
        self.batch_size   = batch_size  
        self.shuffle=shuffle
        self.on_epoch_end()  

        

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        
    def __getitem__(self,index):
        'Generate one batch of data'
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        images=np.array([self.X[k] for k in indexes])
    

        return images

model1=ResnetModel50().model
model2=ResnetModel50().model
model3=ResnetModel50().model
model4=ResnetModel50().model
model5=ResnetModel50().model

model1.load_weights('../log/train_0.h5')
model2.load_weights('../log/train_1.h5')
model3.load_weights('../log/train_2.h5')
model4.load_weights('../log/train_3.h5')
model5.load_weights('../log/train_4.h5')


def preprocess(image):
    aug=albumentations.Compose([
                albumentations.Resize(137,236,always_apply=True),
                albumentations.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225),always_apply=True)
            ])
    image=image.reshape(137,236).astype(float)
    image=Image.fromarray(image).convert('RGB')
    return aug(image=np.array(image))["image"]

# Create Submission File
tgt_cols = ['grapheme_root','vowel_diacritic','consonant_diacritic']

# Create Predictions
row_ids, targets = [], []

for f in glob("../input/test_*.parquet"):
    df=pd.read_parquet(f)
    imageids=df.image_id.values
    df=df.drop('image_id',axis=1)
    image_arrays=df.values
    
    for j, img_ids in tqdm(enumerate(imageids),total=len(imageids)):



# Loop through Test Parquet files (X)
for i in range(0, 4):
    # Test Files Placeholder
    test_files = []

    # Read Parquet file
    df = pd.read_parquet(os.path.join('../input', 'test_image_data_'+str(i)+'.parquet'))
    # Get Image Id values
    image_ids = df['image_id'].values 
    # Drop Image_id column
    df = df.drop(['image_id'], axis = 1)

    # Loop over rows in Dataframe and generate images 
    X = []
    for image_id, index in zip(image_ids, range(df.shape[0])):
        test_files.append(image_id)
        X.append(preprocess(df.loc[df.index[index]].values))

    # Data_Generator
    X=np.array(X)
    print(X.shape)
    test_gen=TestDataGenerator(X)
    # Predict with all 3 models
    preds1 = model1.predict_generator(test_gen, verbose = 1)
    preds2 = model2.predict_generator(test_gen, verbose = 1)
    preds3 = model3.predict_generator(test_gen, verbose = 1)
    preds4 = model4.predict_generator(test_gen, verbose = 1)
    preds5 = model5.predict_generator(test_gen, verbose = 1)
    
    # Loop over Preds    
    for i, image_id in zip(range(len(test_files)), test_files):
        
        for subi, col in zip(range(len(preds1)), tgt_cols):
            sub_preds1 = preds1[subi]
            sub_preds2 = preds2[subi]
            sub_preds3 = preds3[subi]
            sub_preds4 = preds4[subi]
            sub_preds5 = preds5[subi]

            # Set Prediction with average of 5 predictions
            row_ids.append(str(image_id)+'_'+col)
            sub_pred_value = np.argmax((sub_preds1[i] + sub_preds2[i] + sub_preds3[i] + sub_preds4[i] + sub_preds5[i]) / 5)
            targets.append(sub_pred_value)
    
    # Cleanup
    del df
submit_df = pd.DataFrame({'row_id':row_ids,'target':targets}, columns = ['row_id','target'])
submit_df.to_csv('../output/submission.csv', index = False)
print(submit_df.head(40))