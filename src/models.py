import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers


class ResnetModel50:
    def __init__(self):
        self.model=self.load_model()
        

    def load_model(self):
        
        base_model=ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(137,236,3), pooling=None, classes=1000)
        # base_model.trainable=False
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
        model.compile(optimizer='adam', loss = {'root' : 'categorical_crossentropy', 
                    'vowel' : 'categorical_crossentropy', 
                    'consonant': 'categorical_crossentropy'},
                    loss_weights = {'root' : 0.5,
                            'vowel' : 0.25,
                            'consonant': 0.25},
                    metrics={'root' : 'accuracy', 
                    'vowel' : 'accuracy', 
                    'consonant': 'accuracy'})
        # print(model.summary())

        return model
