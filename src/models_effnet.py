import tensorflow as tf
# from tensorflow import keras
from keras.applications.resnet import ResNet50
from keras.applications.resnet_v2 import ResNet50V2
from keras.models import Model, Sequential
from keras import layers
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
# import tensorflow as tf
# from tensorflow import keras

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
        for layer in base_model.layers:
            layer.trainable = True
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

class ResnetModel50_raw:
    def __init__(self):
        self.model=self.load_model()
        

    def load_model(self):
        FACTOR = 0.70
        HEIGHT = 137
        WIDTH = 236
        HEIGHT_NEW = int(HEIGHT * FACTOR)
        WIDTH_NEW = int(WIDTH * FACTOR)
        HEIGHT_NEW = 224
        WIDTH_NEW = 224
        
        base_model=ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(HEIGHT_NEW,WIDTH_NEW,3), pooling=None, classes=1000)
        # base_model.trainable=False
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        grapheme_root = layers.Dense(168, activation = 'softmax', name = 'root')(x)
        vowel_diacritic = layers.Dense(11, activation = 'softmax', name = 'vowel')(x)
        consonant_diacritic = layers.Dense(7, activation = 'softmax', name = 'consonant')(x)

        model = Model(inputs=base_model.input,outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])
        for layer in base_model.layers:
            layer.trainable = True
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

class MobilenetV2_:
    def __init__(self):
        self.model=self.load_model()
        

    def load_model(self):
        FACTOR = 0.70
        HEIGHT = 137
        WIDTH = 236
        HEIGHT_NEW = int(HEIGHT * FACTOR)
        WIDTH_NEW = int(WIDTH * FACTOR)
        
        base_model=MobileNetV2(include_top=False, weights='imagenet',  input_shape=(HEIGHT_NEW,WIDTH_NEW,3), pooling=None, classes=1000)
        # base_model.trainable=False
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        grapheme_root = layers.Dense(168, activation = 'softmax', name = 'root')(x)
        vowel_diacritic = layers.Dense(11, activation = 'softmax', name = 'vowel')(x)
        consonant_diacritic = layers.Dense(7, activation = 'softmax', name = 'consonant')(x)

        model = Model(inputs=base_model.input,outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])
        for layer in base_model.layers:
            layer.trainable = True
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

class ResnetModel50V2:
    def __init__(self):
        self.model=self.load_model()
        

    def load_model(self):
        FACTOR = 0.70
        HEIGHT = 137
        WIDTH = 236
        HEIGHT_NEW = int(HEIGHT * FACTOR)
        WIDTH_NEW = int(WIDTH * FACTOR)
        
        base_model=ResNet50V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(HEIGHT_NEW,WIDTH_NEW,3), pooling=None, classes=1000)
        # base_model.trainable=False
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        grapheme_root = layers.Dense(168, activation = 'softmax', name = 'root')(x)
        vowel_diacritic = layers.Dense(11, activation = 'softmax', name = 'vowel')(x)
        consonant_diacritic = layers.Dense(7, activation = 'softmax', name = 'consonant')(x)

        model = Model(inputs=base_model.input,outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])
        for layer in base_model.layers:
            layer.trainable = True
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

class InceptionModelV3:
    def __init__(self):
        self.model=self.load_model()
        

    def load_model(self):
        FACTOR = 0.70
        HEIGHT = 137
        WIDTH = 236
        HEIGHT_NEW = int(HEIGHT * FACTOR)
        WIDTH_NEW = int(WIDTH * FACTOR)
        
        base_model=InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(HEIGHT_NEW,WIDTH_NEW,3), pooling=None, classes=1000)
        # base_model.trainable=False
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        grapheme_root = layers.Dense(168, activation = 'softmax', name = 'root')(x)
        vowel_diacritic = layers.Dense(11, activation = 'softmax', name = 'vowel')(x)
        consonant_diacritic = layers.Dense(7, activation = 'softmax', name = 'consonant')(x)

        model = Model(inputs=base_model.input,outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])
        for layer in base_model.layers:
            layer.trainable = True
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

class incresnetmodel:
    def __init__(self):
        self.model=self.load_model()
        

    def load_model(self):
        FACTOR = 0.70
        HEIGHT = 137
        WIDTH = 236
        HEIGHT_NEW = int(HEIGHT * FACTOR)
        WIDTH_NEW = int(WIDTH * FACTOR)
        
        base_model=InceptionResNetV2(include_top=False,  input_tensor=None, input_shape=(HEIGHT_NEW,WIDTH_NEW,3), pooling=None, classes=1000)
        # base_model.trainable=False
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        grapheme_root = layers.Dense(168, activation = 'softmax', name = 'root')(x)
        vowel_diacritic = layers.Dense(11, activation = 'softmax', name = 'vowel')(x)
        consonant_diacritic = layers.Dense(7, activation = 'softmax', name = 'consonant')(x)

        model = Model(inputs=base_model.input,outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])
        # for layer in base_model.layers:
        #     layer.trainable = True
        model.compile(optimizer='adam', loss = {'root' : 'categorical_crossentropy', 
                    'vowel' : 'categorical_crossentropy', 
                    'consonant': 'categorical_crossentropy'},
                    loss_weights = {'root' : 0.5,
                            'vowel' : 0.25,
                            'consonant': 0.25},
                    metrics={'root' : 'accuracy', 
                    'vowel' : 'accuracy', 
                    'consonant': 'accuracy'})
        print(model.summary())

        return model



import efficientnet.keras as efn
class effnetB3:
    def __init__(self):
        
        
        self.model=self.load_model()
        

    def load_model(self):
        FACTOR = 0.70
        HEIGHT = 137
        WIDTH = 236
        HEIGHT_NEW = int(HEIGHT * FACTOR)
        WIDTH_NEW = int(WIDTH * FACTOR)
        HEIGHT_NEW = 128
        WIDTH_NEW = 128

        # base_model=EfficientNetB3(include_top=False, weights='imagenet',input_shape=(HEIGHT_NEW,WIDTH_NEW,3))
        base_model=efn.EfficientNetB2(include_top=False, weights='imagenet',input_shape=(HEIGHT_NEW,WIDTH_NEW,3))
        # base_model.trainable=False
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        grapheme_root = layers.Dense(168, activation = 'softmax', name = 'root')(x)
        vowel_diacritic = layers.Dense(11, activation = 'softmax', name = 'vowel')(x)
        consonant_diacritic = layers.Dense(7, activation = 'softmax', name = 'consonant')(x)

        model = Model(inputs=base_model.input,outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])
        # for layer in base_model.layers:
        #     layer.trainable = True
        model.compile(optimizer='adam', loss = {'root' : 'categorical_crossentropy', 
                    'vowel' : 'categorical_crossentropy', 
                    'consonant': 'categorical_crossentropy'},
                    loss_weights = {'root' : 0.5,
                            'vowel' : 0.25,
                            'consonant': 0.25},
                    metrics={'root' : 'accuracy', 
                    'vowel' : 'accuracy', 
                    'consonant': 'accuracy'})
        print(model.summary())

        return model

