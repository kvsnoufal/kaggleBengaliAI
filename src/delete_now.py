# import tensorflow as tf
# from tensorflow import keras as K
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers
# from tensorflow.keras.models import Model

# def error_metric(y_true,y_pred):
#     y_true=tf.cast(y_true, dtype='float32')
#     y_pred=tf.cast(y_pred, dtype='float32')

#     return tf.reduce_sum(tf.math.subtract(y_true ,y_pred))


# def error2(y_true,y_pred):
#     return np.sum(y_true-y_pred)

# y_true = np.array([1,2,3])

# y_pred = np.array([1,2,5])
# print(y_true)
# print(y_pred)
# print(error_metric(y_true, y_pred))
# print(error2(y_true, y_pred))

# model=Sequential([layers.Dense(1,input_shape=(1,)),layers.Dense(1)])
# model.compile(loss=error2)
# model.summary()

# model.fit(np.array([1,2,3,4,5,6]),np.array([1,1,1,0,0,0]),epochs=1)


# InputShape=(676,875,3)

# model = Sequential()
# model.add(layers.Conv2D(32,(3,3),padding='same',input_shape=InputShape))
# model.add(layers.MaxPooling2D((5,5), padding='same'))
# model.add(layers.Conv2D(16,(5,5), activation='relu', padding='same'));
# model.add(layers.MaxPooling2D((3,3), padding='same'));
# model.add(layers.Conv2D(16,(5,5), activation='relu', padding='same'));
# model.add(layers.MaxPooling2D((3,3), padding='same'));
# model.add(layers.Conv2D(16,(5,5), activation='relu', padding='same'));
# model.add(layers.MaxPooling2D((2,2), padding='same'));
# model.add(layers.Conv2D(16,(5,5), activation='relu', padding='same'))
# model.add(layers.Flatten())
# model.add(layers.Dense(16, activation='relu'));
# model.add(layers.Dropout(0.5));
# model.add(layers.Dense(128, activation='linear'));
# print(model.summary())


# model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# from tensorflow.keras.layers import *
# inputs=[]
# outputs=[]

# inp1=layers.Input(shape=(64,1),name='real_input')
# out1=layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inp1)
# out1=layers.Conv1D(filters=64, kernel_size=3, activation='relu')(out1)
# out1=layers.Dropout(0.2)(out1)
# out1=layers.MaxPooling1D(pool_size=2)(out1)
# out1=layers.Flatten()(out1)
# inputs.append(inp1)
# outputs.append(out1)

# inp2=layers.Input(shape=(64,1),name='imaginary_input')
# out2=layers.Conv1D(filters=12, kernel_size=3, activation='relu')(inp1)
# out2=layers.Conv1D(filters=6, kernel_size=3, activation='relu')(out2)
# out2=layers.Dropout(0.2)(out2)
# out2=layers.MaxPooling1D(pool_size=2)(out2)
# out2=layers.Flatten()(out2)
# inputs.append(inp2)
# outputs.append(out2)

# x = layers.Concatenate()(outputs)
# x = BatchNormalization()(x)
# x = Dropout(0.3)(x)
# x = Dense(128, activation="relu")(x)
# x = Dropout(0.3)(x)
# x = BatchNormalization()(x)
# x = Dense(32, activation="relu")(x)
# x = Dropout(0.3)(x)

# real_output = layers.Dense(16, activation = 'linear', name = 'real_output')(x)
# imaginary_output = layers.Dense(16, activation = 'linear', name = 'imaginary_output')(x)

# model=Model(inputs=inputs,outputs =[ real_output,imaginary_output])
# print(model.summary())
# model.compile(optimizer='adam',loss={"real_output":'mse','imaginary_output':"mse"},loss_weights={"real_output":0.5,"imaginary_output":0.5})


# real_X=np array of shape 100000,64,1
# imag_X= np array of shape 100000,64,1

# X_train=[real_x,imag_x]

# y_train=[real_y,imag_y]
# model.fit(X-train,ytrain,epcohs=100, eval_data=(X_val,y_val))




import pymsteams
myTeamsMessage=pymsteams.connectorcard("https://outlook.office.com/webhook/ea049498-10df-479d-8d38-b2e08be6ddcb@eee3385e-742f-4e2e-b130-e496ed7d6a49/IncomingWebhook/72392d123f1841d48cdaf5cbc6dfad4b/46285699-313e-41a1-8916-a23e741851f5")
myTeamsMessage.text("test")
myTeamsMessage.title("title")
myTeamsMessage.send()