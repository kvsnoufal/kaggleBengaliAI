
from models import ResnetModel
from dataset import BengaliData
from tensorflow.keras.callbacks import *

EPOCHS=1

model=ResnetModel()

model=model.model
print(model.summary())
train_gen=BengaliData(folds=[0,1,2,3],img_height=137,img_width=236,mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))
valid_gen=BengaliData(folds=[4],img_height=137,img_width=236,mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))
reduceLR = ReduceLROnPlateau(monitor = 'val_root_loss',
                             patience = 2,
                             factor = 0.1,
                             min_lr = 1e-8,
                             verbose = 1)
# Callback : Save best model
chkPoint = ModelCheckpoint('../log/train_1.h5',
                           monitor = 'val_root_loss',
                           save_best_only = True,
                           save_weights_only = False,
                           mode = 'auto',
                           period = 1,
                           verbose = 0)
# Callback : Early Stop
earlyStop = EarlyStopping(monitor='val_root_loss',
                          mode = 'auto',
                          patience = 4,
                          min_delta = 0,
                          verbose = 2)
CALLBACKS = [reduceLR, chkPoint, earlyStop]


model.fit_generator(train_gen,epochs=EPOCHS,callbacks=CALLBACKS,validation_data=valid_gen) 