import ast
import os
from dataset import BengaliData
from tensorflow.keras.callbacks import *
# from keras.callbacks import *
# from keras.models import load_model
import pickle
from modelDispatcher import modelDispatcher
from sklearn.metrics import recall_score
import numpy as np
from mixupAug import MixupImageDataGenerator


EPOCHS=int(os.environ.get("EPOCHS"))
TRAIN_FOLDS=ast.literal_eval(os.environ.get("TRAIN_FOLDS"))
TEST_FOLDS=ast.literal_eval(os.environ.get("TEST_FOLDS"))
BATCH_SIZE=int(os.environ.get("BATCH_SIZE"))

MODEL=os.environ.get("MODEL")

print(MODEL,BATCH_SIZE)
print(EPOCHS,TRAIN_FOLDS,TEST_FOLDS)
FACTOR = 0.70
HEIGHT = 137
WIDTH = 236
HEIGHT_NEW = int(HEIGHT * FACTOR)
WIDTH_NEW = int(WIDTH * FACTOR)
HEIGHT_NEW = int(HEIGHT * FACTOR)
WIDTH_NEW = int(WIDTH * FACTOR)
HEIGHT_NEW = 224
WIDTH_NEW = 224

def macro_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')

class CustomCallback(Callback):
    def __init__(self, val_data, batch_size = BATCH_SIZE):
        super().__init__()
        self.valid_data = val_data
        self.batch_size = batch_size
    
    def on_epoch_begin(self,epoch, logs={}):
        self.recall_scores = []
        self.avg_recall = []
        
    def on_epoch_end(self, epoch, logs={}):
        batches = len(self.valid_data)
        total = batches * self.batch_size
        self.val_recalls = {0: [], 1:[], 2:[]}
        
        for batch in range(batches):
            xVal, yVal = self.valid_data.__getitem__(batch)
            val_preds = self.model.predict(xVal)
            
            for i in range(3):
                preds = np.argmax(val_preds[i], axis=1)
                true = np.argmax(yVal[i], axis=1)
                self.val_recalls[i].append(macro_recall(true, preds))
        
        for i in range(3):
            self.recall_scores.append(np.average(self.val_recalls[i]))

        avg_result = np.average(self.recall_scores, weights=[2, 1, 1])
        self.avg_recall.append(avg_result)    

        if avg_result == max(self.avg_recall):
            print("Avg. Recall Improved. Saving model.")
            print(f"Avg. Recall: {round(avg_result, 4)}")
            self.model.save_weights(f'../log/best_avg_recall_{MODEL}_{TEST_FOLDS[0]}.h5')
        return


if __name__ == "__main__":
    model=modelDispatcher[MODEL]().model
    # model.load_weights(f'../log/train_{MODEL}_{TEST_FOLDS[0]}.h5')
    

    train_gen=BengaliData(folds=TRAIN_FOLDS,img_height=137,img_width=236,height_scaled=HEIGHT_NEW,width_scaled=WIDTH_NEW,mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225),batch_size=BATCH_SIZE)
    valid_gen=BengaliData(folds=TEST_FOLDS,img_height=137,img_width=236,height_scaled=HEIGHT_NEW,width_scaled=WIDTH_NEW,mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225),batch_size=BATCH_SIZE)
    train_generator = MixupImageDataGenerator(train_gen,batch_size=BATCH_SIZE)
    validation_generator = MixupImageDataGenerator(valid_gen,batch_size=BATCH_SIZE)

    reduceLR = ReduceLROnPlateau(monitor = 'val_root_loss',
                             patience = 2,
                             factor = 0.1,
                             min_lr = 1e-16,
                             verbose = 1)
    # Callback : Save best model
    chkPoint = ModelCheckpoint(f'../log/train_{MODEL}_{TEST_FOLDS[0]}.h5',
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
    csv_logger = CSVLogger(f'../log/csvLog_{MODEL}_{TEST_FOLDS[0]}.csv')

    custom_callback = CustomCallback(valid_gen)
    CALLBACKS = [reduceLR, chkPoint, earlyStop,csv_logger]

    train_history = model.fit_generator(train_generator,epochs=EPOCHS,\
                                        steps_per_epoch=BATCH_SIZE,\
                                        callbacks=CALLBACKS,validation_data=validation_generator)
    model.save(f"../log/modelsave_{MODEL}_{TEST_FOLDS[0]}.p")
    with open(f'../log/trainHistoryDict_{MODEL}_{TEST_FOLDS[0]}.p', 'wb') as file_pi:
        pickle.dump(train_history.history, file_pi)
    