export KAGGLE_USERNAME=kvsnoufal
export KAGGLE_KEY=e2ac13495fadc045443b79a59b883bdc

export EPOCHS=50
export BATCH_SIZE=128

export MODEL="resnet50V2"


export TRAIN_FOLDS="[1,2,3,4]"
export TEST_FOLDS="[0]"
python train.py

export TRAIN_FOLDS="[0,2,3,4]"
export TEST_FOLDS="[1]"
python train.py


export TRAIN_FOLDS="[1,0,3,4]"
export TEST_FOLDS="[2]"
python train.py


export TRAIN_FOLDS="[1,2,0,4]"
export TEST_FOLDS="[3]"
python train.py


export TRAIN_FOLDS="[1,2,3,0]"
export TEST_FOLDS="[4]"
python train.py