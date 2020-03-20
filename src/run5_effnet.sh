export KAGGLE_USERNAME=kvsnoufal
export KAGGLE_KEY=e2ac13495fadc045443b79a59b883bdc

export EPOCHS=500
export BATCH_SIZE=128

export MODEL="densenet121"


# export TRAIN_FOLDS="[1,2,3,4]"
# export TEST_FOLDS="[0]"
# python train.py

# #  root_accuracy: 0.9773 - vowel_accuracy: 0.9929 - consonant_accuracy: 0.99192
# export TRAIN_FOLDS="[0,2,3,4]"
# export TEST_FOLDS="[1]"
# python train.py
# # python train_withmixup.py


# export TRAIN_FOLDS="[1,0,3,4]"
# export TEST_FOLDS="[2]"
# python train.py


export TRAIN_FOLDS="[1,2,0,4]"
export TEST_FOLDS="[3]"
python train.py


# export TRAIN_FOLDS="[1,2,3,0]"
# export TEST_FOLDS="[4]"
# python train.py