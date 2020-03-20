export KAGGLE_USERNAME=kvsnoufal
export KAGGLE_KEY=e2ac13495fadc045443b79a59b883bdc

export EPOCHS=50
export BATCH_SIZE=128
export MODEL="resnet50"

# export TRAIN_FOLDS="[1,2,3,4]"
# export TEST_FOLDS="[0]"
# python train.py

# '''loss: 0.1260 - root_loss: 0.1761 - vowel_loss: 0.0795 - \
# consonant_loss: 0.0724 - root_accuracy: 0.9507 - \
# vowel_accuracy: 0.9807 - \
# consonant_accuracy: 0.9799 - val_loss: 0.1129 - \
# val_root_loss: 0.1683 - val_vowel_loss: 0.0566 - \
# val_consonant_loss: 0.0585 - val_root_accuracy: 0.9517 - val_vowel_accuracy: 0.9857 - val_consonant_accuracy: 0.9839'

# export MODEL="resnet50_raw"
# export TRAIN_FOLDS="[0,2,3,4]"
# export TEST_FOLDS="[1]"
# python train.py
# s: 0.0526 - root_accuracy: 0.9624 - vowel_accuracy: 0.9855 - consonant_accuracy: 0.9844 - val_loss: 0.1198 - val_root_loss: 0.1815 - val_vowel_loss: 0.0556 - val_consonant_loss: 0.0607 - val_root_accuracy: 0.9473 - val_vowel_accuracy: 0.9850 - val_consonant_accuracy: 0.9827

export MODEL="mobinetv2"
export TRAIN_FOLDS="[0,2,3,4]"
export TEST_FOLDS="[1]"
python train.py
# loss: 0.1280 - root_loss: 0.1801 - vowel_loss: 0.0719 - consonant_loss: 0.0799 - root_accuracy: 0.9438 - vowel_accuracy: 0.9785 - consonant_accuracy: 0.9742 - val_loss: 0.1608 - val_root_loss: 0.2452 - val_vowel_loss: 0.0733 - val_consonant_loss: 0.0794 - val_root_accuracy: 0.9258 - val_vowel_accuracy: 0.9802 - val_consonant_accuracy: 0.9762

# export TRAIN_FOLDS="[1,0,3,4]"
# export TEST_FOLDS="[2]"
# python train.py

# export TRAIN_FOLDS="[1,2,0,4]"
# export TEST_FOLDS="[3]"
# python train.py

# export TRAIN_FOLDS="[1,2,3,0]"
# export TEST_FOLDS="[4]"
# python train.py