from models import *
modelDispatcher={
    "resnet50": ResnetModel50,
    "resnet50_raw": ResnetModel50_raw,
    "mobinetv2":MobilenetV2_,
    "resnet50V2":ResnetModel50V2,
    "inceptionv3":InceptionModelV3,
    "inceptionresnet":incresnetmodel,
    "effnet":effnetB3,
    "densenet121":dnet121
    
}
