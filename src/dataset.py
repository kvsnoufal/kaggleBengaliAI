import tensorflow as tf
from tensorflow.keras.utils import Sequence,to_categorical
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import albumentations
from albumentations import *


class BengaliData(Sequence):
    def __init__(self,folds,img_height,img_width,height_scaled,width_scaled,mean,std, batch_size=64,shuffle=True):
        df=pd.read_csv("../input/train_folds.csv")
        # print(df.shape)
        df=df[["image_id","grapheme_root","vowel_diacritic","consonant_diacritic","kfold"]]
        df=df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids=df.image_id.values
        self.grapheme_root=to_categorical(df.grapheme_root.values)
        self.vowel_diacritic=to_categorical(df.vowel_diacritic.values)
        self.consonant_diacritic=to_categorical(df.consonant_diacritic.values)
        self.batch_size   = batch_size  
        self.shuffle=shuffle
        self.on_epoch_end()  

        if (len(folds)==1):
            self.aug=albumentations.Compose([
                albumentations.Resize(height_scaled,width_scaled,always_apply=True),
                albumentations.Normalize(mean,std,always_apply=True)
            ])
        else:
            self.aug=albumentations.Compose([
                albumentations.Resize(height_scaled,width_scaled,always_apply=True),
                # RandomRotate90(),
                # Flip(),
                # Transpose(),
                # OneOf([
                #     IAAAdditiveGaussianNoise(),
                #     GaussNoise(),
                # ], p=0.2),
                # OneOf([
                #     MotionBlur(p=0.2),
                #     MedianBlur(blur_limit=3, p=0.1),
                #     Blur(blur_limit=3, p=0.1),
                # ], p=0.2),
                # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                # OneOf([
                #     OpticalDistortion(p=0.3),
                #     GridDistortion(p=0.1),
                #     IAAPiecewiseAffine(p=0.3),
                # ], p=0.2),
                # OneOf([
                #     # CLAHE(clip_limit=2),
                #     IAASharpen(),
                #     IAAEmboss(),
                #     RandomBrightnessContrast(),]),
                albumentations.Cutout(num_holes=8, max_h_size=40, max_w_size=20, fill_value=0, always_apply=False, p=0.5),
                CoarseDropout(max_holes=8, max_height=40, max_width=20, p=0.5),
        
                albumentations.Normalize(mean,std,always_apply=True),
                    # HorizontalFlip(),
                

            ])

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_ids) / self.batch_size))
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    def __getitem__(self,index):
        'Generate one batch of data'
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # select data and load images
        grapheme_root = np.array([self.grapheme_root[k] for k in indexes])
        vowel_diacritic = np.array([self.vowel_diacritic[k] for k in indexes])
        consonant_diacritic = np.array([self.consonant_diacritic[k] for k in indexes])

        images = [joblib.load(f"../input/image_pickles/{self.image_ids[k]}.pkl") for k in indexes]
        images=[image.reshape(137,236).astype(float) for image in images]
        images=[Image.fromarray(image).convert('RGB') for image in images]
        images=[self.aug(image=np.array(image))["image"] for image in images]
        images=np.array(images)
        labels=[grapheme_root,vowel_diacritic,consonant_diacritic]

        return images,labels


        
# dataset=BengaliData(folds=[0,1],img_height=137,img_width=236,mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))



# print(dataset)
# for x,y in dataset:
#     print(x.shape,y.shape)
#     break


# class DataGenerator(Sequence):
#   'Generates data for Keras'
#   def __init__(self, images_paths, labels, batch_size=64, image_dimensions = (96 ,96 ,3), shuffle=False, augment=False):
#       self.labels       = labels              # array of labels
#       self.images_paths = images_paths        # array of image paths
#       self.dim          = image_dimensions    # image dimensions
#       self.batch_size   = batch_size          # batch size
#       self.shuffle      = shuffle             # shuffle bool
#       self.augment      = augment             # augment data bool
#       self.on_epoch_end()

#   def __len__(self):
#       'Denotes the number of batches per epoch'
#       return int(np.floor(len(self.images_paths) / self.batch_size))

#   def on_epoch_end(self):
#       'Updates indexes after each epoch'
#       self.indexes = np.arange(len(self.images_paths))
#       if self.shuffle:
#           np.random.shuffle(self.indexes)

#   def __getitem__(self, index):
#       'Generate one batch of data'
#       # selects indices of data for next batch
#       indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

#       # select data and load images
#       labels = np.array([self.labels[k] for k in indexes])
#       images = [cv2.imread(self.images_paths[k]) for k in indexes]
        
#       # preprocess and augment data
#       if self.augment == True:
#           images = self.augmentor(images)
        
#       images = np.array([preprocess_input(img) for img in images])
#       return images, labels
    
    
#   def augmentor(self, images):
#       'Apply data augmentation'
#       sometimes = lambda aug: iaa.Sometimes(0.5, aug)
#       seq = iaa.Sequential(
#               [
#               # apply the following augmenters to most images
#               iaa.Fliplr(0.5),  # horizontally flip 50% of all images
#               iaa.Flipud(0.2),  # vertically flip 20% of all images
#               sometimes(iaa.Affine(
#                   scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#                   # scale images to 80-120% of their size, individually per axis
#                   translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#                   # translate by -20 to +20 percent (per axis)
#                   rotate=(-10, 10),  # rotate by -45 to +45 degrees
#                   shear=(-5, 5),  # shear by -16 to +16 degrees
#                   order=[0, 1],
#                   # use nearest neighbour or bilinear interpolation (fast)
#                   cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
#                   mode=ia.ALL
#                   # use any of scikit-image's warping modes (see 2nd image from the top for examples)
#               )),
#               # execute 0 to 5 of the following (less important) augmenters per image
#               # don't execute all of them, as that would often be way too strong
#               iaa.SomeOf((0, 5),
#                          [sometimes(iaa.Superpixels(p_replace=(0, 1.0),
#                                                            n_segments=(20, 200))),
#                              # convert images into their superpixel representation
#                              iaa.OneOf([
#                                      iaa.GaussianBlur((0, 1.0)),
#                                      # blur images with a sigma between 0 and 3.0
#                                      iaa.AverageBlur(k=(3, 5)),
#                                      # blur image using local means with kernel sizes between 2 and 7
#                                      iaa.MedianBlur(k=(3, 5)),
#                                      # blur image using local medians with kernel sizes between 2 and 7
#                              ]),
#                              iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
#                              # sharpen images
#                              iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
#                              # emboss images
#                              # search either for all edges or for directed edges,
#                              # blend the result with the original image using a blobby mask
#                              iaa.SimplexNoiseAlpha(iaa.OneOf([
#                                      iaa.EdgeDetect(alpha=(0.5, 1.0)),
#                                      iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),
#                                                             direction=(0.0, 1.0)),
#                              ])),
#                              iaa.AdditiveGaussianNoise(loc=0,
#                                                        scale=(0.0, 0.01 * 255),
#                                                        per_channel=0.5),
#                              # add gaussian noise to images
#                              iaa.OneOf([
#                                      iaa.Dropout((0.01, 0.05), per_channel=0.5),
#                                      # randomly remove up to 10% of the pixels
#                                      iaa.CoarseDropout((0.01, 0.03),
#                                                        size_percent=(0.01, 0.02),
#                                                        per_channel=0.2),
#                              ]),
#                              iaa.Invert(0.01, per_channel=True),
#                              # invert color channels
#                              iaa.Add((-2, 2), per_channel=0.5),
#                              # change brightness of images (by -10 to 10 of original value)
#                              iaa.AddToHueAndSaturation((-1, 1)),
#                              # change hue and saturation
#                              # either change the brightness of the whole image (sometimes
#                              # per channel) or change the brightness of subareas
#                              iaa.OneOf([
#                                      iaa.Multiply((0.9, 1.1), per_channel=0.5),
#                                      iaa.FrequencyNoiseAlpha(
#                                              exponent=(-1, 0),
#                                              first=iaa.Multiply((0.9, 1.1),
#                                                                 per_channel=True),
#                                              second=iaa.ContrastNormalization(
#                                                      (0.9, 1.1))
#                                      )
#                              ]),
#                              sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),
#                                                                  sigma=0.25)),
#                              # move pixels locally around (with random strengths)
#                              sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
#                              # sometimes move parts of the image around
#                              sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
#                          ],
#                          random_order=True
#                          )
#               ],
#               random_order=True
#       )
#       return seq.augment_images(images)
