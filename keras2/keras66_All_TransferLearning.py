from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetLarge, NASNetMobile
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from keras.applications import Xception


all_model = [VGG16(), VGG19(), 
             ResNet50(), ResNet50V2(), 
             ResNet101(), ResNet101V2(), ResNet152(), ResNet152V2(),
             DenseNet121(), DenseNet169(), DenseNet201(),
             InceptionV3(), InceptionResNetV2(), MobileNet(), 
             MobileNetV2(), MobileNetV3Small(), MobileNetV3Large(),
             NASNetLarge(), NASNetMobile(), EfficientNetB0(), 
             EfficientNetB1(), EfficientNetB7(), Xception()]


for model in all_model:
    model = model
    # model.trainable = False
    print(f"모델명 :: {model.name}")
    print(f"전체 가중치 갯수 :: {len(model.weights)}")
    print(f"훈련 가능 가중치 갯수 :: {len(model.trainable_weights)}\n")

'''

모델명 :: vgg16
전체 가중치 갯수 :: 32      
훈련 가능 가중치 갯수 :: 32 

모델명 :: vgg19
전체 가중치 갯수 :: 38      
훈련 가능 가중치 갯수 :: 38 

모델명 :: resnet50
전체 가중치 갯수 :: 320     
훈련 가능 가중치 갯수 :: 214

모델명 :: resnet50v2        
전체 가중치 갯수 :: 272     
훈련 가능 가중치 갯수 :: 174

모델명 :: resnet101
전체 가중치 갯수 :: 626     
훈련 가능 가중치 갯수 :: 418

모델명 :: resnet101v2       
전체 가중치 갯수 :: 544     
훈련 가능 가중치 갯수 :: 344

모델명 :: resnet152
전체 가중치 갯수 :: 932
훈련 가능 가중치 갯수 :: 622

모델명 :: resnet152v2
전체 가중치 갯수 :: 816
훈련 가능 가중치 갯수 :: 514

모델명 :: densenet121
전체 가중치 갯수 :: 606
훈련 가능 가중치 갯수 :: 364

모델명 :: densenet169
전체 가중치 갯수 :: 846
훈련 가능 가중치 갯수 :: 508

모델명 :: densenet201
전체 가중치 갯수 :: 1006
훈련 가능 가중치 갯수 :: 604

모델명 :: inception_v3
전체 가중치 갯수 :: 378
훈련 가능 가중치 갯수 :: 190

모델명 :: inception_resnet_v2
전체 가중치 갯수 :: 898
훈련 가능 가중치 갯수 :: 490

모델명 :: mobilenet_1.00_224
전체 가중치 갯수 :: 137
훈련 가능 가중치 갯수 :: 83

모델명 :: mobilenetv2_1.00_224
전체 가중치 갯수 :: 262
훈련 가능 가중치 갯수 :: 158

모델명 :: MobilenetV3small
전체 가중치 갯수 :: 210
훈련 가능 가중치 갯수 :: 142

모델명 :: MobilenetV3large
전체 가중치 갯수 :: 266
훈련 가능 가중치 갯수 :: 174

모델명 :: NASNet
전체 가중치 갯수 :: 1546
훈련 가능 가중치 갯수 :: 1018

모델명 :: NASNet
전체 가중치 갯수 :: 1126
훈련 가능 가중치 갯수 :: 742

모델명 :: efficientnetb0
전체 가중치 갯수 :: 314
훈련 가능 가중치 갯수 :: 213

모델명 :: efficientnetb1
전체 가중치 갯수 :: 442
훈련 가능 가중치 갯수 :: 301

모델명 :: efficientnetb7
전체 가중치 갯수 :: 1040
훈련 가능 가중치 갯수 :: 711

모델명 :: xception
전체 가중치 갯수 :: 236
훈련 가능 가중치 갯수 :: 156


'''


