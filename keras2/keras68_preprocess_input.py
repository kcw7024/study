from keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np


model = ResNet50(weights='imagenet')

img_path = 'D:/study_data/_data/dog/test01.jpg'
img = image.load_img(img_path, target_size=(224, 224))
print(img) # <PIL.Image.Image image mode=RGB size=224x224 at 0x1A78E54EAC0>

x = image.img_to_array(img)
print("========================== image.img_to_array ==========================")
print(x, "\n", x.shape) # (224, 224, 3)

x = np.expand_dims(x, axis=0)
print("========================== np.expand_dims(x, axis=0) ==========================")
print(x, "\n", x.shape) # (1, 224, 224, 3)
print(np.min(x), np.max(x)) # 0.0 255.0

x = preprocess_input(x)
print("========================== preprocess_input(x) ==========================")
print(x, "\n", x.shape) # (1, 224, 224, 3)
print(np.min(x), np.max(x)) # -123.68 151.061


print("========================== model.predict(x) ==========================")
preds = model.predict(x)
print(preds, '\n', preds.shape) #  (1, 1000)

print("결과는 : ", decode_predictions(preds, top=5)[0])

'''
결과는 :  
[('n02112018', 'Pomeranian', 0.8936566), 
('n02085620', 'Chihuahua', 0.022235315), 
('n02112350', 'keeshond', 0.02153512), 
('n02113624', 'toy_poodle', 0.011202739), 
('n02085936', 'Maltese_dog', 0.008025256)]   

'''