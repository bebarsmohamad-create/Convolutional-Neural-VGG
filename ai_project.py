
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

train_gen = ImageDataGenerator(rescale=1./255)
val_gen   = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
   "bebarsmohamad-create/Convolutional-Neural-VGG/test",  
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    "/content/drive/MyDrive/dataset/val", 
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=3
)

img = image.load_img("/content/drive/MyDrive/dataset/train/01_palm/frame_00_01_0002.png", target_size=(224, 224)) 
img = image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
print(pred)
