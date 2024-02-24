import numpy as np
from classifiers import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

classifier = Meso4()
classifier.load("weights/Meso4_DF.h5")

dataGenerator = ImageDataGenerator(rescale=1.0 / 255)
generator = dataGenerator.flow_from_directory(
    "test_images",
    target_size=(256, 256),  # signifies size of the image 256x256
    batch_size=1,
    class_mode="binary",
    subset="training",
)
total_images = generator.samples
print(total_images)

for _ in total_images:
    X, y = generator.next()  # this line fetches the next batch of images
    pred1 = classifier.predict(X, y)

    print(pred1)
