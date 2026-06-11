from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = load_model("model/alzheimer_model.h5")

classes = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented"
]

img_path = r"dataset\Very_Mild_Demented"
files = os.listdir(img_path)[:10]

for file in files:
    path = os.path.join(img_path, file)

    img = image.load_img(path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)

    print(file)
    print("Raw Prediction:", pred[0])
    print("Predicted:", classes[np.argmax(pred)])
    print("-" * 50)