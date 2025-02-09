import cv2
from gtts import gTTS
from playsound import playsound
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf


with tf.device('/cpu:0'):
    model = load_model('trained_model.keras', compile=False)
    model.summary()


def speech(text):
    print(text)
    language = "en"
    output = gTTS(text=text, lang=language, slow=False)

    output.save("./sounds/output.mp3")
    playsound("./sounds/output.mp3")


camera = cv2.VideoCapture(0)
labels = ['DONOT CROSS', 'NOTHING DETECTED', 'TRAFFIC SIGNAL DETECTED', 'YOU CAN CROSS', 'ZEBRA CROSSING DETECTED']

while True:
    with tf.device('/cpu:0'):
        ret, image = camera.read()
        if image is not None:
            image_resized=cv2.resize(image, (128,128))
        else:
            continue
        
        image_array = np.array(image_resized).astype('float32') / 255.0
         
        image_batch = np.expand_dims(image_array, axis=0)
        
        print("Shape of image_batch:", image_batch.shape)
    
        probabilities = model.predict(image_batch)
        # print("Predicted probabilities:", probabilities)
        for predictions in (probabilities[0]):
            if predictions>=0.1:
                print(labels[np.argmax(probabilities)])
                  
        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)
        # 27 is the ASCII for the esc key on your keyboard.
        if keyboard_input == 27:
            break

camera.release()
cv2.destroyAllWindows()


i = 0
new_sentence = []
for label in labels:
    if i == 0:
        new_sentence.append(label)
    else:
        new_sentence.append(label)

    i += 1

speech(" ".join(new_sentence))


