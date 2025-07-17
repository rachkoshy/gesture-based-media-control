import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []
        z_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks: #if hand detected
            for hand_landmarks in results.multi_hand_landmarks: # for each detected hand
                for mark in (hand_landmarks.landmark): 
                    x_.append(mark.x)
                    y_.append(mark.y)
                    z_.append(mark.z)

                x_min = min(x_)
                y_min = min(y_)
                z_min = min(x_)
                x_range = max(x_) - x_min
                y_range = max(y_) - y_min
                z_range = max(x_) - z_min


                for mark in hand_landmarks.landmark:
                    data_aux.append((mark.x-x_min)/x_range) #normalize the data
                    data_aux.append((mark.y-y_min)/y_range)
                    data_aux.append((mark.z-z_min)/z_range)
                break


            data.append(data_aux) #for each image if hand detected
            labels.append(int(dir_))

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f) #to save in dictionary
f.close()

print(np.array(data).shape, data)
print(np.array(labels).shape, labels)