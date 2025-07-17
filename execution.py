import torch
from model import Gesture
import mediapipe as mp
import cv2
import torch.nn.functional as F
import win32api
import win32con
import time
import screen_brightness_control as sbc

gestures = {
    0: "Palm", #pause play
    1: "Fist", #neutral/ (or stop)
    2: "Thumbs Up", #vol up
    3: "Thumbs Down", #vol down
    4: "V Up", #brightness up
    5: "V Down", #brightness down
    6: "Thumb Left", #mute
    7: "Point Left", #prev
    8: "Thumb Right", #next
    9: "Neutral" #none
}

actions = {
    0: "Pause/Play",
    1: "Neutral",
    2: "Volume Up",
    3: "Volume Down",
    4: "Brightness Up",
    5: "Brightness down",
    6: "Mute/Unmute",
    7: "Previous Track",
    8: "Next Track",
    9: "Neutral"
}

key_mapping = { # Virtual key codes for media keys
    0: 0xB3,
    1: 0xB2,
    2: 0xAF,
    3: 0xAE,
    6: 0xAD,
    7: 0xB1,
    8: 0xB0
}

prev_key = 9 # default to neutral 
def control(int_key, prev_key=9):
    if int_key == 1: # reprogrammed to neutral
        int_key = 9 

    if prev_key == int_key:
        time.sleep(0.5)

    if int_key == 9: # neutral
        return int_key, "None"
    
    if int_key in (0,6,7,8) and int_key == prev_key: #toggle buttons shouldnt be repeatedly pressed
        pass
    
    elif int_key in (0,6,7,8):
        #pause/play, stop, mute
        win32api.keybd_event(key_mapping[int_key], 0,0,0)
        time.sleep(.05)
        win32api.keybd_event(key_mapping[int_key],0 ,win32con.KEYEVENTF_KEYUP ,0)

    elif int_key in (2,3): # volume up/down called 2 times to increase the volume
        win32api.keybd_event(key_mapping[int_key], 0,0,0)
        time.sleep(.05)
        win32api.keybd_event(key_mapping[int_key],0 ,win32con.KEYEVENTF_KEYUP ,0)

        win32api.keybd_event(key_mapping[int_key], 0,0,0)
        time.sleep(.05)
        win32api.keybd_event(key_mapping[int_key],0 ,win32con.KEYEVENTF_KEYUP ,0)

    elif int_key ==4:
        sbc.set_brightness(sbc.get_brightness()[0]+20)
    elif int_key ==5:
        sbc.set_brightness(sbc.get_brightness()[0]-20)

    elif int_key in gestures.keys:
        win32api.keybd_event(key_mapping[int_key], 0,0,0)
        time.sleep(.05)
        win32api.keybd_event(key_mapping[int_key],0 ,win32con.KEYEVENTF_KEYUP ,0)

    return int_key, actions[int_key]


model = Gesture() #load the model
model.load_state_dict(torch.load('Gesture_model3d.pth')) #load the weights

mp_drawing = mp.solutions.drawing_utils #for drawing the landmarks
mp_hands = mp.solutions.hands #for detecting the hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #capture the video from the webcam

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.3) as hands: #set the confidence values
    while (cap.isOpened()):
        ret, frame = cap.read()
        ret, frame = cap.read()
        action = "None"

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert the frame to RGB

        # image.flags.writeable = False #set the flag to false
        results = hands.process(img_rgb) #process the image
        # image.flags.writeable = True #set the flag to true

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        data_aux = []

        x_ = []
        y_ = []
        z_ = []

        #Render the detections
        if results.multi_hand_landmarks: #if the hand is detected
            for hand_landmarks in results.multi_hand_landmarks: #for each hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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
                    data_aux.append((mark.x-x_min)/x_range)
                    data_aux.append((mark.y-y_min)/y_range)
                    data_aux.append((mark.z-z_min)/z_range)
                break           

            model_inp = torch.tensor(data_aux).reshape(1,-1) # convert list to tensor
            softmax_probs = F.softmax(model(model_inp),dim=1) #get the softmax probabilities from model outputs
            predicted_class = torch.argmax(softmax_probs, dim=1) # get the class with maximum probability
            
            prev_key, action = control(predicted_class.item(), prev_key) # control the media player
            print(gestures[predicted_class.item()]) # print the gesture into terminal

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, action, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (150, 100, 70), 3,
                    cv2.LINE_AA)
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()