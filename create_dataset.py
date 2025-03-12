import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.3)

# Directory containing image data for different classes
DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image in the directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        
        # Load the image and convert it to RGB
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to find hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x and y coordinates for each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            # Only append if the data_aux has exactly 42 elements (21 landmarks with x, y for each)
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)

# Save the collected data and labels to a pickle file
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
