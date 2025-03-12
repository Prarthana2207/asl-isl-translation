import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26  # 26 alphabets
dataset_size = 100  # collect 100 images per alphabet

cap = cv2.VideoCapture(0)  # adjust the webcam device index as needed

for j in range(number_of_classes):
    alphabet = chr(65 + j)  # convert index to alphabet (A-Z)
    if not os.path.exists(os.path.join(DATA_DIR, alphabet)):
        os.makedirs(os.path.join(DATA_DIR, alphabet))

    print(f'Collecting data for alphabet {alphabet}')

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, alphabet, f'{counter}.jpg'), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
