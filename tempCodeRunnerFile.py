# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 
               11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 
               20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

previous_character = ''
start_time = None
image_display_time = 2  # Time in seconds to wait for confirmation
current_thumbnail = None  # To store the current character's thumbnail

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video frame.")
        break

    H, W, _ = frame.shape

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to find hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            # Extract x and y coordinates for each landmark
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)

        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        # Ensure data_aux has the correct size (42 features)
        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)

            # Check if the character has been consistent for 2 seconds
            if previous_character == predicted_character:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time > image_display_time:
                    # Load the image of the predicted character
                    char_image_path = f'./isl_images/{predicted_character}.png'  # Path to stored character images
                    print(f"Loading image for {predicted_character}: {char_image_path}")
                    current_thumbnail = cv2.imread(char_image_path)
                    
                    if current_thumbnail is not None:
                        current_thumbnail = cv2.resize(current_thumbnail, (100, 100))
                        print(f"Confirmed Character: {predicted_character}")
                    else:
                        print(f"Image not found for {predicted_character}")
                    start_time = None  # Reset timer
            else:
                # New prediction detected
                print(f"New prediction detected: {predicted_character}")
                previous_character = predicted_character
                start_time = time.time()

                # Update the thumbnail image immediately
                char_image_path = f'./isl_images/{predicted_character}.png'  # Path to stored character images
                current_thumbnail = cv2.imread(char_image_path)
                if current_thumbnail is not None:
                    current_thumbnail = cv2.resize(current_thumbnail, (100, 100))
                else:
                    print(f"Image not found for {predicted_character}")

    # Overlay the thumbnail image if available
    if current_thumbnail is not None:
        x_offset, y_offset = 10, 10  # Position of the thumbnail (top-left corner)
        y_end = y_offset + current_thumbnail.shape[0]
        x_end = x_offset + current_thumbnail.shape[1]

        # Overlay the thumbnail image onto the frame
        frame[y_offset:y_end, x_offset:x_end] = current_thumbnail

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()