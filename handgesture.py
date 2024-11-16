import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ASL Alphabet Mapping (simplified for demo)
# This can be expanded with more gesture definitions
ASL_ALPHABET = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V'
}

# Function to detect ASL based on hand landmarks
def recognize_asl(landmarks):
    # Map landmarks to fingers
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Logic for recognizing a few ASL letters based on relative positions
    if index_tip[1] < thumb_tip[1] and middle_tip[1] < index_tip[1] and ring_tip[1] < middle_tip[1] and pinky_tip[1] < ring_tip[1]:
        return "A"  # Example for gesture 'A'
    elif index_tip[1] > thumb_tip[1] and middle_tip[1] > index_tip[1] and ring_tip[1] > middle_tip[1] and pinky_tip[1] > ring_tip[1]:
        return "B"  # Example for gesture 'B'
    elif index_tip[1] < thumb_tip[1] and middle_tip[1] > index_tip[1] and ring_tip[1] > middle_tip[1]:
        return "C"  # Example for gesture 'C'

    # Additional gestures for other letters can be added here based on hand positions
    return "Unknown Gesture"


# OpenCV video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the image horizontally for better display
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB (MediaPipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    # If hand landmarks are found
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmark coordinates (x, y) relative to the image size
            hand_landmarks = [(lm.x, lm.y) for lm in landmarks.landmark]

            # Recognize the ASL gesture based on the hand landmarks
            asl_gesture = recognize_asl(hand_landmarks)
            
            # Display the recognized ASL letter on the frame
            cv2.putText(frame, f'Gesture: {asl_gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the output frame
    cv2.imshow("Sign Language Recognition", frame)
    
    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
