from ultralytics import YOLO
import cv2
import torch
import keyboard
import math

model = YOLO("Pose_Estimation_YOLO/yolo11n-pose.pt")
cap = cv2.VideoCapture(0)  # For webcam

# Check for a CUDA-enabled GPU and set the device accordingly.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Variables for gesture stability
# We'll track the gesture for a few frames to prevent flickering.
pressed_key = None
last_action = None
gesture_frame_count = 0
min_frames_to_hold = 3  # Number of consecutive frames to confirm a gesture
default_angle = 170  # change as per your default angle
tilt_threshold = 20  # variable for tilt threshold
head_movement_threshold = 4  # variable for head movement threshold
# variable to store the neutral y position of the head
neutral_y_position_head = None

# Main loop for video capture and processing
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally to correct for mirrored camera feeds
    frame = cv2.flip(frame, 1)

    # Use YOLOv8 to predict poses on the specified device.
    results = model(frame, show=False, conf=0.7, device=device)

    # Variables for current gesture state
    current_action = None
    current_action_text = "Action: No gesture detected"

    # Check if a person was detected
    if results[0].keypoints.shape[1] > 0:
        keypoints = results[0].keypoints.xy.cpu().numpy()[0]

        # --- Detect Head Tilt ---
        # Keypoints: nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4)
        if len(keypoints) > 4:  # Ensure we have keypoints for eyes and ears
            left_eye = keypoints[1]
            right_eye = keypoints[2]
            left_ear = keypoints[3]
            right_ear = keypoints[4]
            nose = keypoints[0]
            # Calculate the angle of the line connecting the eyes.
            eye_angle = 0
            if right_eye[0] != left_eye[0]:
                eye_angle = math.degrees(math.atan2(
                    right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

            # Calculate the angle of the line connecting the ears.
            ear_angle = 0
            if right_ear[0] != left_ear[0]:
                ear_angle = math.degrees(math.atan2(
                    right_ear[1] - left_ear[1], right_ear[0] - left_ear[0]))

            # Average the angles for a more stable head tilt measurement
            average_angle = (eye_angle + ear_angle) / 2
            print('Default Angle: ', average_angle)

            head_tilting_left = False
            head_tilting_right = False
            head_tilting_upward = False

            nose_y = nose[1].item()
            # Check if the angle indicates a head tilt.
            # 1. Head Up and Down gesture (persistent state)
            if neutral_y_position_head is None:
                # Set the neutral position on the first frame
                neutral_y_position_head = nose_y

            if average_angle > tilt_threshold and average_angle < default_angle:
                # Angle increases when the head tilts to the left.
                head_tilting_left = True
            elif average_angle < -tilt_threshold and average_angle > -default_angle:
                # Angle decreases when the head tilts to the right.
                head_tilting_right = True
            elif neutral_y_position_head is not None:
                vertical_head_movement = neutral_y_position_head - nose_y
                print("Vertical head movement:", vertical_head_movement)
                if vertical_head_movement > head_movement_threshold:
                    head_tilting_upward = True

            # Determine the current action based on detected gestures
            if head_tilting_upward:
                current_action = 'space'
                current_action_text = "Action: Head tilting upward (Space)"
            elif head_tilting_left:
                current_action = 'a'
                current_action_text = "Action: Head tilting left (A)"
            elif head_tilting_right:
                current_action = 'd'
                current_action_text = "Action: Head tilting right (D)"
            else:
                # Head is facing straight (idle state)
                current_action = None
                current_action_text = "Action: Facing straight (Idle)"

    # --- Add stability logic ---
    if current_action == last_action:
        gesture_frame_count += 1
    else:
        gesture_frame_count = 1
    last_action = current_action

    # --- Control keyboard based on confirmed action ---
    # Only press the key if the gesture has been stable for a few frames
    if gesture_frame_count >= min_frames_to_hold and current_action and current_action != pressed_key:
        if pressed_key:
            keyboard.release(pressed_key)
        keyboard.press(current_action)
        pressed_key = current_action

    # Only release the key if no gesture is detected for a few frames
    elif not current_action and pressed_key:
        keyboard.release(pressed_key)
        pressed_key = None
        gesture_frame_count = 0
        last_action = None

    # Plot the YOLOv1 pose on the frame
    annotated_frame = results[0].plot()

    # Draw the action text on the image
    cv2.putText(annotated_frame, current_action_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with the drawn skeleton
    cv2.imshow('YOLOv11 Pose Estimation', annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
keyboard.release('a')
keyboard.release('d')
keyboard.release('space')
