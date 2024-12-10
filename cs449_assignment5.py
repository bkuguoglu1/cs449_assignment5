import cv2
import mediapipe as mp
import math
import time
import os
import pyautogui
import platform
# Initialize video capture
cap = cv2.VideoCapture(0)

# Ensure the camera is opened successfully
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit(1)
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Gesture timers
pinch_hold_time = 0.1
three_finger_hold_time = 0.4
pinch_start_time = None
three_finger_start_time = None
# Gesture hold times
thumbs_hold_time = 0.4
pause_play_hold_time = 0.4
three_finger_hold_time = 0.4
menu_hold_time = 0.4

# Gesture configuration
GESTURE_MAPPING = {
    "Pinch": "Mouse Click",
    "Thumbs Up": "Volume Up",
    "Thumbs Down": "Volume Down",
    "Three-Finger Touch": "Show Windows",
    "Pause/Play": "Pause/Play Media",       # Example action
}
# Gesture state
is_pinch = False
is_three_finger = False
# Menu state
menu_active = False
menu_selected_option = 1
menu_options = [
    "Close Menu",             # Top-left
    "Set Timer (20 minutes)", # Top-right     
    "Terminate Program", # Bottom-left
    "Gesture List"      # Bottom-right
]


prev_index_y = None  # To store the Y-coordinate of the index finger in the previous frame

def detect_air_tap(index_tip, prev_index_y, downward_threshold=0.03, upward_threshold=0.02, tap_duration=0.3):
    """
    Detect an air tap gesture based on the index finger's vertical movement and timing.
    
    Args:
        index_tip: The current position of the index finger tip (landmark).
        prev_index_y: The Y-coordinate of the index finger in the previous frame.
        downward_threshold: Minimum downward movement to detect a downward tap motion.
        upward_threshold: Minimum upward movement to complete the tap gesture.
        tap_duration: Maximum allowed duration (in seconds) for the full tap gesture.

    Returns:
        tuple: (bool, float) 
            - bool: True if an air tap gesture is detected, False otherwise.
            - float: The current Y-coordinate of the index finger for the next frame.
    """
    if prev_index_y is None:
        return False, index_tip.y  # No previous data to compare

    # Calculate the vertical movement of the index finger
    delta_y = prev_index_y - index_tip.y

    # Static variables to track tap state and timing
    if not hasattr(detect_air_tap, "is_tapping"):
        detect_air_tap.is_tapping = False  # Initialize tap state
        detect_air_tap.tap_start_time = None  # Initialize tap start time

    if not detect_air_tap.is_tapping:
        # Detect the downward motion (start of the tap)
        if delta_y > downward_threshold:  # Finger moving downward quickly
            detect_air_tap.is_tapping = True  # Enter tapping state
            detect_air_tap.tap_start_time = time.time()  # Record start time
    else:
        # Ensure tap duration does not exceed the threshold
        if time.time() - detect_air_tap.tap_start_time > tap_duration:
            detect_air_tap.is_tapping = False  # Reset tapping state
            return False, index_tip.y  # Tap duration exceeded

        # Detect the upward motion (end of the tap)
        if delta_y < -upward_threshold:  # Finger moving upward quickly
            detect_air_tap.is_tapping = False  # Reset tapping state
            return True, index_tip.y  # Air tap gesture completed

    return False, index_tip.y  # No tap gesture detected


def trigger_mission_control():
    """Trigger macOS Mission Control using AppleScript."""
    os.system("osascript -e 'tell application \"System Events\" to key code 126 using control down'")
def change_volume(direction):
    """Change the volume on macOS using osascript."""
    if direction == 'up':
        os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) + 10)'")
    elif direction == 'down':
        os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) - 10)'")

def calculate_distance(landmark1, landmark2):
    """Calculate the Euclidean distance between two landmarks or a landmark and a tuple."""
    if isinstance(landmark2, tuple):  # If landmark2 is a tuple
        return math.sqrt((landmark1.x - landmark2[0]) ** 2 + (landmark1.y - landmark2[1]) ** 2)
    return math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

def move_cursor_safe(index_tip, screen_width, screen_height):
    """
    Moves the cursor safely within screen boundaries.
    
    Args:
        index_tip: The current position of the index finger tip (landmark).
        screen_width: The width of the screen.
        screen_height: The height of the screen.
    """
    # Calculate the normalized cursor position
    cursor_x = int((1 - index_tip.x) * screen_width)  # Flip X-axis
    cursor_y = int(index_tip.y * screen_height)
    
    # Ensure the cursor position is within screen boundaries
    cursor_x = max(0, min(cursor_x, screen_width - 1))
    cursor_y = max(0, min(cursor_y, screen_height - 1))
    
    # Move the cursor
    #pyautogui.moveTo(cursor_x, cursor_y)


def detect_pinch(thumb_tip, index_tip):
    """Detect pinch gesture."""
    distance = calculate_distance(thumb_tip, index_tip)
    return distance < 0.05  # Adjust threshold for pinch detection

#def detect_three_finger_touch(index_tip, middle_tip, ring_tip):
    """Detect three-finger touch gesture."""
    center_x = (index_tip.x + middle_tip.x + ring_tip.x) / 3
    center_y = (index_tip.y + middle_tip.y + ring_tip.y) / 3

    distances = [
        calculate_distance(index_tip, (center_x, center_y)),
        calculate_distance(middle_tip, (center_x, center_y)),
        calculate_distance(ring_tip, (center_x, center_y))
    ]
    return all(d < 0.05 for d in distances)  # Adjust threshold for touch detection

def detect_five_finger_slide(index_tip, middle_tip, ring_tip, thumb_tip, pinky_tip):
    """Detect if all five fingers are near a central point."""
    # Calculate the central point of the five fingers
    center_x = (index_tip.x + middle_tip.x + ring_tip.x + thumb_tip.x + pinky_tip.x) / 5
    center_y = (index_tip.y + middle_tip.y + ring_tip.y + thumb_tip.y + pinky_tip.y) / 5

    # Calculate distances of each fingertip from the central point
    distances = [
        calculate_distance(index_tip, (center_x, center_y)),
        calculate_distance(middle_tip, (center_x, center_y)),
        calculate_distance(ring_tip, (center_x, center_y)),
        calculate_distance(thumb_tip, (center_x, center_y)),
        calculate_distance(pinky_tip, (center_x, center_y))
    ]

    # Return True if all fingers are close to the central point
    return all(d < 0.05 for d in distances)  # Adjust threshold as necessary
thumbs_start_time = None  # Global variable to track the gesture hold time

def detect_thumbs_up_down(thumb_tip, wrist, thumbs_start_time, upward_threshold=0.03, downward_threshold=0.03, hold_time=0.5):
    """
    Detect thumbs up or thumbs down gesture.

    Args:
        thumb_tip: The current position of the thumb tip landmark.
        wrist: The current position of the wrist landmark.
        thumbs_start_time: Timer tracking when the thumb gesture started.
        upward_threshold: Minimum vertical distance to detect a thumbs-up gesture.
        downward_threshold: Minimum vertical distance to detect a thumbs-down gesture.
        hold_time: Time to hold the gesture to confirm detection.

    Returns:
        tuple: (gesture_type, thumbs_start_time)
            - gesture_type: 'up' for thumbs up, 'down' for thumbs down, None otherwise.
            - thumbs_start_time: Updated timer value.
    """
    vertical_difference = thumb_tip.y - wrist.y  # Calculate vertical distance from thumb to wrist

    if vertical_difference < -upward_threshold:  # Thumb is above the wrist
        if thumbs_start_time is None:
            thumbs_start_time = time.time()  # Start the timer
        elif time.time() - thumbs_start_time > hold_time:  # Gesture held for enough time
            return 'up', None  # Reset timer after detection
    elif vertical_difference > downward_threshold:  # Thumb is below the wrist
        if thumbs_start_time is None:
            thumbs_start_time = time.time()  # Start the timer
        elif time.time() - thumbs_start_time > hold_time:  # Gesture held for enough time
            return 'down', None  # Reset timer after detection
    else:
        thumbs_start_time = None  # Reset timer if gesture is not maintained

    return None, thumbs_start_time


def navigate_menu(index_tip, middle_tip, thumb_tip, wrist):
    """Navigate the menu using gestures."""
    global menu_selected_option, menu_active
    if index_tip.y < middle_tip.y:  # Navigate up
        menu_selected_option = max(1, menu_selected_option - 1)
        time.sleep(0.3)  # Prevent rapid selection
    elif index_tip.y > middle_tip.y:  # Navigate down
        menu_selected_option = min(len(menu_options), menu_selected_option + 1)
        time.sleep(0.3)
    elif thumb_tip.y < wrist.y:  # Select option
        perform_menu_action(menu_selected_option)
def is_flat_palm(hand_landmarks, wrist):
    """Detect if the hand is in a flat palm posture."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

def is_stop_sign(hand_landmarks):
    """
    Detect if the hand is in a stop sign gesture (all fingers extended).
    
    Args:
        hand_landmarks: The landmarks of the detected hand.

    Returns:
        bool: True if the hand is in a stop sign gesture, False otherwise.
    """
    fingers_extended = []
    for finger_tip, finger_base in [
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]:
        # Loosen the condition with a small margin for noise
        if hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_base].y + 0.05:
            fingers_extended.append(True)
        else:
            fingers_extended.append(False)

    return all(fingers_extended)  # True if all fingers are extended



def perform_action(gesture_name):
    """Perform the given gesture action."""
    print(f"Performing action: {gesture_name}")
    action = GESTURE_MAPPING.get(gesture_name)
    if action == "Mouse Click":
        pyautogui.click()
    elif action == "Volume Up":
        pyautogui.press('volumeup')
    elif action == "Volume Down":
        pyautogui.press('volumedown')
    elif action == "Pause/Play":
        if platform.system() == "Darwin":
            os.system("osascript -e 'tell application \"System Events\" to key code 16'")
        elif platform.system() == "Windows":
            pyautogui.hotkey('playpause')
    elif action == "Show Windows":
        if platform.system() == "Windows":
            pyautogui.hotkey('win', 'tab')
        elif platform.system() == "Darwin":
            os.system("open -a Mission\\ Control")


def draw_box_menu(frame, cursor_x, cursor_y):
    """
    Draws a four-option menu on the screen and highlights the option near the cursor.
    
    Args:
        frame: The video frame to draw on.
        cursor_x: Normalized X-coordinate of the cursor (0 to 1).
        cursor_y: Normalized Y-coordinate of the cursor (0 to 1).
        
    Returns:
        int: The index of the highlighted menu option.
    """
    global menu_options

    h, w, _ = frame.shape  # Frame dimensions
    box_size = 200  # Size of each menu box

    # Coordinates for menu boxes based on the corrected layout
    positions = [
        (0, 0, box_size, box_size),  # Top-left: Close Menu
        (w - box_size, 0, w, box_size),  # Top-right: Set Timer
        (0, h - box_size, box_size, h),  # Bottom-left: Gesture List
        (w - box_size, h - box_size, w, h),  # Bottom-right: Terminate Program
    ]

    highlighted_option = -1  # Default to no highlight

    for i, (x1, y1, x2, y2) in enumerate(positions):
        # Draw box
        color = (200, 200, 200)  # Default box color
        if x1 / w <= cursor_x <= x2 / w and y1 / h <= cursor_y <= y2 / h:
            color = (0, 255, 0)  # Highlighted box color
            highlighted_option = i  # Set highlighted option index
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

        # Add menu text
        if i < len(menu_options):  # Prevent out-of-range errors
            text_x = x1 + 10
            text_y = y1 + 30
            cv2.putText(frame, menu_options[i], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return highlighted_option



def perform_menu_action(option, frame, menu_active):
    """
    Perform an action based on the selected menu option.

    Args:
        option (int): The selected menu option (0-based index).
        frame: The current frame to display feedback.
        menu_active (bool): Menu state (True if active).
    """
    if option == 0:  # Close Menu
        print("Menu Closed")
        return False  # Deactivate menu
    elif option == 1:  # Set Timer
        print("Timer started for 20 minutes.")
        cv2.putText(frame, "Timer started for 20 minutes.", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return menu_active
    elif option == 2:  # Terminate Program
        print("Terminating program...")
        cv2.putText(frame, "Terminating...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.destroyAllWindows()
        exit(0)
    elif option == 3:  # Gesture Guide
        print("Displaying Gesture Guide...")
        gestures = [
            "1. Air Tap - Click",
            "2. Thumbs Up - Volume Up",
            "3. Thumbs Down - Volume Down",
            "4. Five-Finger Slide - Trigger Mission Control (macOS)",
            "5. Stop Sign - Activate Menu"
        ]

        # Copy the frame to avoid modifying the original
        gesture_frame = frame.copy()

        # Display gestures once on the copied frame
        for i, gesture in enumerate(gestures):
            cv2.putText(
                gesture_frame,
                gesture,
                (10, 100 + i * 30),  # Display each gesture below the previous one
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        # Show the frame with the gestures
        cv2.imshow('Gesture-Based Interaction', gesture_frame)

        # Wait for user to press 'q' to close the guide
        print("Press 'q' to exit Gesture Guide.")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("Gesture Guide closed.")
                break

        # After the guide is closed, return to menu
        return menu_active


# Open webcam feed
cap = cv2.VideoCapture(0)

try:
    gesture_guide_active = False  # Tracks whether the Gesture Guide is active
    thumbs_start_time = None  # Timer for thumbs gesture
    menu_active = False  # Menu state
    menu_start_time = None  # Timer for menu activation
    menu_activation_time = None  # Timer to track when menu was activated
    highlighted_option = -1  # Default highlighted menu option
    focus_start_time = None  # Timer to track focus on a menu option
    focused_option = -1  # Currently focused menu option
    prev_index_y = None  # For air tap detection
    air_tap_cooldown_time = 0.5  # Cooldown to prevent immediate selection
    air_tap_last_time = 0  # Last air tap time
    three_finger_start_time = None  # Timer for five-finger slide gesture
    feedback_message = ""  # Message to display for feedback
    feedback_start_time = None  # Timer for feedback message
    feedback_duration = 2.0  # Feedback display duration

    focus_duration = 2.0  # Required focus time (in seconds)

    # Screen dimensions for mapping the cursor
    screen_width, screen_height = pyautogui.size()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Default cursor position
        cursor_x, cursor_y = 0.5, 0.5

        if results.multi_hand_landmarks:
            # Detect two-hand stop gesture for menu activation
            if len(results.multi_hand_landmarks) == 2:  # Check for two hands
                hand1, hand2 = results.multi_hand_landmarks[:2]
                hand1_stop = is_stop_sign(hand1)
                hand2_stop = is_stop_sign(hand2)

                if hand1_stop and hand2_stop:
                    if not menu_active and menu_start_time is None:
                        menu_start_time = time.time()
                        feedback_message = "Two-hand stop gesture detected..."
                        feedback_start_time = time.time()
                    elif not menu_active and time.time() - menu_start_time > 0.5:  # Hold for 0.5 seconds
                        menu_active = True
                        menu_activation_time = time.time()  # Record menu activation time
                        feedback_message = "Menu Activated"
                        feedback_start_time = time.time()
                        highlighted_option = -1  # Reset highlighted option
                        focus_start_time = None  # Reset focus timer
                        focused_option = -1  # Reset focused option
                        menu_start_time = None  # Reset menu activation timer
                else:
                    menu_start_time = None  # Reset if the gesture is interrupted

            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # Update cursor position and ensure within screen bounds
                cursor_x = int((1 - index_tip.x) * screen_width)  # Flip X-axis
                cursor_y = int(index_tip.y * screen_height)
                cursor_x = max(0, min(cursor_x, screen_width - 1))
                cursor_y = max(0, min(cursor_y, screen_height - 1))

                # Move cursor on screen
                pyautogui.moveTo(cursor_x, cursor_y)

                if menu_active:
                    # Cursor navigation for menu
                    highlighted_option = draw_box_menu(frame, cursor_x / screen_width, cursor_y / screen_height)

                    # Focus and hold for selection
                    if focused_option != highlighted_option:
                        focused_option = highlighted_option
                        focus_start_time = time.time()  # Reset focus timer
                    elif focused_option == highlighted_option and focus_start_time is not None:
                        elapsed_focus_time = time.time() - focus_start_time
                        progress = min(int(elapsed_focus_time / focus_duration * 100), 100)  # Progress in percentage
                        cv2.putText(
                            frame,
                            f"Focusing: {progress}%",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        if elapsed_focus_time > focus_duration:  # Focus held for required time
                            print(f"Selected Option: {highlighted_option}")
                            
                            if highlighted_option == 0:  # Close Menu
                                print("Menu Closed")
                                menu_active = False  # Close the menu

                            elif highlighted_option == 1:  # Set Timer
                                print("Timer started for 20 minutes.")
                                cv2.putText(frame, "Timer started for 20 minutes.", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            elif highlighted_option == 2:  # Terminate Program
                                print("Terminating program...")
                                cv2.putText(frame, "Terminating...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cap.release()
                                cv2.destroyAllWindows()
                                exit(0)

                            elif highlighted_option == 3:  # Gesture Guide
                                print("Gesture Guide activated.")
                                gestures = [
                                    "1. Air Tap - Click",
                                    "2. Thumbs Up - Volume Up",
                                    "3. Thumbs Down - Volume Down",
                                    "4. Five-Finger Slide - Trigger Mission Control (macOS)",
                                    "5. Stop Sign - Activate Menu"
                                ]
                                # Set a flag to keep Gesture Guide visible
                                gesture_guide_active = True

                            focus_start_time = None  # Reset focus timer

                    # Ensure the Gesture Guide remains visible if active
                    if 'gesture_guide_active' in locals() and gesture_guide_active:
                        gestures = [
                            "1. Air Tap - Click",
                            "2. Thumbs Up - Volume Up",
                            "3. Thumbs Down - Volume Down",
                            "4. Five-Finger Slide - Trigger Mission Control (macOS)",
                            "5. Stop Sign - Activate Menu"
                        ]
                        for i, gesture in enumerate(gestures):
                            cv2.putText(
                                frame,
                                gesture,
                                (10, 100 + i * 30),  # Display each gesture below the previous one
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2
                            )

                    # Show the updated frame
                    cv2.imshow('Gesture-Based Interaction', frame)
                else:
                    # Regular interaction logic when not in menu

                    # Thumbs Up/Down for Volume Control
                    thumbs_gesture, thumbs_start_time = detect_thumbs_up_down(
                        thumb_tip, wrist, thumbs_start_time, upward_threshold=0.2, downward_threshold=0.2, hold_time=0.5
                    )
                    if thumbs_gesture == 'up':
                        change_volume('up')
                        feedback_message = "Thumbs Up - Volume Up"
                        feedback_start_time = time.time()
                    elif thumbs_gesture == 'down':
                        change_volume('down')
                        feedback_message = "Thumbs Down - Volume Down"
                        feedback_start_time = time.time()

                    # Detect five-finger slide gesture for macOS Mission Control
                    if detect_five_finger_slide(index_tip, middle_tip, ring_tip, thumb_tip, pinky_tip):
                        if three_finger_start_time is None:
                            three_finger_start_time = time.time()
                        elif time.time() - three_finger_start_time > 0.5:  # Hold for 0.5 seconds
                            feedback_message = "Five-Finger Slide Detected"
                            feedback_start_time = time.time()
                            print("Gesture: Five-Finger Slide")
                            trigger_mission_control()  # Trigger Mission Control on macOS
                            three_finger_start_time = None

                    # Detect air tap for mouse click
                    air_tap_detected, prev_index_y = detect_air_tap(index_tip, prev_index_y)
                    if air_tap_detected:
                        pyautogui.click()
                        feedback_message = "Air Tap - Mouse Click"
                        feedback_start_time = time.time()

        else:
            # Reset state if no hands are detected
            menu_start_time = None
            three_finger_start_time = None

        # Display feedback message
        if feedback_message and feedback_start_time and time.time() - feedback_start_time < feedback_duration:
            cv2.putText(
                frame,
                feedback_message,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # Show frame
        cv2.imshow('Gesture-Based Interaction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
