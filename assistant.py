import cv2
import time
import numpy as np
import mediapipe as mp
import pyautogui
import math
import sys

# For volume control on Windows
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class HandGestureAssistant:
    def __init__(self):
        # Initialize mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Initialize Windows volume control
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))
        self.volume_range = self.volume.GetVolumeRange()
        
        # System control variables
        self.prev_hand_state = None
        self.gesture_hold_frames = 0
        self.last_gesture_time = time.time()
        self.cooldown = 0.5  # seconds between gesture activations
        self.current_mode = "IDLE"  # Current control mode
        self.prev_positions = []  # Store previous positions for smoothing
        self.cursor_active = False  # For cursor control mode
        self.scroll_active = False  # For scroll control
        
        # Volume control
        self.volume_smoothing = []  # For volume smoothing
        self.max_volume_samples = 5
        self.current_volume = self.volume.GetMasterVolumeLevelScalar()
        
        # Cursor control
        self.cursor_smoothing = []
        self.max_cursor_samples = 5
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialization message
        print("Hand Gesture Assistant initialized!")
        print("Available gestures:")
        print("- Thumbs up: Activate system")
        print("- Thumbs down: Deactivate system")
        print("- Fist: Pause media")
        print("- Open palm: Play media")
        print("- Victory sign with thumb: Next/previous track")
        print("- Three fingers: Volume up/down")
        print("- 'L' shape: Cursor control")
        print("- Two fingers: Scroll")
        print("- Middle finger: Close program")
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def get_finger_states(self, hand_landmarks, image_width, image_height):
        """Determine which fingers are extended"""
        finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_anchors = [
            self.mp_hands.HandLandmark.THUMB_IP,  # Thumb has different anchor
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        
        extended = [False] * 5
        
        # Special handling for thumb based on angle
        thumb_tip = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x * image_width,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y * image_height
        ])
        thumb_ip = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x * image_width,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y * image_height
        ])
        wrist = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * image_width,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * image_height
        ])
        
        # For the other fingers, compare tip position to PIP joint
        for i in range(1, 5):
            tip = np.array([
                hand_landmarks.landmark[finger_tips[i]].x * image_width,
                hand_landmarks.landmark[finger_tips[i]].y * image_height
            ])
            pip = np.array([
                hand_landmarks.landmark[finger_anchors[i]].x * image_width,
                hand_landmarks.landmark[finger_anchors[i]].y * image_height
            ])
            
            # Finger is extended if tip is higher than PIP joint
            if tip[1] < pip[1]:  # Y decreases upward
                extended[i] = True
        
        # Special check for thumb based on position relative to index finger
        index_mcp = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height
        ])
        
        # Check if thumb is to the side of the index finger (extended)
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x:
            extended[0] = True
            
        return extended
    
    def detect_gestures(self, hand_landmarks, image_width, image_height):
        """Detect various hand gestures"""
        # Get finger extension state
        extended = self.get_finger_states(hand_landmarks, image_width, image_height)
        
        # Get fingertip positions
        thumb_tip = (hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x * image_width,
                     hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y * image_height)
        index_tip = (hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                     hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
        middle_tip = (hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width,
                      hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)
        
        wrist = (hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * image_width,
                 hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * image_height)
        
        # Calculate hand center
        hand_center = (0, 0)
        num_landmarks = 21
        for i in range(num_landmarks):
            hand_center = (hand_center[0] + hand_landmarks.landmark[i].x * image_width / num_landmarks,
                          hand_center[1] + hand_landmarks.landmark[i].y * image_height / num_landmarks)
        
        # Detect open palm (all fingers extended)
        is_palm = all(extended)
        
        # Detect fist (no fingers extended)
        is_fist = not any(extended)
        
        # Detect pinch (thumb and index finger close)
        thumb_index_distance = self.calculate_distance(thumb_tip, index_tip)
        is_pinch = thumb_index_distance < 40  # Adjust threshold as needed
        
        # Detect victory sign with thumb (index, middle and thumb extended, others closed)
        is_victory_with_thumb = extended[0] and extended[1] and extended[2] and not extended[3] and not extended[4]
        
        # Detect thumbs up (only thumb extended and pointing up)
        is_thumbs_up = extended[0] and not any(extended[1:]) and thumb_tip[1] < wrist[1]
        
        # Detect thumbs down (only thumb extended and pointing down)
        is_thumbs_down = extended[0] and not any(extended[1:]) and thumb_tip[1] > wrist[1]
        
        # Detect middle finger (only middle finger extended)
        is_middle_finger = not extended[0] and not extended[1] and extended[2] and not extended[3] and not extended[4]
        
        # Detect L shape (thumb and index extended, others closed)
        is_L_shape = extended[0] and extended[1] and not extended[2] and not extended[3] and not extended[4]
        
        # Detect two fingers (index and middle extended, others closed)
        is_two_fingers = not extended[0] and extended[1] and extended[2] and not extended[3] and not extended[4]
        
        # Detect three fingers (index, middle, and ring extended, others closed)
        is_three_fingers = not extended[0] and extended[1] and extended[2] and extended[3] and not extended[4]
        
        # Return detected gestures
        return {
            "palm": is_palm,
            "fist": is_fist,
            "pinch": is_pinch,
            "victory_with_thumb": is_victory_with_thumb,
            "thumbs_up": is_thumbs_up,
            "thumbs_down": is_thumbs_down,
            "middle_finger": is_middle_finger,
            "L_shape": is_L_shape,
            "two_fingers": is_two_fingers,
            "three_fingers": is_three_fingers,
            "hand_center": hand_center,
            "index_tip": index_tip,
            "thumb_index_distance": thumb_index_distance
        }
    
    def smooth_value(self, value, value_list, max_samples):
        """Apply smoothing to a value using a moving average"""
        value_list.append(value)
        if len(value_list) > max_samples:
            value_list.pop(0)
        return sum(value_list) / len(value_list)
    
    def process_gestures(self, gestures, frame):
        """Map detected gestures to system controls"""
        current_time = time.time()
        frame_height, frame_width, _ = frame.shape
        
        # Process gesture actions with cooldown
        process_action = False
        if current_time - self.last_gesture_time >= self.cooldown:
            process_action = True
        
        hand_center = gestures["hand_center"]
        index_tip = gestures["index_tip"]
        
        # Add current position to history
        self.prev_positions.append(hand_center)
        if len(self.prev_positions) > 5:
            self.prev_positions.pop(0)
        
        # Calculate movement direction if we have enough history
        movement_x, movement_y = 0, 0
        if len(self.prev_positions) >= 3:
            movement_x = self.prev_positions[-1][0] - self.prev_positions[0][0]
            movement_y = self.prev_positions[-1][1] - self.prev_positions[0][1]
        
        # Significant movement threshold
        movement_threshold = 30
        has_x_movement = abs(movement_x) > movement_threshold
        has_y_movement = abs(movement_y) > movement_threshold
        
        # ACTIVATION GESTURES
        
        # Thumbs up activates the system
        if gestures["thumbs_up"] and self.current_mode == "IDLE" and process_action:
            self.current_mode = "ACTIVE"
            print("System activated")
            self.last_gesture_time = current_time
        
        # Thumbs down deactivates the system
        elif gestures["thumbs_down"] and self.current_mode != "IDLE" and process_action:
            prev_mode = self.current_mode
            self.current_mode = "IDLE"
            # Reset active control modes
            self.cursor_active = False
            self.scroll_active = False
            print(f"System deactivated (exited {prev_mode} mode)")
            self.last_gesture_time = current_time
        
        # Middle finger closes the program
        elif gestures["middle_finger"] and self.current_mode != "IDLE" and process_action:
            print("Closing Hand Gesture Assistant...")
            self.last_gesture_time = current_time
            cv2.putText(frame, "Fucking Off", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow('Hand Gesture Assistant', frame)
            cv2.waitKey(1000)
            self.cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        
        # MODE SELECTION GESTURES (when in ACTIVE mode)
        
        # L shape activates cursor control
        elif gestures["L_shape"] and self.current_mode == "ACTIVE" and process_action:
            self.current_mode = "CURSOR"
            self.cursor_active = True
            self.cursor_smoothing = []  # Reset cursor smoothing
            print(f"Cursor control mode activated (Screen: {self.screen_width}x{self.screen_height})")
            self.last_gesture_time = current_time
        
        # Two fingers activates scroll control
        elif gestures["two_fingers"] and self.current_mode == "ACTIVE" and process_action:
            self.current_mode = "SCROLL"
            self.scroll_active = True
            print("Scroll control mode activated")
            self.last_gesture_time = current_time
        
        # Three fingers activates volume control
        elif gestures["three_fingers"] and self.current_mode == "ACTIVE" and process_action:
            self.current_mode = "VOLUME"
            self.volume_smoothing = []  # Reset volume smoothing
            print("Volume control mode activated")
            self.last_gesture_time = current_time
        
        # Victory with thumb activates media navigation
        elif gestures["victory_with_thumb"] and self.current_mode == "ACTIVE" and process_action:
            self.current_mode = "MEDIA_NAV"
            print("Media navigation mode activated")
            self.last_gesture_time = current_time
        
        # DIRECT ACTIONS (can be used from ACTIVE mode)
        
        # Fist for pause
        elif gestures["fist"] and (self.current_mode == "ACTIVE" or self.current_mode == "MEDIA_NAV") and process_action:
            pyautogui.press('pause')
            print("Media paused")
            self.last_gesture_time = current_time
        
        # Open palm for play
        elif gestures["palm"] and (self.current_mode == "ACTIVE" or self.current_mode == "MEDIA_NAV") and process_action:
            pyautogui.press('playpause')
            print("Media play/pause")
            self.last_gesture_time = current_time
        
        # PROCESS ACTIVE CONTROL MODES
        
        # Cursor control - actively follows index finger with improved smoothing
        if self.current_mode == "CURSOR" and gestures["L_shape"]:
            # Map webcam coordinates to screen coordinates with adjustment for aspect ratio
            webcam_aspect = frame_width / frame_height
            screen_aspect = self.screen_width / self.screen_height
            
            # Adjust for the aspect ratio difference
            if webcam_aspect > screen_aspect:  # Webcam is wider
                # Scale based on height
                relative_x = index_tip[0] / frame_width
                relative_y = index_tip[1] / frame_height
            else:  # Screen is wider
                # Scale based on width
                relative_x = index_tip[0] / frame_width
                relative_y = index_tip[1] / frame_height
            
            # Apply non-linear mapping for better precision (optional)
            # This makes movements near the center more precise
            relative_x = 0.5 + (relative_x - 0.5) * 1.3  # Adjust sensitivity
            relative_y = 0.5 + (relative_y - 0.5) * 1.3
            
            # Clamp values
            relative_x = max(0, min(1, relative_x))
            relative_y = max(0, min(1, relative_y))
            
            cursor_x = int(relative_x * self.screen_width)
            cursor_y = int(relative_y * self.screen_height)
            
            # Apply smoothing
            if len(self.cursor_smoothing) < 2:  # Not enough data for smoothing yet
                self.cursor_smoothing.append((cursor_x, cursor_y))
            else:
                smoothed_x = self.smooth_value(cursor_x, [p[0] for p in self.cursor_smoothing], self.max_cursor_samples)
                smoothed_y = self.smooth_value(cursor_y, [p[1] for p in self.cursor_smoothing], self.max_cursor_samples)
                
                # Move cursor smoothly
                pyautogui.moveTo(int(smoothed_x), int(smoothed_y), duration=0.05)
                
                # Add current position to smoothing list
                self.cursor_smoothing.append((cursor_x, cursor_y))
                if len(self.cursor_smoothing) > self.max_cursor_samples:
                    self.cursor_smoothing.pop(0)
            
            # Add click capability - when thumb and index get very close
            if gestures["thumb_index_distance"] < 20:
                pyautogui.click()
                print("Mouse click")
                time.sleep(0.3)  # Prevent multiple clicks
            
            # Draw cursor position feedback on frame
            cv2.circle(frame, (int(index_tip[0]), int(index_tip[1])), 10, (0, 0, 255), -1)
            cv2.putText(frame, "CURSOR", (int(index_tip[0])+15, int(index_tip[1])-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Scroll control
        elif self.current_mode == "SCROLL" and gestures["two_fingers"]:
            if has_y_movement:
                # Convert movement to scroll amount
                scroll_amount = int(movement_y)  # More sensitive scrolling
                pyautogui.scroll(scroll_amount)  # Positive for scroll up, negative for scroll down
                self.prev_positions = self.prev_positions[-2:]  # Keep only recent positions
        
        # Volume control with three fingers - smoother implementation
        elif self.current_mode == "VOLUME" and gestures["three_fingers"]:
            if has_y_movement:
                # Get relative hand position within frame
                rel_hand_y = hand_center[1] / frame_height
                
                # Map position to volume (top of frame = max volume, bottom = min volume)
                # Invert and constrain to reasonable range (0.1-0.9 of frame)
                rel_volume = 1.0 - max(0.1, min(0.9, rel_hand_y))
                rel_volume = (rel_volume - 0.1) / 0.8  # Rescale to 0-1
                
                # Apply smoothing
                smoothed_volume = self.smooth_value(rel_volume, self.volume_smoothing, self.max_volume_samples)
                
                # Set volume
                self.volume.SetMasterVolumeLevelScalar(smoothed_volume, None)
                self.current_volume = smoothed_volume
                
                # Draw volume indicator
                bar_width = int(smoothed_volume * 300)
                cv2.rectangle(frame, (20, 50), (320, 80), (0, 0, 0), -1)
                cv2.rectangle(frame, (20, 50), (20 + bar_width, 80), (0, 255, 0), -1)
                cv2.putText(frame, f"Volume: {int(smoothed_volume * 100)}%", (30, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Media navigation with victory sign + thumb
        elif self.current_mode == "MEDIA_NAV" and gestures["victory_with_thumb"]:
            if has_x_movement and abs(movement_x) > abs(movement_y) and process_action:
                if movement_x > 0:
                    pyautogui.press('nexttrack')
                    print("Next track")
                else:
                    pyautogui.press('prevtrack')
                    print("Previous track")
                self.last_gesture_time = current_time
                self.prev_positions = []  # Reset after action
        
        # Draw current mode on frame
        cv2.putText(frame, f"Mode: {self.current_mode}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Main loop for hand gesture assistant"""
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    print("Failed to read from webcam")
                    break
                
                # Flip the frame horizontally for more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for mediapipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame
                results = self.hands.process(rgb_frame)
                
                h, w, c = frame.shape
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Detect gestures
                        gestures = self.detect_gestures(hand_landmarks, w, h)
                        
                        # Process gestures to control system
                        frame = self.process_gestures(gestures, frame)
                else:
                    # No hands detected, reset position history
                    self.prev_positions = []
                    
                    # If we were in cursor or scroll mode, we need to reset
                    if self.cursor_active or self.scroll_active:
                        if self.current_mode != "IDLE":
                            self.current_mode = "ACTIVE"
                            self.cursor_active = False
                            self.scroll_active = False
                
                # Display the frame
                cv2.imshow('Hand Gesture Assistant', frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Hand Gesture Assistant terminated")


if __name__ == "__main__":
    assistant = HandGestureAssistant()
    assistant.run()
