import cv2
import time
import sys

class CameraStream:
    """
    A modular and production-ready class to handle real-time webcam streaming.
    Designed as a foundational template that can be expanded with AI processing
    (like MediaPipe drowsiness detection) in the `process_frame` method.
    """

    def __init__(self, camera_id=0, window_name="Live Camera Feed"):
        """
        Initializes the camera stream settings.
        
        Args:
            camera_id (int): The device index for the camera (0 is default).
            window_name (str): The name of the display window.
        """
        self.camera_id = camera_id
        self.window_name = window_name
        
        # FPS calculation variables
        self.prev_frame_time = 0
        self.new_frame_time = 0

    def initialize_camera(self):
        """
        Attempts to open the webcam and verifies the connection.
        
        Returns:
            cv2.VideoCapture: The initialized camera object if successful.
        """
        print(f"[INFO] Initializing webcam with ID {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)

        # Handle webcam error properly
        if not cap.isOpened():
            print(f"[ERROR] Could not open webcam (ID: {self.camera_id}).")
            print("Check if the camera is connected or if another app is using it.")
            sys.exit(1) # Exit safely with error code

        # Optional: Set camera resolution (e.g., 1280x720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        return cap

    def calculate_fps(self):
        """
        Calculates the Frames Per Second (FPS) based on the time elapsed 
        between the current and previous frame.
        
        Returns:
            int: The calculated FPS.
        """
        self.new_frame_time = time.time()
        # Ensure we don't divide by zero
        if self.new_frame_time == self.prev_frame_time:
            return 0
            
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        return int(fps)

    def process_frame(self, frame):
        """
        Pipeline for frame processing. This is where future AI models 
        (like MediaPipe Face Mesh for drowsiness detection) will be integrated.
        
        Args:
            frame (numpy.ndarray): The current video frame.
            
        Returns:
            numpy.ndarray: The processed frame ready for display.
        """
        # Step 1: Flip the frame horizontally for a natural "mirror" view
        frame = cv2.flip(frame, 1)

        # FUTURE AI PIPELINE GOES HERE
        # e.g., results = detector.process(frame)
        # e.g., draw_landmarks(frame, results)

        return frame

    def draw_overlay(self, frame, fps):
        """
        Draws text and UI elements onto the frame.
        
        Args:
            frame (numpy.ndarray): The current video frame.
            fps (int): The current frames per second.
        """
        # Draw FPS counter
        fps_text = f"FPS: {fps}"
        cv2.putText(frame, fps_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw exit instructions
        cv2.putText(frame, "Press 'q' to EXIT", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)

    def run(self):
        """
        The main loop that continuously captures frames, processes them, 
        and displays them until the user exits.
        """
        cap = self.initialize_camera()
        
        # Initialize time for the first FPS calculation
        self.prev_frame_time = time.time()
        
        print("[INFO] Stream started. Press 'q' in the window to exit.")

        while True:
            # 1. Capture the frame
            success, frame = cap.read()

            # Handle capture errors (e.g., webcam gets disconnected)
            if not success or frame is None:
                print("[WARNING] Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'
                continue

            # 2. Process the frame (AI integration point)
            processed_frame = self.process_frame(frame)

            # 3. Calculate metrics
            fps = self.calculate_fps()

            # 4. Draw overlays (FPS, instructions, etc.)
            self.draw_overlay(processed_frame, fps)

            # 5. Display the frame
            cv2.imshow(self.window_name, processed_frame)

            # 6. Listen for keyboard input to safely exit
            # cv2.waitKey(1) pauses for 1 millisecond to process GUI events
            # 0xFF handles cross-platform bit masking for key inputs
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exit signal received. Shutting down...")
                break

        # 7. Cleanup resources safely
        self.cleanup(cap)

    def cleanup(self, cap):
        """
        Releases the webcam hardware and destroys all UI windows.
        """
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows destroyed. Goodbye!")


if __name__ == "__main__":
    # Instantiate and run the camera stream
    stream = CameraStream(camera_id=0)
    stream.run()
