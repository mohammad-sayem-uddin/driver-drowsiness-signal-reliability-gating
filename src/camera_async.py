"""
Asynchronous Camera Capture
===========================
A non-blocking webcam interface.  Normally, cv2.VideoCapture.read() is a
blocking call, meaning the CPU idles while waiting for the USB bus and
camera sensor to return a frame. In an Edge AI context, this idle time
destroys FPS.

CameraAsync uses a background thread to continually pull frames into a
queue, ensuring the main processing loop never waits on I/O.
"""

import cv2
import threading
import time

class CameraAsync:
    def __init__(self, camera_id=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera ID {camera_id}")
            
        # Advisory settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Verify actual settings
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.ret = False
        self.frame = None
        self.stopped = False
        
        # Pull the very first frame to ensure we have data before the thread runs
        self.ret, self.frame = self.cap.read()
        
        # Start the background thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Background thread loop: continuously read frames from the camera."""
        while not self.stopped:
            if not self.cap.isOpened():
                break
                
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame
            else:
                # If we fail to read a frame, add a tiny sleep to avoid slamming the CPU
                time.sleep(0.01)

    def read(self):
        """
        Returns the most recently grabbed frame.
        Since this just reads memory updated by the background thread, it takes ~0.00ms.
        """
        return self.ret, self.frame

    def isOpened(self):
        return self.cap.isOpened() and not self.stopped

    def release(self):
        """Safely terminate the background thread and release hardware."""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
