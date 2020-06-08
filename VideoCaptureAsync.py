import threading
import time

import cv2


class VideoCaptureAsync:
    """
    Get some frames from video device using thread.
    """

    def __init__(self, src, width=640, height=480):
        # initialize the video camera stream, read the first frame
        # from the stream and initialize the thread
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()

        self.read_lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())

        # initialize the variable used to indicate if the thread should
        # be started
        self.started = False

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        # start the thread to read frames from the video device
        if self.started:
            print('[!] Asynchronous video capturing already has been started.')
            return None
        self.thread.start()
        # set the thread indicator
        self.started = True
        return self

    def update(self):
        # keep looping infinitely while the thread is started
        while self.started:
            # read the next frame from the video device
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        # give the frame most recently read
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        # indicate that the thread should be stopped and join
        self.started = False
        self.thread.join()
        self.cap.release()


if __name__ == '__main__':
    cap = VideoCaptureAsync(src=0).start()
    time.sleep(2.0)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Camera is not responding")
            break
    cap.stop()
    cv2.destroyAllWindows()
