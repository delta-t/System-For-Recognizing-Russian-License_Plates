import cv2
import os
import numpy as np
import time


def frame_preprocessing(frame, car_net, classes, colors, limit_for_confidence, number_cascade):
    # analyzing the input frame and getting bounding boxes for all cars on the frame and all car license plates
    out_frame = frame.copy()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    car_net.setInput(blob)
    detections = car_net.forward()
    # analyzing the input frame
    for ind in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the predictions
        confidence = detections[0, 0, ind, 2]
        # Filter out weak detections by ensuring the confidence is greater than
        # the minimum confidence
        if confidence > limit_for_confidence:
            # Extract the index of the class labels from detections, then compute
            # the (x, y)-coordinates of the bounding box for the object
            idx = int(detections[0, 0, ind, 1])
            box = detections[0, 0, ind, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding boxes for everything car
            if classes[idx] == 'car':
                label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
                car_frame = frame[startY:endY, startX:endX]
                car_number = car_frame.copy()
                height_inc_car, width_inc_car = car_number.shape[:2]
                if width_inc_car > 0.1 * w:
                    # looking for a license plate on car frame
                    plaques = []
                    plaques = number_cascade.detectMultiScale(car_number, scaleFactor=1.3,
                                                              minNeighbors=4)

                    if len(plaques) > 0:
                        for (xx, yy, ww, hh) in plaques:
                            cv2.rectangle(out_frame, (startX + xx, startY + yy),
                                          (startX + xx + ww, startY + yy + hh), (0, 0, 255), 2)
                    cv2.rectangle(out_frame, (startX, startY), (endX, endY),
                                  colors[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(out_frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, colors[idx], 2)
    return out_frame


if __name__ == '__main__':
    # read any image, initialize MobileNetSSDNet and HaarCascadeClassifier and try to analyze the image
    first_frame = cv2.imread("input.jpg")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    caffe_net_filepath = os.path.join(script_dir, 'MobileNetSSD_deploy.caffemodel')
    caffe_net_filepath = os.path.abspath(caffe_net_filepath)
    proto_filepath = os.path.join(script_dir, 'MobileNetSSD_deploy.prototxt.txt')
    proto_filepath = os.path.abspath(proto_filepath)
    car_Net = cv2.dnn.readNetFromCaffe(proto_filepath, caffe_net_filepath)
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    limitForConfidence = 0.4
    NumberCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
    start = time.time()
    first_out = frame_preprocessing(first_frame, car_Net, CLASSES, COLORS, limitForConfidence, NumberCascade)

    # print elapsed time and show the result
    print(time.time() - start)
    cv2.imshow("first", first_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
