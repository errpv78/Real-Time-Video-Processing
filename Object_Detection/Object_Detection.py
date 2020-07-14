from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2


def object_detection():
    # construct the argument parser and parse the arguments

    prototxt = 'caffe_models/MobileNetSSD_deploy.prototxt.txt'
    model = 'caffe_models/MobileNetSSD_deploy.caffemodel'
    confidence = 0.2

    # initialize the list of class labels MobileNet SSD was trained to detect
    # and generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")

    # Reads a network model stored in Caffe framework's format.
    # prototxt: path to .prototxt file with text description of network architecture.
    # caffeModel: path to .caffemodel file with learned network.
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)
    fps = FPS().start()

    # loop over the frames from the video stream
    while True:
        # resize the video stream window at a maximum width of 500 pixels
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=1000)

        # grab the frame dimensions and convert it to a blob
        # Binary Large Object = BLOB
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and get the detections
        net.setInput(blob)
        detections = net.forward()

        objects_detected = set()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the probability of the prediction
            probability = detections[0, 0, i, 2]

            # filter out weak detections by ensuring that probability is
            # greater than the min probability
            if probability > confidence:
                # extract the index of the class label from the
                # 'detections', then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx], probability * 100)
                objects_detected.add(label.split(':')[0])
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        if 'person' not in objects_detected:
            cv2.putText(frame, 'No Person Detected', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()
    cap.release()