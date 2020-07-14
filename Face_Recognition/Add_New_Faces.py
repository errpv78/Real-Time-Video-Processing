from imutils.video import VideoStream
import imutils
import time
import cv2
import os
from imutils import paths
import shutil
import face_recognition
import pickle
from os import path


# Function for adding new face in face recognition dataset through live stream from webcam
def add_face_data_from_video():
    # load OpenCV's Haar cascade for face detection from disk
    # haarcascade designed by OpenCV to detect the frontal face
    cascade = 'haarcascades/haarcascade_frontalface_default.xml'

    # Loading the Cascade classifier
    detector = cv2.CascadeClassifier(cascade)

    while True:
        name = input("Enter name of person: ")
        new_folder = 'face_recognition_dataset/' + name
        try:
            os.makedirs(new_folder)
            break
        except OSError as e:
            print('Folder Already exists')
            action = input('To delete existing folder press d, else any other key to change name of entered folder: ')
            if action == 'd':
                # os.rmdir: removes an empty directory
                # shutil.emtree: removes a directory and all its contents
                shutil.rmtree(new_folder)
                os.makedirs(new_folder)
                break

    output = new_folder
    print("To successfully encode the face please make atleast 10-15 clicks")
    print("[INFO] starting video stream...")

    # Starting the video stream
    cap = cv2.VideoCapture(0)
    total = 0

    # loop over the frames from the video stream
    while True:
        frame = cap.read()
        orig = frame.copy()

        # resizing with keeping the aspect ratio same
        frame = imutils.resize(frame, width=400)

        # Detect faces in the grayscale frame:
        # scaleFactor: Parameter specifying how much the image size is
        # reduced at each image scale.
        # minNeighbors: Parameter specifying how many neighbors each
        # candidate rectangle should have to retain it.
        # minSize: Minimum possible object size. Objects smaller than
        # that are ignored.
        rects = detector.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30))

        # loop over the face detections and draw them on the frame
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the output frame

        frame_msg = "Press K to click, q to quit"
        cv2.imshow(frame_msg, frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `k` key was pressed, write the *original* frame to disk
        # so we can later process it and use it for face recognition
        if key == ord("k"):
            total += 1
            p = os.path.sep.join([output, "{}.png".format(
                str(total).zfill(5))])
            cv2.imwrite(p, orig)
        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break

            # print the total faces saved and do a bit of cleanup
    print("[INFO] {} face images stored in {}".format(total, new_folder))
    print("[INFO] cleaning up...")
    cv2.destroyAllWindows()
    cap.release()


# Function for adding new face in face recognition dataset through images
def add_face_data_from_images():
    # haarcascade designed by OpenCV to detect the frontal face
    cascade = 'haarcascades/haarcascade_frontalface_default.xml'

    # Loading the Cascade classifier
    detector = cv2.CascadeClassifier(cascade)

    imagePaths = []
    while len(imagePaths) == 0:
        name = input('Enter name of folder in face_recognition_folder contaning images of person with that folder: ')
        dir_path = 'face_recognition_dataset/' + name
        imagePaths = list(paths.list_images(dir_path))
        if len(imagePaths) == 0:
            print('Directory not found or directory is empty')
    images = []
    labels = []
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=400)
        # update the data and labels lists, respectively
        images.append(image)
        labels.append(label)
        os.remove(imagePath)
    output = dir_path

    print("To effectively encode face please have atleast 10-15 pics")
    total = 0
    for frame in images:
        l = -1
        while True:
            cv2.imshow('Press s to select image, enter to rotate', frame)
            l = cv2.waitKey(0) & 0xFF
            if l == ord('s'):
                break
            else:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if l:
            cv2.destroyAllWindows()
            orig = frame.copy()
            frame = imutils.resize(frame, width=400)
            # detect faces in the grayscale frame
            rects = detector.detectMultiScale(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
                minNeighbors=5, minSize=(30, 30))
            # loop over the face detections and draw them on the frame
            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            total += 1
            p = os.path.sep.join([output, "{}.png".format(
                str(total).zfill(5))])
            cv2.imwrite(p, orig)
    print("[INFO] {} face images stored in {}".format(total, name))
    print("[INFO] cleaning up...")
