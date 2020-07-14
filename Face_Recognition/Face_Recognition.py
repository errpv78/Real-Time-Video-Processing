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


def encode_face_data():
    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")

    # Face recognition dataset
    dataset = 'face_recognition_dataset/'

    # Output encoding file
    encodings_file = '../Face_Liveness/encodings.pickle'

    # Face detection method (cnn or hog)
    detection_method = 'cnn'
    imagePaths = list(paths.list_images(dataset))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):

        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
                                                     len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        # face_recognition.face_locations
        """img – An image (as a numpy array)
        number_of_times_to_upsample – How many times to upsample the 
        image looking for faces. Higher numbers find smaller faces.
        model – Which face detection model to use. “hog” is less 
        accurate but faster on CPUs. “cnn” is a more accurate 
        deep-learning model which is GPU/CUDA accelerated (if 
        available). The default is “hog”.
        Returns: A list of tuples of found face locations in css (top,
        right, bottom, left) order"""
        boxes = face_recognition.face_locations(rgb,
                                                model=detection_method)

        # compute the facial embedding for the face
        # face_encodings()
        """face_image – The image that contains one or more faces
        known_face_locations – Optional - the bounding boxes of each
        face if you already know them.
        Returns: A list of 128-dimensional face encodings (one for
        each face in the image)"""
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings... to ", encodings_file)
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(encodings_file, "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("Total Images Encoded:", len(knownNames))
    print("Total Faces Encoded:", len(set(knownNames)))


def recognize_faces_from_image():
    try:
        while True:
            while True:
                img_name = input('Enter image file name(with extension), image must be in recognize_faces folder: ')
                img_path = 'recognize_faces/' + img_name
                if path.exists(img_path):
                    break
                else:
                    print('File Not Found!')

            encodings_file = '../Face_Liveness/encodings.pickle'
            detection_method = 'hog'

            # load the known faces and embeddings
            print("[INFO] loading encodings...")
            data = pickle.loads(open(encodings_file, "rb").read())
            # load the input image and convert it from BGR to RGB

            image = cv2.imread(img_path)
            image = cv2.resize(image, (600, 600))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # detect the (x, y)-coordinates of the bounding boxes corresponding
            # to each face in the input image, then compute the facial embeddings
            # for each face
            print("[INFO] recognizing faces...")
            boxes = face_recognition.face_locations(rgb,
                                                    model=detection_method)
            encodings = face_recognition.face_encodings(rgb, boxes)
            # initialize the list of names for each face detected
            names = []

            # loop over the facial embeddings
            for encoding in encodings:
                # attempt to match each face in the input image to our known
                # encodings
                matches = face_recognition.compare_faces(data["encodings"],
                                                         encoding)
                name = "Unknown"

                # check to see if we have found a match
                if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    # determine the recognized face with the largest number of
                    # votes (note: in the event of an unlikely tie Python will
                    # select first entry in the dictionary)
                    name = max(counts, key=counts.get)

                # update the list of names
                names.append(name)

            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # draw the predicted face name on the image
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
            # show the output image
            cv2.imshow("To test another photo press c, else q to quit", image)
            l = cv2.waitKey(0) & 0xFF
            if l == ord('q'):
                cv2.destroyAllWindows()
                break
    except:
        cv2.destroyAllWindows()
        print("Memory Error, Call to cuDNN failed")


def recognize_face_live_stream():
    # Initializing variables
    encodings = "encodings.pickle"
    display = 1
    detection_method = 'hog'

    # load the known faces and embeddings
    print("[INFO] loading encodings...")
    data = pickle.loads(open(encodings, "rb").read())

    # initialize the video stream and pointer to output video file, then
    # allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(0)

    # loop over frames from the video file stream
    try:
        while True:
            # grab the frame from the threaded video stream
            ret, frame = cap.read()

            # convert the input frame from BGR to RGB then resize it to have
            # a width of 750px (to speedup processing)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # resize with aspect ratio
            rgb = imutils.resize(frame, width=750)
            r = frame.shape[1] / float(rgb.shape[1])

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input frame, then compute
            # the facial embeddings for each face
            boxes = face_recognition.face_locations(rgb,
                                                    model=detection_method)
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            # loop over the facial embeddings
            for encoding in encodings:

                # attempt to match each face in the input image to our known encodings
                matches = face_recognition.compare_faces(data["encodings"],
                                                         encoding)
                name = "Unknown"

                # check to see if we have found a match
                if True in matches:

                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get)

                # update the list of names
                names.append(name)

            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # rescale the face coordinates
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom),
                              (0, 255, 0), 1)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)

            cv2.imshow("Press q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

                # do a bit of cleanup
        cv2.destroyAllWindows()
        cap.release()

    except:
        cv2.destroyAllWindows()
        cap.release()
        print("Memory Error, Call to cuDNN failed")

while True:
    print("\n\n****MENU****")
    t = int(input('Press 1 to add new face though images\n \
              Press 2 to add new face via live stream\n \
              Press 3 to recognize faces in images\n \
              Press 4 to recognize faces in live video stream \n \
              Press 5 to quit: \n'))
    if t==1:
        add_face_data_from_images()
        print('Please wait, updating the encoding files')
        encode_face_data()
    elif t==2:
        add_face_data_from_video()
        print('Please wait, updating the encoding files')
        encode_face_data()
    elif t==3:
        recognize_faces_from_image()
    elif t==4:
        recognize_face_live_stream()
    elif t==5:
        print('Thanks! Make yourself a good day!!')
        break
    else:
        print('Invalid Entry')