# Real-Time-Video-Processing

Clone the repository and install the requirement by:<br>
<code>pip install -r requirements.txt</code>

## Face_Recognition: Pre-Encoded Face Detection and Tracking
Change directory to Face_Recognition and depending on os run:<br>
For windows:<br>
<code>python Face_Recognition.py</code>  

For linux:<br>
<code>python3 Face_Recognition.py</code>  

Adding new face from images and also from live video. Recognizing faces from images and live video. If face is not pre-recorded then unknown is shown.<br>
face_recognition_dataset: dataset for creating face recognition encodings<br>
haarcascades: haarcascades file<br>
recognize_faces: directory to put images in which faces are to be recognized<br>
Add_New_Face.py: Functions to add new faces to existing encodings<br>
Face_Recognition.py: Menu driven programm to add new face from photos folder(the name of folder should be name of person and the folder should be located in face_recognition_dataset), add new face from live vide, recognize face from images(images should be in recognize_faces folder), and recognize face from webcam video.<br>


## Object_Detection
Change directory to Object_Detection and depending on os run:<br>
For windows:<br>
<code>python Object_Detection.py</code>  

For linux:<br>
<code>python3 Object_Detection.py</code>  

Object detection using pre-trained caffe model with classes: "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"<br>
Object_Detection.py: Detecting objects from webcam video<br>
caffe_models: pre-trained models used<br>


## Face_Liveness: Detecting wheteher a real face is there a a spoofed face(image of face)
Run the Live_Model_Demo.ipynb in jupyter<br>

caffe_models: pre-trained models for face_detection<br>
live_dataset: real and fake face images<br>
training_videos: Either use pre-recorded real and fake face videos, or from live video<br>
Live_Model_Demo.ipynb: Final demo for detecting real and fake faces<br>
Liveness_Data_Generation.py: Extracting faces from videos<br>
Liveness_Model.py: Model Architiecture for training<br>
Model_Train.ipynb: Training the model<br>

## Combined_Model_Demo.ipynb
Combined object detection, face recogniiton and face liveness detection in live video stream.<br>


## Refrences
https://www.pyimagesearch.com/start-here/
