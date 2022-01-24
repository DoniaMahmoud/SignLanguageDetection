from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 200
WEIGHTS_FILE = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
# Prepare data generator for standardizing frames before sending them into the model.
data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
# Loading the model.
MODEL_NAME = 'asl_alphabet_9575.h5'
model = load_model(MODEL_NAME)
print("we are in the right dir")

app = Flask(__name__)


@app.route('/uploader', methods=[ 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename("A1.jpg"))
        print("i recived a response")
        IMAGE_SIZE = 200
        CROP_SIZE = 400

        # Creating list of available classes stored in classes.txt.
        classes_file = open("classes.txt")
        classes_string = classes_file.readline()
        classes = classes_string.split()
        classes.sort()  # The predict function sends out output in sorted order.
        frame = cv2.imread("A1.jpg")
        frame = cv2.rectangle(frame,(0,0),(200,200), (0,255,0),3)
        

        # Target area where the hand gestures should be.
       # cv2.rectangle(frame, (0, 0), (CROP_SIZE, CROP_SIZE), (0, 255, 0), 3)
        # Preprocessing the frame before input to the model.
        #cropped_image = frame[0:CROP_SIZE, 0:CROP_SIZE]
        resized_frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        cv2.imwrite("/app/downimg.jpg",reshaped_frame[0])
        frame_for_model = data_generator.standardize(np.float64(reshaped_frame))

        # Predicting the frame.
        prediction = np.array(model.predict(frame_for_model))
        predicted_class = classes[prediction.argmax()]  # Selecting the max confidence index.
        print(predicted_class)
        return predicted_class


if __name__ == '__main__':
    app.run(port=80,host='0.0.0.0', debug=False)

