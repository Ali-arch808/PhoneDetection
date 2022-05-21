import cv2
import sys
import os
import numpy as np

from keras.models import Sequential, Model
import efficientnet.keras as enet
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation

# import Face_D_T


FacecascPath = "haarcascade_frontalface_default.xml"
font = cv2.FONT_HERSHEY_SIMPLEX
############### changeble variables
Live = True
Activate_Eye_Detection = False

take_images = True
## in these four variables just one must be true


face_id = 2
image_count = 1

############### Initializing and loading Model

from keras.backend import sigmoid


class SwishActivation(Activation):

    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'


def swish_act(x, beta=1):
    return (x * sigmoid(beta * x))


from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

model = enet.EfficientNetB0(include_top=False, input_shape=(150, 50, 3), pooling='avg', weights='imagenet')

# Adding 2 fully-connected layers to B0.
x = model.output

x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation(swish_act)(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation(swish_act)(x)

x = Dense(64)(x)

x = Dense(32)(x)

x = Dense(16)(x)

# Output layer
predictions = Dense(1, activation="sigmoid")(x)

model_final = Model(inputs=model.input, outputs=predictions)

model_final.load_weights("PhoneDetection-CNN_weights_3_August.h5")

##################  Dfining predictor
from keras.preprocessing.image import ImageDataGenerator

path_right = "write_read_del_right/"
path_left = "write_read_del_left/"


def MyPrediction(test_dir):
    desired_batch_size = 1
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 50),
        color_mode="rgb",
        shuffle=False,
        class_mode='binary',
        batch_size=desired_batch_size)
    filenames = test_generator.filenames
    #     print(filenames)
    nb_samples = len(filenames)
    #     print(nb_samples)
    predict = model_final.predict_generator(test_generator, steps=np.ceil(nb_samples / desired_batch_size))
    #     prediction=predict[0][0]
    return predict[0][0]


################ changeble variables


no_object_detected = False
Initial_Face_Detection = False

Do_Template_Matching = False
Template_Matching_Max_Duration = 10
Template_Matching_Start_Time = 0

m_scale = 1

TICK_FREQUENCY = cv2.getTickFrequency()


class face_obj:
    x = 0
    y = 0
    width = 0
    height = 0


class face_pos:
    x = 0
    y = 0


facePose = face_pos()
minLoc = face_pos()


class frame_obj:
    x = 0
    y = 0
    width = 640
    height = 480


trackedFace = face_obj()
frameSize = frame_obj()
outputRect = face_obj()
faceRoi = face_obj()
facePos = face_pos()

FaceCascade = cv2.CascadeClassifier(FacecascPath)

if Live == True:
    video_capture = cv2.VideoCapture(0)
else:
    video_capture = cv2.VideoCapture(Video_Location)


def centerOfRect(rect):
    center = face_pos()
    center.x = rect.x + rect.width / 2
    center.y = rect.y + rect.height / 2
    return center


def facePosition(facePosition):
    facePos = face_pos();
    facePos.x = (int)(facePosition.x / m_scale);
    facePos.y = (int)(facePosition.y / m_scale);
    return facePos


def doubleRectSize(inputRect):
    outputRect.width = inputRect.width * 2
    outputRect.height = inputRect.height * 2

    # // Center rect around original center
    outputRect.x = inputRect.x - inputRect.width / 2
    outputRect.y = inputRect.y - inputRect.height / 2

    # // Handle edge cases
    if outputRect.x < frameSize.x:
        outputRect.width += outputRect.x
        outputRect.x = frameSize.x

    if outputRect.y < frameSize.y:
        outputRect.height += outputRect.y
        outputRect.y = frameSize.y

    if outputRect.x + outputRect.width > frameSize.width:
        outputRect.width = frameSize.width - outputRect.x

    if outputRect.y + outputRect.height > frameSize.height:
        outputRect.height = frameSize.height - outputRect.y

    return outputRect


def FaceTemplate(frame, face):
    # face.x += face.width / 4
    # face.y += face.height / 4
    # face.width /= 2
    # face.height /= 2
    faceTemplate = frame[int(face.y):int(face.y + face.height), int(face.x):int(face.x + face.width)]
    return faceTemplate


while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        # sleep(5)
        pass

    # Capture frame-by-frame
    rt, frame = video_capture.read()

    ############################### Robust Face Detection and Tracking : Start ####################################
    Faces = FaceCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(Faces) >= 1:  # if a face is detected
        ###############print("overal cascade")
        Faces = Faces[0]
        foundFace = True
        trackedFace.x = Faces[0]
        trackedFace.y = Faces[1]
        trackedFace.width = Faces[2]
        trackedFace.height = Faces[3]
        faceTemplate = FaceTemplate(frame, trackedFace)
        dimentions = faceTemplate.shape
        faceTemplate_rows = dimentions[0]
        faceTemplate_cols = dimentions[1]
        faceRoi = doubleRectSize(trackedFace)
        CenterOfRect = centerOfRect(trackedFace)
        facePos = facePosition(CenterOfRect)
        no_object_detected = False
        Initial_Face_Detection = True

    elif len(Faces) < 1 and Initial_Face_Detection == True:
        # Roi = frame [int(faceRoi.y):int(faceRoi.y+faceRoi.height), int(faceRoi.x):int(faceRoi.x+faceRoi.width)]
        Roi = frame[int(faceRoi.y):int(faceRoi.y + faceRoi.height), int(faceRoi.x):int(faceRoi.x + faceRoi.width)]
        ########################cv2.imshow('Roi', Roi)
        ########################print("Template Matching")
        if faceTemplate_rows * faceTemplate_cols == 0 or faceTemplate_rows <= 1 or faceTemplate_cols <= 1:
            foundFace = False
            Do_Template_Matching = False
            Template_Matching_Start_Time = Template_Matching_Current_Time = 0
            no_object_detected = True  #########################################3 this just a temp. must learn to clear faces variable ######################################

        Template_Matching_Rsult = cv2.matchTemplate(Roi, faceTemplate, cv2.TM_SQDIFF_NORMED)
        cv2.normalize(Template_Matching_Rsult, Template_Matching_Rsult, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        MinMaxLoc = cv2.minMaxLoc(Template_Matching_Rsult)
        # print(minLoc)
        minLoc = face_pos()

        minLoc.x = MinMaxLoc[2][0]
        minLoc.y = MinMaxLoc[2][1]
        minLoc.x += faceRoi.x
        minLoc.y += faceRoi.y
        trackedFace.x = minLoc.x
        trackedFace.y = minLoc.y
        trackedFace.width = faceTemplate_cols
        trackedFace.height = faceTemplate_rows
        # trackedFace = doubleRectSize(trackedFace)
        faceTemplate = FaceTemplate(frame, trackedFace)
        faceRoi = doubleRectSize(trackedFace)
        CenterOfRect = centerOfRect(trackedFace)
        facePos = facePosition(CenterOfRect)
        no_object_detected = False

    ############################### Robust Face Detection and Tracking : End ####################################

    ###################### save image : start   #######################
    right_ear = frame[int(trackedFace.y):int(trackedFace.y + trackedFace.height * 1.2),
                int(trackedFace.x - trackedFace.width / 5):int(trackedFace.x + trackedFace.width / 5)]
    left_ear = frame[int(trackedFace.y):int(trackedFace.y + trackedFace.height * 1.2),
               int(trackedFace.x + trackedFace.width * (4 / 5)):int(trackedFace.x + trackedFace.width * (6 / 5))]

    cv2.imwrite(path_right + "Right/Right.jpg", right_ear)
    cv2.imwrite(path_left + "left/left.jpg", left_ear)

    ##################  Classifying ######################     *****
    prediction_right = MyPrediction(path_right)
    prediction_left = MyPrediction(path_left)

    prediction = max(prediction_right, prediction_left)

    ###################### save image when the P is pressed: end   #######################

    ############################### Visualization : Start ############################################################
    if no_object_detected == False:
        cv2.circle(frame, (int(facePos.x), int(facePos.y)), 30, (0, 255, 0));
        # cv2.rectangle(frame, (int(trackedFace.x),                     int(trackedFace.y)), (int(trackedFace.x+trackedFace.width), int(trackedFace.y+trackedFace.height)), (0, 255, 0), 2)
        cv2.rectangle(frame, (int(trackedFace.x - trackedFace.width / 5), int(trackedFace.y)),
                      (int(trackedFace.x + trackedFace.width / 5), int(trackedFace.y + trackedFace.height * 1.2)),
                      (255, 0, 0), 2)
        cv2.rectangle(frame, (int(trackedFace.x + trackedFace.width * (4 / 5)), int(trackedFace.y)),
                      (int(trackedFace.x + trackedFace.width * (6 / 5)), int(trackedFace.y + trackedFace.height * 1.2)),
                      (255, 0, 0), 2)
    #       prediction=category[int(round(predict[i][0]))]
    #         cv2.putText(frame,'Phone:',prediction,(1,50), font, 1,(255,255,255),2)
    print(prediction)
    cv2.imshow('Video', frame)
    ############################### Visualization : End ############################################################
    os.remove(path_right + "Right/" + "Right.jpg")
    os.remove(path_left + "left/" + "left.jpg")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
