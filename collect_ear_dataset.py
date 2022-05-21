import cv2
import sys
import os
#import Face_D_T


FacecascPath = "haarcascade_frontalface_default.xml"

############### changeble variables
Live = True
Activate_Eye_Detection = False


take_images = True


## For every subject take_pos_images_right_Ear must be True
take_pos_images_right_Ear= True
take_pos_images_left_Ear= not take_pos_images_right_Ear
## in these four variables just one must be true


subject="Abbs.Alavi"
image_count = 1
make_dir=take_pos_images_right_Ear

if make_dir==True :
        os.mkdir('dataset/'+subject)
        os.mkdir('dataset/'+subject+'/p')
        os.mkdir('dataset/'+subject+'/n')
        
################ changeble variables




no_object_detected = False
Initial_Face_Detection = False


Do_Template_Matching = False
Template_Matching_Max_Duration = 10
Template_Matching_Start_Time = 0


m_scale = 1


TICK_FREQUENCY = cv2.getTickFrequency()




class face_obj:
        x=0
        y=0
        width=0
        height=0


class face_pos:
        x = 0
        y = 0

facePose = face_pos()
minLoc = face_pos()



class frame_obj:
    x=0
    y=0
    width=640
    height=480



trackedFace =  face_obj()
frameSize = frame_obj()
outputRect = face_obj()
faceRoi = face_obj()
facePos = face_pos ()



FaceCascade = cv2.CascadeClassifier(FacecascPath)

if Live == True :
    video_capture = cv2.VideoCapture(0)
else :
    video_capture = cv2.VideoCapture(Video_Location)





def centerOfRect (rect):
    center = face_pos()
    center.x=rect.x + rect.width / 2
    center.y=rect.y + rect.height / 2
    return center


def facePosition(facePosition):
    facePos = face_pos();
    facePos.x = (int)(facePosition.x / m_scale);
    facePos.y = (int)(facePosition.y / m_scale);
    return facePos


def doubleRectSize (inputRect):

    outputRect.width = inputRect.width * 2
    outputRect.height = inputRect.height * 2

#// Center rect around original center
    outputRect.x = inputRect.x - inputRect.width / 2
    outputRect.y = inputRect.y - inputRect.height / 2

#// Handle edge cases
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



def FaceTemplate (frame, face):
    #face.x += face.width / 4
    #face.y += face.height / 4
    #face.width /= 2
    #face.height /= 2
    faceTemplate = frame[int(face.y):int(face.y+face.height), int(face.x):int(face.x+face.width)]
    return faceTemplate



while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        #sleep(5)
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
        trackedFace.y= Faces[1]
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
        #Roi = frame [int(faceRoi.y):int(faceRoi.y+faceRoi.height), int(faceRoi.x):int(faceRoi.x+faceRoi.width)]
        Roi = frame [int(faceRoi.y):int(faceRoi.y+faceRoi.height), int(faceRoi.x):int(faceRoi.x+faceRoi.width)]
        ########################cv2.imshow('Roi', Roi)
        ########################print("Template Matching")
        if  faceTemplate_rows * faceTemplate_cols == 0 or faceTemplate_rows <= 1 or faceTemplate_cols <= 1:
            foundFace = False
            Do_Template_Matching = False
            Template_Matching_Start_Time = Template_Matching_Current_Time = 0
            no_object_detected = True  #########################################3 this just a temp. must learn to clear faces variable ######################################

        Template_Matching_Rsult = cv2.matchTemplate(Roi, faceTemplate, cv2.TM_SQDIFF_NORMED)
        cv2.normalize(Template_Matching_Rsult, Template_Matching_Rsult,  0, 1, cv2.NORM_MINMAX,cv2.CV_32F)
        MinMaxLoc = cv2.minMaxLoc(Template_Matching_Rsult)
            #print(minLoc)
        minLoc = face_pos()

        minLoc.x = MinMaxLoc[2][0]
        minLoc.y = MinMaxLoc[2][1]
        minLoc.x += faceRoi.x
        minLoc.y += faceRoi.y
        trackedFace.x = minLoc.x
        trackedFace.y = minLoc.y
        trackedFace.width = faceTemplate_cols
        trackedFace.height = faceTemplate_rows
            #trackedFace = doubleRectSize(trackedFace)
        faceTemplate = FaceTemplate(frame, trackedFace)
        faceRoi = doubleRectSize(trackedFace)
        CenterOfRect = centerOfRect(trackedFace)
        facePos = facePosition(CenterOfRect)
        no_object_detected = False
    ############################### Robust Face Detection and Tracking : End ####################################

###################### save image when the P is pressed: start   #######################
    right_ear = frame[int(trackedFace.y):int(trackedFace.y+trackedFace.height*1.2), int(trackedFace.x-trackedFace.width/5):int(trackedFace.x+trackedFace.width/5)]
    left_ear = frame[int(trackedFace.y):int(trackedFace.y+trackedFace.height*1.2), int(trackedFace.x+trackedFace.width*(4/5)):int(trackedFace.x+trackedFace.width*(6/5))]
    whole_frame = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if take_pos_images_right_Ear==True:
            if take_images == True:
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    if image_count < 10 :
                        cv2.imwrite("dataset/" + subject + '/p/' +subject + str(image_count) + "_p.jpg",right_ear)
                        cv2.imwrite("dataset/" + subject + '/n/' +subject + str(image_count) + "_n.jpg",left_ear)
                        image_count +=1

                    if  10 <= image_count < 100:
                        cv2.imwrite("dataset/" + subject + '/p/' +subject + str(image_count) + "_p.jpg", right_ear)
                        cv2.imwrite("dataset/" + subject + '/n/' +subject + str(image_count) + "_n.jpg",left_ear)
                        image_count +=1

                    if 100 <= image_count < 1000:
                        cv2.imwrite("dataset/" + subject + '/p/' +subject + str(image_count) + "_p.jpg", right_ear)
                        cv2.imwrite("dataset/" + subject + '/n/' +subject + str(image_count) + "_n.jpg",left_ear)
                        image_count +=1

    if take_pos_images_left_Ear== True:
            if take_images == True:
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    if image_count < 10 :
                        cv2.imwrite("dataset/" + subject + '/p/' +subject + str(image_count) + "_p.jpg",left_ear)
                        cv2.imwrite("dataset/" + subject + '/n/' +subject + str(image_count) + "_n.jpg", right_ear)
                        image_count +=1

                    if  10 <= image_count < 100:
                        cv2.imwrite("dataset/" + subject + '/p/' +subject + str(image_count) + "_p.jpg",left_ear)
                        cv2.imwrite("dataset/" + subject + '/n/' +subject + str(image_count) + "_n.jpg", right_ear)
                        image_count +=1

                    if 100 <= image_count < 1000:
                        cv2.imwrite("dataset/" + subject + '/p/' +subject + str(image_count) + "_p.jpg", left_ear)
                        cv2.imwrite("dataset/" + subject + '/n/' +subject + str(image_count) + "_n.jpg", right_ear)
                        image_count +=1



###################### save image when the P is pressed: end   #######################

############################### Visualization : Start ############################################################
    if no_object_detected == False:
        cv2.circle(frame, (int(facePos.x), int(facePos.y)), 30, (0, 255, 0));
        #cv2.rectangle(frame, (int(trackedFace.x),                     int(trackedFace.y)), (int(trackedFace.x+trackedFace.width), int(trackedFace.y+trackedFace.height)), (0, 255, 0), 2)
        cv2.rectangle(frame, (int(trackedFace.x-trackedFace.width/5), int(trackedFace.y)), (int(trackedFace.x+trackedFace.width/5), int(trackedFace.y+trackedFace.height*1.2)), (255, 0, 0), 2)
        cv2.rectangle(frame, (int(trackedFace.x+trackedFace.width*(4/5)), int(trackedFace.y)), (int(trackedFace.x+trackedFace.width*(6/5)), int(trackedFace.y+trackedFace.height*1.2)), (255, 0, 0), 2)
    cv2.imshow('Video', frame)
    ############################### Visualization : End ############################################################



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
