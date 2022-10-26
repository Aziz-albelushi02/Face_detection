import cv2
from random import randrange

#Load some pre-trained data that had frontal face images from opencv repo ('haarcascade-frontalface-default.xml')
#it will help to detect frontal faces using webcam
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose to detect faces in 
#calling image read function from opencv repo 



# To capture video from webcam or even a recored MP4 that has faces
webcam = cv2.VideoCapture(1)
# note on cv2.VideoCapture(0) for frontal camera and (1) is for the selfie webcam



while True:
    
    #read current frame
    successful_frame_read, frame = webcam.read()
    
    
    #convert image into black && white (grayscale)
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    #Detecting faces 
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    print(face_coordinates)
    

    #Draw rectangles around the faces opencv takes an rectangle and draws it on the image the next one takes a tuple on the upper left hand coordiantes 
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(265), randrange(265),randrange(265)), 2)

    
    #displays the image when its running
    cv2.imshow('Clever Programmer Face Detector', frame)
    
    #helps to change frame per milsec
    cv2.waitKey(1)











    #print("test complete") # complete
