import os
from PIL import Image, ImageTk
import cv2
import numpy as np
from time import sleep



def takeimage(id):
    print("\nTakign images for user : ",id)
    
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector=cv2.CascadeClassifier(harcascadePath)
    
    cam=cv2.VideoCapture(0)
    cv2.namedWindow("CaptureImage  : " )
    
    count=0
    while True:
        ret , frame= cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = detector.detectMultiScale(gray, 1.3, 5)
       
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
            count =count +1
            
            cv2.imwrite("TrainingImage\ "+id +''+ str(count) + ".jpg", gray[y:y+h,x:x+w])
            #display the frame
            cv2.imshow('frame',frame)
        #wait for 100 miliseconds 
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is morethan 100
        elif count>=60:
            break
    cam.release()
    cv2.destroyAllWindows() 
    
    print("\nImages Taken")
    
    
def TrainImages():
    print("Trainign Images")
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    #recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
#    harcascadePath = "haarcascade_frontalface_default.xml"
#    detector =cv2.CascadeClassifier(harcascadePath)
    
    faces,Id = getImagesAndLabels("TrainingImage")
    
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    print("Image Trained")#+",".join(str(f) for f in Id)
#    message.configure(text= res)

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    
#    print(imagePaths)
    
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        
        abc=os.path.split(imagePath)[-1].split(".")[0]
        print("\n",abc)
        Id=int(abc)
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids
        
def castVote():
    print("Casting Vote")
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
   
    cam = cv2.VideoCapture(0)
      
       
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5) 
        
        conf=100
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            print(Id," : ",conf)                                   
            
        if conf<45:
                break 
            
        cv2.imshow('im',im)
        
        if (cv2.waitKey(1)==ord('q')):
            break
        
 
    cam.release()
    cv2.destroyAllWindows()
    
    print("Vote casted")
     
num=np.random.randint(low=100000,high=999999,size=[1,])
#takeimage(str(num[0]))   
#TrainImages()
#sleep(5)
castVote()
