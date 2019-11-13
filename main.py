import os
from PIL import Image, ImageTk
import cv2
import numpy as np
from time import sleep
from flask import Flask ,redirect, url_for, request 

app = Flask(__name__) 

def takeimage(id):
    try:
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        cam=cv2.VideoCapture(0)
        cv2.namedWindow("CaptureImage" )
        
        count=0
        while True:
            ret , frame= cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = detector.detectMultiScale(gray, 1.3, 5)
           
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
                count =count +1
                
                cv2.imwrite("TrainingImage\ "+str(id) +''+ str(count) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
    #            str2="For Voter : " + id
                cv2.imshow("im",frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 60
            elif count>=60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        return 1
    except:
        return 2
    finally:
        cam.release()
        cv2.destroyAllWindows() 
    
    
def TrainImages():
    try:
        recognizer = cv2.face_LBPHFaceRecognizer.create()
        
        faces,Id = getImagesAndLabels("TrainingImage")
        
        recognizer.train(faces, np.array(Id))
        recognizer.save("TrainingImageLabel\Trainner.yml")
        return 1
    except:
        return 0
    
#    print("Image Trained")
    #+",".join(str(f) for f in Id)
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
#        print("\n",abc)
        Id=int(abc)
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids
        
def castVote():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
        recognizer.read("TrainingImageLabel\Trainner.yml")
        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath);    
       
        cam = cv2.VideoCapture(0)
          
        count=0
        while True:
            count=count+1
            if count>1000:
                return 0,0
            ret, im =cam.read()
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            faces=faceCascade.detectMultiScale(gray, 1.2,5) 
            
            conf=100
            for(x,y,w,h) in faces:
                cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
                Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
                print(Id," : ",conf)                                   
                
            if conf<45:
                    return Id,conf
    #                break 
                
            cv2.imshow('im',im)
            
            if (cv2.waitKey(1)==ord('q')):
                break
            
     
        cam.release()
        cv2.destroyAllWindows()
    except:
        return -1,-1
    finally:
        cam.release()
        cv2.destroyAllWindows()
    

    
     
#num=np.random.randint(low=100000,high=999999,size=[1,])
#takeimage(str(num[0]))   
#TrainImages()
#sleep(5)
#castVote()

@app.route("/takeimage",methods=['POST','GET']) 
def takeimg(): 
    vid = request.args.get('vid') 
    print("Taking images for voter id : ",vid)
    
    var=takeimage(vid)
#    var=1
    if var==1:
        return {'status':200}
    else :
        return {'status':204}
    

@app.route("/train",methods=['POST','GET']) 
def train(): 
    print("Training the images")

    var=TrainImages()
    if var==1:
        return {'status':200}
    else :
        return {'status':204}
    
@app.route("/cast",methods=['POST','GET']) 
def cast(): 
    print("casting the vote")
    vid = request.args.get('vid') 

    uid,conf=castVote()
    suid=str(uid)
#    print(suid)
    if suid[:6]==vid:
        return {'status':200}
    else :
        return {'status':204}
    



if __name__ == '__main__': 
       app.run(debug = True) 
