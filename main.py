import cv2

def takeimage(name):
    print("\nTakign images for user : ",name)
    
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
            
            cv2.imwrite("TrainingImage\ "+name +'.'+ str(count) + ".jpg", gray[y:y+h,x:x+w])
            #display the frame
            cv2.imshow('frame',frame)
        #wait for 100 miliseconds 
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is morethan 100
        elif count>60:
            break
    cam.release()
    cv2.destroyAllWindows() 
    
    print("\nImage Taken")
        
        
takeimage("shubham")
