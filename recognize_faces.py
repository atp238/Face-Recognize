import cv2
import pandas as pd
import os
import time
import datetime
import numpy as np

def recognize_faces():
    print("Đang đọc file CSV...")
    df = pd.read_csv("StudentDetails/StudentDetails.csv")
    print("Nội dung của DataFrame:")
    print(df)
    print("Các cột trong DataFrame:", df.columns)
    print("Kiểu dữ liệu của các cột:")
    print(df.dtypes)

    # Ensure correct column names
    if 'Id' not in df.columns or 'Name' not in df.columns:
        if '01' in df.columns and 'pta' in df.columns:
            df = df.rename(columns={'01': 'Id', 'pta': 'Name'})
        elif 'ID' in df.columns:
            df = df.rename(columns={'ID': 'Id'})
        else:
            raise KeyError("Không tìm thấy cột 'Id' hoặc 'Name' trong DataFrame")

    # Ensure 'Id' column is string type
    df['Id'] = df['Id'].astype(str)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)    
    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            print(f"Recognized ID: {Id}, Confidence: {conf}")  # Debug thông tin nhận dạng
            
            if(conf < 70):  # Đã tăng ngưỡng lên 70
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                
                Id = str(Id)
                aa = df.loc[df['Id'] == Id]['Name'].values
                if len(aa) > 0:
                    tt = f"{Id}-{aa[0]}"
                    name = aa[0]
                else:
                    tt = f"{Id}-Unknown"
                    name = 'Unknown'
                print(f"Matched Name: {name}")  # Debug thông tin tên đã khớp
                attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
            else:
                Id = 'Unknown'                
                tt = str(Id)  
            if(conf > 90):
                noOfFile = len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite(f"ImagesUnknown/Image{noOfFile}.jpg", im[y:y+h,x:x+w])            
            cv2.putText(im, str(tt), (x,y+h), font, 1, (255,255,255), 2)        
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName=f"Attendance/Attendance_{date}_{Hour}-{Minute}-{Second}.csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    return attendance

if __name__ == "__main__":
    result = recognize_faces()
    print(result)