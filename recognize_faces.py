import cv2
import pandas as pd
import os
import time
import datetime
import numpy as np

def recognize_faces():
    print("Đang đọc file CSV...")
    df = pd.read_csv("StudentDetails/StudentDetails.csv")
    
    if 'Id' not in df.columns or 'Name' not in df.columns:
        if '01' in df.columns and 'pta' in df.columns:
            df = df.rename(columns={'01': 'Id', 'pta': 'Name'})
        elif 'ID' in df.columns:
            df = df.rename(columns={'ID': 'Id'})
        else:
            raise KeyError("Không tìm thấy cột 'Id' hoặc 'Name' trong DataFrame")

    df['Id'] = df['Id'].astype(str)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)    
    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    
    # Initialize attendance DataFrame with proper dtypes
    attendance = pd.DataFrame({
        'Id': pd.Series(dtype='str'),
        'Name': pd.Series(dtype='str'),
        'Date': pd.Series(dtype='str'),
        'Time': pd.Series(dtype='str'),
        'Confidence': pd.Series(dtype='float64')
    })
    
    recognized_faces = {}

    while True:
        ret, im = cam.read()
        if not ret:
            print("Không thể đọc được frame từ camera")
            continue
            
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)    
        
        current_frame_records = []
        frame_has_recognition = False

        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            
            if(conf < 70):  # Threshold for acceptable recognition
                frame_has_recognition = True
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                
                Id = str(Id)
                aa = df.loc[df['Id'] == Id]['Name'].values
                if len(aa) > 0:
                    name = aa[0]
                    tt = f"{Id}-{name} ({conf:.1f}%)"
                    
                    # Extract and store the face image
                    face_img = im[y:y+h, x:x+w].copy()
                    
                    current_frame_records.append({
                        'Id': Id,
                        'Name': name,
                        'Date': date,
                        'Time': timeStamp,
                        'Confidence': conf,
                        'Image': face_img
                    })
                else:
                    tt = f"{Id}-Unknown"
            else:
                tt = "Unknown"
                
            cv2.putText(im, str(tt), (x,y+h), font, 1, (255,255,255), 2)

        # Update attendance and store face images
        if current_frame_records:
            current_frame_records.sort(key=lambda x: x['Confidence'])
            
            for record in current_frame_records:
                Id = record['Id']
                existing_record = attendance[attendance['Id'] == Id]
                
                if existing_record.empty:
                    # Create a new DataFrame row with explicit types
                    new_record_df = pd.DataFrame({
                        'Id': [record['Id']],
                        'Name': [record['Name']],
                        'Date': [record['Date']],
                        'Time': [record['Time']],
                        'Confidence': [record['Confidence']]
                    })
                    attendance = pd.concat([attendance, new_record_df], ignore_index=True)
                    recognized_faces[Id] = record['Image']
                elif record['Confidence'] < existing_record['Confidence'].iloc[0]:
                    # Better confidence record
                    idx = existing_record.index[0]
                    attendance.loc[idx] = {
                        'Id': record['Id'],
                        'Name': record['Name'],
                        'Date': record['Date'],
                        'Time': record['Time'],
                        'Confidence': record['Confidence']
                    }
                    recognized_faces[Id] = record['Image']

        cv2.imshow('im', im) 
        if (cv2.waitKey(1) == ord('q')):
            break
            
    # Save attendance
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second = timeStamp.split(":")
    fileName = f"Attendance/Attendance_{date}_{Hour}-{Minute}-{Second}.csv"
    attendance.to_csv(fileName, index=False)
    
    cam.release()
    cv2.destroyAllWindows()
    return attendance, im
