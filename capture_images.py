import cv2
import os
import unicodedata
import pandas as pd

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def update_csv(Id, name):
    csv_file = 'StudentDetails/StudentDetails.csv'
    df = pd.read_csv(csv_file)
    
    # Check if Id already exists
    if Id in df['Id'].values:
        # Update the name if Id exists
        df.loc[df['Id'] == Id, 'Name'] = name
    else:
        # Append new row if Id doesn't exist
        new_row = pd.DataFrame({'Id': [Id], 'Name': [name]})
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Save the updated DataFrame back to CSV
    df.to_csv(csv_file, index=False)

def capture_images(Id, name):
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        # Create a subfolder for the person
        person_folder = os.path.join("TrainingImage", f"{name}_{Id}")
        os.makedirs(person_folder, exist_ok=True)

        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                sampleNum = sampleNum + 1
                # Save image in the person's subfolder
                img_path = os.path.join(person_folder, f"{name}.{Id}.{sampleNum}.jpg")
                cv2.imwrite(img_path, gray[y:y+h,x:x+w])
                cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        
        # Update CSV file
        update_csv(Id, name)
        
        res = f"Ảnh đã được lưu với ID : {Id} - Tên : {name} trong thư mục {person_folder}"
        return res
    else:
        if (is_number(Id)):
            return "Enter Alphabetical Name"
        if (name.isalpha()):
            return "Enter Numeric Id"

def delete_images(Id):
    # Find and delete the folder for the given Id
    for folder in os.listdir("TrainingImage"):
        if folder.endswith(f"_{Id}"):
            folder_path = os.path.join("TrainingImage", folder)
            for file in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, file))
            os.rmdir(folder_path)
            
            # Update CSV file
            csv_file = 'StudentDetails/StudentDetails.csv'
            df = pd.read_csv(csv_file)
            df = df[df['Id'] != int(Id)]
            df.to_csv(csv_file, index=False)
            
            return f"Đã xóa ảnh và thông tin của ID: {Id}"
    
    return f"Không tìm thấy ảnh cho ID: {Id}"

if __name__ == "__main__":
    Id = input("Enter ID: ")
    name = input("Enter Name: ")
    result = capture_images(Id, name)
    print(result)
