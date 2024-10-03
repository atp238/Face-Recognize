import cv2
import os
import csv
import unicodedata

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
        res = f"Ảnh đã được lưu với ID : {Id} - Tên : {name} trong thư mục {person_folder}"
        
        # Ensure the CSV file exists with correct headers
        csv_file = 'StudentDetails/StudentDetails.csv'
        if not os.path.isfile(csv_file):
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Id', 'Name'])  # Write correct headers
        
        # Append the new data
        with open(csv_file, 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([Id, name])
        csvFile.close()
        return res
    else:
        if (is_number(Id)):
            return "Enter Alphabetical Name"
        if (name.isalpha()):
            return "Enter Numeric Id"

if __name__ == "__main__":
    Id = input("Enter ID: ")
    name = input("Enter Name: ")
    result = capture_images(Id, name)
    print(result)