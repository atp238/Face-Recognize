import cv2
import os
import numpy as np
import pandas as pd
import time
import datetime
from PIL import Image
import dlib
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Helper functions
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def update_csv(Id, name):
    csv_file = 'StudentDetails/StudentDetails.csv'
    df = pd.read_csv(csv_file)
    if Id in df['Id'].values:
        df.loc[df['Id'] == Id, 'Name'] = name
    else:
        new_row = pd.DataFrame({'Id': [Id], 'Name': [name]})
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_file, index=False)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)
        except Exception as e:
            print(f"Lỗi khi đọc ảnh {imagePath}: {e}")
    return faces, Ids

def delete_user(Id):
    csv_file = 'StudentDetails/StudentDetails.csv'
    df = pd.read_csv(csv_file)

    # Chuyển đổi ID thành chuỗi và loại bỏ khoảng trắng
    Id = str(Id).strip()

    # Kiểm tra nếu ID có trong DataFrame
    if Id in df['Id'].astype(str).values:
        df = df[df['Id'] != Id]  # Xóa dòng có ID tương ứng
        df.to_csv(csv_file, index=False)
        return f"Đã xóa người dùng với ID: {Id}"
    else:
        return "Không tìm thấy ID trong danh sách."

# Face detection and recognition algorithms
def train_lbph(faces, labels):
    if len(faces) == 0 or len(labels) == 0:
        print("Không có dữ liệu để huấn luyện LBPH!")
        return None

    # Chuyển đổi faces thành grayscale nếu chưa phải
    gray_faces = []
    for face in faces:
        if len(face.shape) == 3:  # Nếu ảnh là màu
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face
        gray_faces.append(gray_face)

    # Đảm bảo tất cả các khuôn mặt có cùng kích thước
    target_size = (100, 100)  # Có thể điều chỉnh kích thước này
    resized_faces = [cv2.resize(face, target_size) for face in gray_faces]

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.train(resized_faces, np.array(labels))
        print("Huấn luyện LBPH thành công!")
        return recognizer
    except cv2.error as e:
        print(f"Lỗi khi huấn luyện LBPH: {e}")
        return None

def train_eigenfaces(faces, labels):
    pca = PCA(n_components=100)
    faces_pca = pca.fit_transform(np.array(faces).reshape(len(faces), -1))
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(faces_pca, np.array(labels))
    return recognizer, pca

def train_fisherfaces(faces, labels):
    lda = LDA(n_components=100)
    faces_lda = lda.fit_transform(np.array(faces).reshape(len(faces), -1), labels)
    recognizer = cv2.face.FisherFaceRecognizer_create()
    recognizer.train(faces_lda, np.array(labels))
    return recognizer, lda

def train_cnn(faces, labels):
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    
    face_descriptors = []
    for face in faces:
        dets = detector(face, 1)
        if len(dets) > 0:
            shape = sp(face, dets[0])
            face_descriptor = facerec.compute_face_descriptor(face, shape)
            face_descriptors.append(face_descriptor)
    
    return face_descriptors, labels

# Main functions
def capture_images(Id, name):
    if is_number(Id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        
        person_folder = os.path.join("TrainingImage", f"{name}_{Id}")
        os.makedirs(person_folder, exist_ok=True)
        
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)        
                sampleNum += 1
                img_path = os.path.join(person_folder, f"{name}.{Id}.{sampleNum}.jpg")
                cv2.imwrite(img_path, gray[y:y+h, x:x+w])
                cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        
        update_csv(Id, name)
        return f"Ảnh đã được lưu với ID : {Id} - Tên : {name} trong thư mục {person_folder}"
    else:
        return "Invalid input. Please enter a numeric ID and alphabetic name."

def train_model():
    print("Đang tải dữ liệu huấn luyện...")
    faces, labels = getImagesAndLabels("TrainingImage")
    
    if len(faces) == 0 or len(labels) == 0:
        print("Không có dữ liệu để huấn luyện. Vui lòng chụp ảnh trước khi huấn luyện.")
        return "Huấn luyện thất bại: Không có dữ liệu"
    
    print(f"Đã tải {len(faces)} khuôn mặt và {len(labels)} nhãn.")
    
    print("Đang huấn luyện mô hình LBPH...")
    lbph_model = train_lbph(faces, labels)
    if lbph_model is None:
        return "Huấn luyện thất bại: Lỗi khi huấn luyện LBPH"
    
    print("Đang huấn luyện mô hình Eigenfaces...")
    eigenfaces_model, pca = train_eigenfaces(faces, labels)
    
    print("Đang huấn luyện mô hình Fisherfaces...")
    fisherfaces_model, lda = train_fisherfaces(faces, labels)
    
    print("Đang huấn luyện mô hình CNN...")
    cnn_descriptors, cnn_labels = train_cnn(faces, labels)
    
    # Save models
    print("Đang lưu các mô hình...")
    lbph_model.save("TrainingImageLabel/lbph_model.yml")
    eigenfaces_model.save("TrainingImageLabel/eigenfaces_model.yml")
    fisherfaces_model.save("TrainingImageLabel/fisherfaces_model.yml")
    np.save("TrainingImageLabel/cnn_descriptors.npy", cnn_descriptors)
    np.save("TrainingImageLabel/cnn_labels.npy", cnn_labels)
    np.save("TrainingImageLabel/pca.npy", pca.components_)
    np.save("TrainingImageLabel/lda.npy", lda.coef_)
    
    return "Huấn luyện hoàn tất thành công"

def recognize_face(image, models):
    lbph_model, eigenfaces_model, fisherfaces_model, cnn_data, pca, lda = models
    
    # LBPH
    lbph_id, lbph_conf = lbph_model.predict(image)
    
    # Eigenfaces
    image_pca = pca.transform(image.reshape(1, -1))
    eigen_id, eigen_conf = eigenfaces_model.predict(image_pca)
    
    # Fisherfaces
    image_lda = lda.transform(image.reshape(1, -1))
    fisher_id, fisher_conf = fisherfaces_model.predict(image_lda)
    
    # CNN with dlib
    cnn_descriptors, cnn_labels = cnn_data
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    
    dets = detector(image, 1)
    if len(dets) > 0:
        shape = sp(image, dets[0])
        face_descriptor = facerec.compute_face_descriptor(image, shape)
        distances = [np.linalg.norm(np.array(face_descriptor) - np.array(desc)) for desc in cnn_descriptors]
        cnn_id = cnn_labels[np.argmin(distances)]
        cnn_conf = min(distances)
    else:
        cnn_id, cnn_conf = -1, float('inf')
    
    # Voting or choosing the best result
    all_ids = [lbph_id, eigen_id, fisher_id, cnn_id]
    all_confs = [lbph_conf, eigen_conf, fisher_conf, cnn_conf]
    
    best_id = max(set(all_ids), key=all_ids.count)
    best_conf = min(all_confs)
    
    return best_id, best_conf


def recognize_faces():
    print("Đang đọc file CSV...")
    df = pd.read_csv("StudentDetails/StudentDetails.csv")
    
    # Ensure correct column names
    if 'Id' not in df.columns or 'Name' not in df.columns:
        if '01' in df.columns and 'pta' in df.columns:
            df = df.rename(columns={'01': 'Id', 'pta': 'Name'})
        elif 'ID' in df.columns:
            df = df.rename(columns={'ID': 'Id'})
        else:
            raise KeyError("Không tìm thấy cột 'Id' hoặc 'Name' trong DataFrame")

    df['Id'] = df['Id'].astype(str)

    # Load models
    lbph_model = cv2.face.LBPHFaceRecognizer_create()
    lbph_model.read("TrainingImageLabel/lbph_model.yml")
    
    eigenfaces_model = cv2.face.EigenFaceRecognizer_create()
    eigenfaces_model.read("TrainingImageLabel/eigenfaces_model.yml")
    
    fisherfaces_model = cv2.face.FisherFaceRecognizer_create()
    fisherfaces_model.read("TrainingImageLabel/fisherfaces_model.yml")
    
    cnn_descriptors = np.load("TrainingImageLabel/cnn_descriptors.npy")
    cnn_labels = np.load("TrainingImageLabel/cnn_labels.npy")
    
    pca_components = np.load("TrainingImageLabel/pca.npy")
    lda_coef = np.load("TrainingImageLabel/lda.npy")
    
    pca = PCA(n_components=100)
    pca.components_ = pca_components
    
    lda = LDA(n_components=100)
    lda.coef_ = lda_coef
    
    models = (lbph_model, eigenfaces_model, fisherfaces_model, (cnn_descriptors, cnn_labels), pca, lda)
    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)    
    
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.2, 5)    
        
        for (x,y,w,h) in faces:
            cv2.rectangle(im, (x,y), (x+w,y+h), (225,0,0), 2)
            Id, conf = recognize_face(gray[y:y+h, x:x+w], models)
            
            if conf < 70:  # Increased threshold to 70
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
                attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
            else:
                Id = 'Unknown'                
                tt = str(Id)  
            
            if conf > 90:
                noOfFile = len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite(f"ImagesUnknown/Image{noOfFile}.jpg", im[y:y+h,x:x+w])            
            cv2.putText(im, str(tt), (x,y+h), font, 1, (255,255,255), 2)        
        
          
        
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')    
        cv2.imshow('im', im) 
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = f"Attendance/Attendance_{date}_{Hour}-{Minute}-{Second}.csv"
    attendance.to_csv(fileName, index=False)
    
    cam.release()
    cv2.destroyAllWindows()
    return attendance

if __name__ == "__main__":
    while True:
        print("\nHệ thống nhận diện khuôn mặt")
        print("1. Chụp ảnh")
        print("2. Huấn luyện mô hình")
        print("3. Nhận diện khuôn mặt")
        print("4. Xóa người dùng")
        print("5. Thoát")
        choice = input("Nhập lựa chọn của bạn (1-5): ")
        
        if choice == '1':
            Id = input("Enter ID: ")
            name = input("Enter Name: ")
            result = capture_images(Id, name)
            print(result)
        elif choice == '2':
            result = train_model()
            print(result)
        elif choice == '3':
            result = recognize_faces()
            print(result)
      
        elif choice == '4':
            Id = input("Nhập ID của người dùng cần xóa: ")
            result = delete_user(Id)  # Gọi hàm xóa với ID
            print(result)
        elif choice == '5':
            print("Exiting the program.")
            break
            print("Invalid choice. Please try again.")