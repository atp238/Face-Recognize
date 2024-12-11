import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time

# Import functions from other files
from capture_images import capture_images, delete_images
from train_model import train_model
from recognize_faces import recognize_faces as perform_recognition

class FaceRecognitionApp:
    def __init__(self, window):
        self.root = window
        self.root.title("Hệ thống nhận diện khuôn mặt")
        self.root.geometry('1200x800')
        self.root.configure(background='#f0f0f0')
        
        # Dictionary to store face images for each person
        self.face_images = {}
        
        # Initialize variables
        self.id_entry = None
        self.name_entry = None
        self.message_var = None
        self.result_tree = None
        self.image_label = None
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', font=('Helvetica', 14))
        
        # Create all the widgets
        self.create_widgets()

    def recognize_faces(self):
        try:
            # Clear previous results
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
            self.face_images.clear()

            # Perform face recognition
            attendance, frame_with_faces = perform_recognition()
            
            if attendance is not None and not attendance.empty:
                # Extract face regions for each recognized person
                gray = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                
                # Process each detected face, access row ,...
                for i, (x, y, w, h) in enumerate(faces):
                    if i < len(attendance):
                        # Extract the face region
                        face_img = frame_with_faces[y:y+h, x:x+w]
                        
                        # Store the face image with the corresponding ID
                        person_id = str(attendance.iloc[i]['Id'])
                        self.face_images[person_id] = face_img
                
                # Update tree view with attendance records
                for _, row in attendance.iterrows(): #iterate through each line in the DataFrame
                    confidence_str = f"{row['Confidence']:.1f}%"
                    self.result_tree.insert('', 'end', values=(
                        row['Id'],
                        row['Name'],
                        row['Time'],
                        confidence_str
                    ))
                
                # Display success message
                self.message_var.set(f"Đã nhận diện được {len(attendance)} người!")
                
                # Select first item to display initial image
                if self.result_tree.get_children():
                    first_item = self.result_tree.get_children()[0]
                    self.result_tree.selection_set(first_item)
                    self.on_tree_select(None)  # Trigger display of first image
            else:
                self.message_var.set("Không nhận diện được khuôn mặt nào!")

        except Exception as error:
            self.message_var.set(f"Lỗi: {str(error)}")
            if self.image_label:
                self.image_label.configure(image='')

    def on_tree_select(self, event):
        """Handle selection in the result tree"""
        
        try:
            selected_items = self.result_tree.selection()
            if not selected_items:
                return
        
            # Get the selected person's ID
            item = selected_items[0]
            person_id = str(self.result_tree.item(item)['values'][0])  #takeid
            
            # Display the corresponding face image
            if person_id in self.face_images:
                self.display_image(self.face_images[person_id])
                self.message_var.set(f"Hiển thị ảnh của ID: {person_id}")
            else:
                self.message_var.set(f"Không tìm thấy ảnh cho ID: {person_id}")
        except Exception as error:
            self.message_var.set(f"Lỗi khi hiển thị ảnh: {str(error)}")

    def display_image(self, image):
        """Display the given image in the image label"""
        if image is not None and self.image_label is not None:
            try:
                # Convert BGR to RGB if necessary
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                image = Image.fromarray(image)
                
                # Resize while maintaining aspect ratio
                display_size = (400, 300)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage and display
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo  # Keep a reference!
            except Exception as e:
                self.message_var.set(f"Lỗi hiển thị ảnh: {str(e)}")

    def create_widgets(self):
        # Header
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.pack(fill=tk.X, padx=20, pady=20)

        title_label = ttk.Label(header_frame, text="Hệ thống nhận diện khuôn mặt", 
                            font=('Helvetica', 35, 'bold'))
        title_label.pack(pady=10)
        
        ttk.Separator(self.root, orient='horizontal').pack(fill=tk.X, padx=20)

        # Main content
        content_frame = ttk.Frame(self.root, padding="20")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left column
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        ttk.Label(left_frame, text="Nhập thông tin", 
                  font=('Helvetica', 20, 'bold')).pack(pady=(0, 20))

        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill=tk.X, pady=10, padx=5)

        ttk.Label(input_frame, text="ID:", font=('Helvetica', 16)).grid(row=0, column=0, sticky=tk.W, pady=10, padx=10)
        self.id_entry = ttk.Entry(input_frame, width=30, font=('Helvetica', 14))
        self.id_entry.grid(row=0, column=1, pady=10, ipady=5, padx=10)

        ttk.Label(input_frame, text="Tên:", font=('Helvetica', 16)).grid(row=1, column=0, sticky=tk.W, pady=10, padx=10)
        self.name_entry = ttk.Entry(input_frame, width=30, font=('Helvetica', 14))
        self.name_entry.grid(row=1, column=1, pady=10, ipady=5, padx=10)

        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=20)

        ttk.Button(button_frame, text="Chụp ảnh", command=self.take_images).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
       
        # Messages
        ttk.Label(left_frame, text="Thông báo", 
                  font=('Helvetica', 18, 'bold')).pack(pady=(20, 10))
        
        message_frame = ttk.Frame(left_frame, relief="sunken", borderwidth=2)
        message_frame.pack(fill=tk.X, pady=10, padx=5)
        
        self.message_var = tk.StringVar()
        ttk.Label(message_frame, textvariable=self.message_var, 
                  font=('Helvetica', 14), wraplength=400, padding=10).pack(pady=10)


        # Right column
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(right_frame, text="Kết quả nhận diện", 
                  font=('Helvetica', 20, 'bold')).pack(pady=(0, 10))

        # Create Treeview with scrollbar
        tree_frame = ttk.Frame(right_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_tree = ttk.Treeview(tree_frame, 
                                      columns=('ID', 'Name', 'Time', 'Confidence'),
                                      show='headings',
                                      height=8,
                                      yscrollcommand=scrollbar.set)
        
        self.result_tree.heading('ID', text='ID')
        self.result_tree.heading('Name', text='Tên')
        self.result_tree.heading('Time', text='Thời gian')
        self.result_tree.heading('Confidence', text='Độ tin cậy')
        
        self.result_tree.column('ID', width=100)
        self.result_tree.column('Name', width=200)
        self.result_tree.column('Time', width=150)
        self.result_tree.column('Confidence', width=100)
        
        # Configure scrollbar
        scrollbar.config(command=self.result_tree.yview)
        
        # Pack the tree
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind selection event
        self.result_tree.bind('<<TreeviewSelect>>', self.on_tree_select)

        # Image display section
        image_frame = ttk.Frame(right_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(image_frame, text="Ảnh nhận diện", 
                 font=('Helvetica', 16, 'bold')).pack(pady=(0, 10))
        
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(pady=10)

        # Footer
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=20)
        
        ttk.Button(footer_frame, text="Thoát", command=self.root.destroy).pack(side=tk.RIGHT, padx=20)

    def take_images(self):
        id = self.id_entry.get()
        name = self.name_entry.get()
        if id and name:
            res = capture_images(id, name)
            self.message_var.set(res)
        else:
            self.message_var.set("Vui lòng nhập ID và tên")

    def train_images(self):
        res = train_model()
        self.message_var.set(res)

    def delete_images(self):
        id = self.id_entry.get()
        if id:
            res = delete_images(id)
            self.message_var.set(res)
        else:
            self.message_var.set("Vui lòng nhập ID để xóa ảnh")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
