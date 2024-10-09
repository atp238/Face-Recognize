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
from recognize_faces import recognize_faces

class FaceRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Hệ thống nhận diện khuôn mặt")
        self.window.geometry('1200x800')
        self.window.configure(background='#f0f0f0')

        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Increase font size for all widgets
        self.style.configure('.', font=('Helvetica', 14))
        
        self.create_widgets()

    def create_widgets(self):
        # Header
        header_frame = ttk.Frame(self.window, padding="10", relief="raised", borderwidth=2)
        header_frame.pack(fill=tk.X, padx=20, pady=20)

        title_label = ttk.Label(header_frame, text="Hệ thống nhận diện khuôn mặt", 
                  font=('Helvetica', 35, 'bold'))
        title_label.pack(pady=10)
        
        # Add a separator
        ttk.Separator(self.window, orient='horizontal').pack(fill=tk.X, padx=20)

        # Main content
        content_frame = ttk.Frame(self.window, padding="20")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left column
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        ttk.Label(left_frame, text="Nhập thông tin", 
                  font=('Helvetica', 20, 'bold')).pack(pady=(0, 20))

        input_frame = ttk.Frame(left_frame, relief="raised", borderwidth=2)
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

        button_style = ttk.Style()
        button_style.configure('TButton', font=('Helvetica', 14), padding=10)

        ttk.Button(button_frame, text="Chụp ảnh", command=self.take_images).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="Train ảnh", command=self.train_images).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="Nhận diện", command=self.recognize_faces).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="Xóa ảnh", command=self.delete_images).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

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
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        ttk.Label(right_frame, text="Kết quả nhận diện", 
                    font=('Helvetica', 20, 'bold')).pack(pady=(0, 10))

        result_frame = ttk.Frame(right_frame, relief="sunken", borderwidth=2)
        result_frame.pack(pady=10, padx=5, fill=tk.BOTH, expand=False)

        self.result_text = tk.Text(result_frame, height=10, width=40, font=('Helvetica', 14))
        self.result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Footer
        footer_frame = ttk.Frame(self.window)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=20)

        ttk.Button(footer_frame, text="Thoát", command=self.window.destroy, style='TButton').pack(side=tk.RIGHT, padx=20)

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

    def recognize_faces(self):
        try:
            res = recognize_faces()
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, str(res))
        except Exception as e:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, f"Lỗi: {str(e)}")

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