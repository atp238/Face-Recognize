import tkinter as tk
from tkinter import messagebox
import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time

# Import functions from other files
from capture_images import capture_images
from train_model import train_model
from recognize_faces import recognize_faces

class FaceRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Hệ thống nhận diện khuôn mặt")
        self.window.geometry('1600x900')
        self.window.configure(background='DarkGrey')

        self.message = tk.Label(window, text="Hệ thống nhận diện khuôn mặt", bg="dark slate gray", fg="white", width=70, height=3, font=('times', 30,))
        self.message.place(x=200, y=20)

        self.lbl = tk.Label(window, text="Nhập ID", width=20, height=2, fg="white", bg="gray25", font=('times', 15, ' bold '))
        self.lbl.place(x=350, y=200)

        self.txt = tk.Entry(window, width=35,  font=('times', 15, ' bold '))
        self.txt.place(x=650, y=210)

        self.lbl2 = tk.Label(window, text="Nhập tên", width=20, fg="white", bg="gray25", height=2, font=('times', 15, ' bold '))
        self.lbl2.place(x=350, y=300)

        self.txt2 = tk.Entry(window, width=35,  font=('times', 15, ' bold '))
        self.txt2.place(x=650, y=315)

        self.lbl3 = tk.Label(window, text="Thông báo : ", width=20, fg="white", bg="gray25", height=2, font=('times', 15, ' bold underline '))
        self.lbl3.place(x=350, y=400)

        self.message = tk.Label(window, text="", bg="gray25", fg="white", width=35, height=2, font=('times', 15, ' bold '))
        self.message.place(x=650, y=400)

        self.lbl3 = tk.Label(window, text="Thông tin điểm danh : ", width=20, fg="white", bg="gray25", height=2, font=('times', 15, ' bold  underline'))
        self.lbl3.place(x=350, y=650)

        self.message2 = tk.Label(window, text="", fg="white", bg="gray25", width=35, height=2, font=('times', 15, ' bold '))
        self.message2.place(x=650, y=650)

        # Buttons
        self.clearButton = tk.Button(window, text="Xóa", command=self.clear, fg="steel blue", bg="OliveDrab1", width=20, height=1, activebackground="white", font=('times', 15, ' bold '))
        self.clearButton.place(x=1000, y=215)

        self.clearButton2 = tk.Button(window, text="Xóa", command=self.clear2, fg="steel blue", bg="OliveDrab1", width=20, height=1, activebackground="white", font=('times', 15, ' bold '))
        self.clearButton2.place(x=1000, y=312)

        self.takeImg = tk.Button(window, text="Chụp ảnh", command=self.TakeImages, fg="steel blue", bg="OliveDrab1", width=20, height=2, activebackground="white", font=('times', 15, ' bold '))
        self.takeImg.place(x=200, y=550)

        self.trainImg = tk.Button(window, text="Train ảnh", command=self.TrainImages, fg="steel blue", bg="OliveDrab1", width=20, height=2, activebackground="white", font=('times', 15, ' bold '))
        self.trainImg.place(x=500, y=550)

        self.trackImg = tk.Button(window, text="Nhận diện", command=self.TrackImages, fg="steel blue", bg="OliveDrab1", width=20, height=2, activebackground="white", font=('times', 15, ' bold '))
        self.trackImg.place(x=800, y=550)

        self.quitWindow = tk.Button(window, text="Thoát", command=self.window.destroy, fg="steel blue", bg="OliveDrab1", width=20, height=2, activebackground="white", font=('times', 15, ' bold '))
        self.quitWindow.place(x=1100, y=550)

    def clear(self):
        self.txt.delete(0, 'end')
        res = ""
        self.message.configure(text=res)

    def clear2(self):
        self.txt2.delete(0, 'end')
        res = ""
        self.message.configure(text=res)

    def TakeImages(self):
        Id = self.txt.get()
        name = self.txt2.get()
        res = capture_images(Id, name)
        self.message.configure(text=res)

    def TrainImages(self):
        res = train_model()
        self.message.configure(text=res)

    def TrackImages(self):
        try:
            res = recognize_faces()
            self.message2.configure(text=str(res))
        except Exception as e:
            self.message2.configure(text=f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()