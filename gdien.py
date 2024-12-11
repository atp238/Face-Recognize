import tkinter as tk
from tkinter import ttk, messagebox
import os
import pandas as pd
from PIL import Image, ImageTk
import cv2
from datetime import datetime

class Dashboard:
    def __init__(self, window):
        self.root = window
        self.root.title("Hệ thống quản lý nhận diện khuôn mặt")
        self.root.geometry('1280x720')
        self.root.configure(background='#f0f0f0')
        
        # Setup style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', font=('Helvetica', 12))
        self.style.configure('Header.TLabel', font=('Helvetica', 24, 'bold'))
        self.style.configure('Stat.TLabel', font=('Helvetica', 20))
        
        self.create_dashboard()
        
    def create_dashboard(self):
        # Header
        header_frame = ttk.Frame(self.root, padding="20")
        header_frame.pack(fill=tk.X)
        
        ttk.Label(header_frame, text="Dashboard Quản Lý", style='Header.TLabel').pack()
        
        # Main content area with 2 columns
        content = ttk.Frame(self.root, padding="20")
        content.pack(fill=tk.BOTH, expand=True)
        
        # Left column - Statistics
        left_frame = ttk.LabelFrame(content, text="Thống kê", padding="20")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Statistics
        self.update_statistics(left_frame)
        
        # Right column - Quick Actions
        right_frame = ttk.LabelFrame(content, text="Chức năng chính", padding="20")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Quick action buttons
        self.create_action_buttons(right_frame)
        
        # Bottom frame for recent activity
        bottom_frame = ttk.LabelFrame(self.root, text="Hoạt động gần đây", padding="20")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.create_recent_activity_table(bottom_frame)
        
    def update_statistics(self, frame):
        # Total registered users
        total_users = self.get_total_users()
        ttk.Label(frame, text=f"Tổng số người dùng: {total_users}", style='Stat.TLabel').pack(pady=10, anchor=tk.W)
        
        # Total attendance records
        total_records = self.get_total_attendance_records()
        ttk.Label(frame, text=f"Tổng số lượt điểm danh: {total_records}", style='Stat.TLabel').pack(pady=10, anchor=tk.W)
        
        # Today's attendance
        today_attendance = self.get_today_attendance()
        ttk.Label(frame, text=f"Điểm danh hôm nay: {today_attendance}", style='Stat.TLabel').pack(pady=10, anchor=tk.W)
        
        # Total images
        total_images = self.get_total_images()
        ttk.Label(frame, text=f"Tổng số ảnh: {total_images}", style='Stat.TLabel').pack(pady=10, anchor=tk.W)
        
    def create_action_buttons(self, frame):
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid columns and rows with weights
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.rowconfigure(0, weight=1)
        button_frame.rowconfigure(1, weight=1)
        
        # Create large buttons with icons
        buttons = [
            ("Cập nhật thêm người dùng", self.open_user_management),
            ("Huấn luyện mô hình", self.open_face_recognition),
            ("Nhận diện khuôn mặt", self.open_reports),
            ("Xoá ảnh", self.open_settings),
        
            ("Thông báo", self.open_settings),
                        ("Thoát", self.open_settings),


        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = ttk.Button(button_frame, text=text, command=command)
            row = i // 2
            col = i % 2
            btn.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
            
    def create_recent_activity_table(self, frame):
        # Create Treeview
        columns = ('Thời gian', 'Hoạt động', 'Chi tiết')
        self.activity_tree = ttk.Treeview(frame, columns=columns, show='headings', height=6)
        
        # Configure columns
        for col in columns:
            self.activity_tree.heading(col, text=col)
            self.activity_tree.column(col, width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.activity_tree.yview)
        self.activity_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack elements
        self.activity_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add sample data
        self.update_recent_activities()
        
    # Utility methods
    def get_total_users(self):
        try:
            df = pd.read_csv('StudentDetails/StudentDetails.csv')
            return len(df)
        except:
            return 0
            
    def get_total_attendance_records(self):
        count = 0
        attendance_dir = 'Attendance'
        if os.path.exists(attendance_dir):
            for file in os.listdir(attendance_dir):
                if file.startswith('Attendance_') and file.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(attendance_dir, file))
                        count += len(df)
                    except:
                        continue
        return count
        
    def get_today_attendance(self):
        count = 0
        today = datetime.now().strftime('%Y-%m-%d')
        attendance_dir = 'Attendance'
        if os.path.exists(attendance_dir):
            for file in os.listdir(attendance_dir):
                if file.startswith(f'Attendance_{today}') and file.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(attendance_dir, file))
                        count += len(df)
                    except:
                        continue
        return count
        
    def get_total_images(self):
        count = 0
        training_dir = 'TrainingImage'
        if os.path.exists(training_dir):
            for root, dirs, files in os.walk(training_dir):
                count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        return count
        
    def update_recent_activities(self):
        # Clear existing items
        for item in self.activity_tree.get_children():
            self.activity_tree.delete(item)
            
        # Add recent attendance records
        attendance_dir = 'Attendance'
        recent_activities = []
        
        if os.path.exists(attendance_dir):
            for file in os.listdir(attendance_dir):
                if file.startswith('Attendance_') and file.endswith('.csv'):
                    file_path = os.path.join(attendance_dir, file)
                    file_time = os.path.getctime(file_path)
                    recent_activities.append((
                        datetime.fromtimestamp(file_time),
                        "Điểm danh",
                        f"File: {file}"
                    ))
                    
        # Sort by time and take most recent
        recent_activities.sort(reverse=True)
        for activity in recent_activities[:10]:  # Show only 10 most recent
            self.activity_tree.insert('', 'end', values=(
                activity[0].strftime('%Y-%m-%d %H:%M:%S'),
                activity[1],
                activity[2]
            ))
            
    # Button command methods
    def open_user_management(self):
        import main_gui
        window = tk.Toplevel(self.root)
        app = main_gui.FaceRecognitionApp(window)
        
    def open_face_recognition(self):
        import recognize_faces
        recognize_faces.recognize_faces()
        self.update_statistics(self.root)
        self.update_recent_activities()
        
    def open_reports(self):
        messagebox.showinfo("Báo cáo", "Chức năng báo cáo đang được phát triển!")
        
    def open_settings(self):
        messagebox.showinfo("Cài đặt", "Chức năng cài đặt đang được phát triển!")

if __name__ == "__main__":
    root = tk.Tk()
    app = Dashboard(root)
    root.mainloop()
