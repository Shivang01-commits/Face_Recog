from kivy.uix.settings import text_type
from kivymd.uix.datatables import MDDataTable
#from kivy.uix.filechooser import error
from kivymd.uix.backdrop.backdrop import MDBoxLayout
from kivymd.uix.banner.banner import MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.pickers.datepicker.datepicker import date
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivy.lang import Builder
import cv2
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen,NoTransition
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage
from kivy.clock import Clock
from threading import Thread
from kivy.metrics import dp
from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.metrics import dp
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.textfield import MDTextField
import threading
import platform
from kivy.uix.camera import Camera
import gc

from kivy.core.window import Window
#if platform != "android":
 #   Window.size = (360, 640)


def is_android():
    return platform.system() == "Linux" and "ANDROID_ARGUMENT" in os.environ

if is_android():
    from android.storage import app_storage_path
    from android.permissions import request_permissions, Permission
    from android.storage import primary_external_storage_path


def ask_android_permissions():
    request_permissions([
        Permission.CAMERA,
        Permission.WRITE_EXTERNAL_STORAGE,
        Permission.READ_EXTERNAL_STORAGE,
    ])

def get_base_path():
    if is_android():
        return app_storage_path()
    else:
        return os.path.join(os.getcwd(), "AttendanceAppFiles")

BASE_DIR = get_base_path()

# Ensure the base directory exists
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

def get_excel_path(semester, subject):
    return os.path.join(BASE_DIR, f"Attendance_{subject}_{semester}.csv")

def get_name_mapping_path(semester):
    return os.path.join(BASE_DIR, f"name_mapping_{semester}.csv")

def get_label_file_path():
    return os.path.join(BASE_DIR, "label.txt")

def get_trained_model_path():
    return os.path.join(BASE_DIR, "trained_model.yml")
    
def get_dataset_path():
    if not os.path.exists(os.path.join(BASE_DIR, "dataset")):
       os.makedirs(os.path.join(BASE_DIR, "dataset"))
    return os.path.join(BASE_DIR, "dataset")

def export_attendance_to_excel(df,attendancefile):
    path = os.path.join(primary_external_storage_path(),attendancefile)
    df.to_excel(path, index=False)

    print(f"Attendance exported to {path}")

def get_model_path(filename):
    if platform == "android":
        from android.storage import resource_path
        base_path = os.environ.get('ANDROID_ARGUMENT')
        return os.path.join(base_path, "OpencvModels", filename)
    else:
        base_path = os.getcwd()
        return os.path.join(base_path, "OpencvModels", filename)


#loading Haar Cascade For FAce Detection
haar_cascade = cv2.CascadeClassifier(get_model_path("haarcascade_frontalface_alt.xml"))



print("App is starting...")
if not os.path.exists(get_label_file_path()):
   with open(get_label_file_path(), "w") as file:
       file.write("0")


#Home Page
attendance_file=None  #global variable for attendance file
name_mapping=None   #global name-mapping file
dialog_check=False
ui_check=False

class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #strong refrence to prevent garbage collector to delete refrences 
        self.original_widgets=list(self.ids.home_layout.children)
        self.dialog=None
    def on_enter(self):
        print("Entered Home Screen")
        Clock.schedule_once(self.setup_home_ui, 0)  # Delay execution
                
    def show_subject_semester_dialog(self,*args):
        self.subject_input=MDTextField(hint_text="Enter Subject")
        self.semester_input=MDTextField(hint_text="Enter Semester in number",input_filter="int",max_text_length=1)
 
        self.dialog=MDDialog(
            title="Enter Subject and Semester",
            type="custom",
            auto_dismiss=False,
            content_cls=BoxLayout(orientation='vertical',spacing=10,padding=10,size_hint_y=None,height=200),
            buttons=[
                MDFlatButton(text="OK",on_release=self.save_subject_semester)
            ]
        )
        
        self.dialog.content_cls.add_widget(self.subject_input)
        self.dialog.content_cls.add_widget(self.semester_input)
        self.dialog.open()
        
    def show_subject_semester_dialog_repeat(self,*args):
        self.dialog.dismiss()
        self.dialog=None
        self.subject_input=MDTextField(hint_text="Enter Subject")
        self.semester_input=MDTextField(hint_text="Enter Semester in number",input_filter="int")
 
        self.dialog=MDDialog(
            title="Enter Subject and Semester",
            type="custom",
            auto_dismiss=False,
            content_cls=BoxLayout(orientation='vertical',spacing=10,padding=10,size_hint_y=None,height=200),
            buttons=[
                MDFlatButton(text="OK",on_release=self.save_subject_semester)
            ]
        )
        
        self.dialog.content_cls.add_widget(self.subject_input)
        self.dialog.content_cls.add_widget(self.semester_input)
        self.dialog.open()
        
    def setup_home_ui(self,dt):
        global dialog_check
        global ui_check
        self.title_label = MDLabel(
            text="Face Recognition Attendance System",
            font_style="H5",
            halign="center",
            theme_text_color="Primary"
        )

        if dialog_check==False:
           Clock.schedule_once(self.show_subject_semester_dialog,1)
         
        if ui_check==True:
                    home_layout = self.ids.get("home_layout")  # Getting the layout using ID
                    print("HomeScreen IDs:", self.ids.keys())  # Debugging line
                    home_layout.clear_widgets()  # Remove old widget
                    #home_layout.add_widget(self.title_label)
                    #home_layout.add_widget(self.ids.train_button)
                    #home_layout.add_widget(self.ids.recognize_button)
                    #home_layout.add_widget(self.ids.view_button)
                    #adding widgets in homelayout from original widget list again as it is deleted by garbage collect of kivy 

                    for widget in reversed(self.original_widgets):
                        self.ids.home_layout.add_widget(widget)

        self.subject=None
        self.semester=None
         

    def close_dialog(self,instance):
        self.dialog.dismiss()
        self.dialog=None
     
    def errorMessage(self):
        self.dialog.dismiss()
        self.dialog=None
        self.dialog=MDDialog(
            title="Error",
            type="custom",
            auto_dismiss=False,
            content_cls=BoxLayout(orientation='vertical',spacing=10,padding=10,size_hint_y=None,height=200),
            buttons=[
                MDFlatButton(text="OK",on_release=self.show_subject_semester_dialog_repeat)
            ]
        )
        self.dialog.open()


    def save_subject_semester(self,instance):
        global dialog_check
        if not self.subject_input.text.strip():
            self.errorMessage()
            return
        self.subject=self.subject_input.text.strip()
        if not self.semester_input.text.strip():
            self.errorMessage()
            return
        if int(self.semester_input.text.strip())<0 or int(self.semester_input.text.strip())>8 or int(self.semester_input.text.strip())==0:
            self.errorMessage()
            return
        self.semester=self.semester_input.text.strip()
        dialog_check=True
        self.dialog.dismiss()
        if self.subject and self.semester:
            self.setup_attendance_file()

    #setups both atendance for sem and subject and mapping file for sem
    def setup_attendance_file(self):
        global attendance_file
        global name_mapping 
       
        #if no attendance.csv file is there
        if not os.path.exists(get_excel_path(self.semester, self.subject)):
            with open(get_excel_path(self.semester, self.subject),"w",newline="") as file:
                writer= csv.writer(file)
                writer.writerow(["Name","Date","Time"]) #Column headers 
            attendance_file=get_excel_path(self.semester, self.subject)
            print(f"Made new attendance file:{attendance_file}")     
        else:
            attendance_file=get_excel_path(self.semester, self.subject)
            print(f"Using attendance file:{attendance_file}")    

        if not os.path.exists(get_name_mapping_path(self.semester)):
            with open(get_name_mapping_path(self.semester),"w",newline="") as file:
                writer= csv.writer(file)
                writer.writerow(["id","Name"]) #Column headers 
            name_mapping=get_name_mapping_path(self.semester)
            print(f"Made new mapping file:{name_mapping}")     
        else:
            name_mapping=get_name_mapping_path(self.semester)
            print(f"Using mapping file:{name_mapping}")  


    def switch_screen(self, screen_name):
        global ui_check
        ui_check=True 
        home_layout = self.ids.home_layout
        home_layout.clear_widgets() 
        self.manager.current=screen_name

#------------------------------------------------------------------------------------------------------------------------------------------------

#Train Model Page 
from kivy.uix.textinput import TextInput 
import time
Builder.load_file("train_model_screen.kv")
class TrainModelScreen(Screen):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        #initialize opencv camera
        self.capture=None 
        self.camera=None
        self.face_cascade=cv2.CascadeClassifier(get_model_path("haarcascade_frontalface_alt.xml"))
        self.capturedFrame=None

    def on_enter(self):
        self.kivycam=None
        self.start_camera()
        

    def start_camera(self):
        try:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                raise Exception("OpenCV camera failed")
            self.use_opencv = True
            print("Using OpenCv camera")
            Clock.schedule_interval(self.update_video, 1.0 / 30)
        except:
            self.use_opencv = False
            self.start_kivy_camera()


    def start_kivy_camera(self):
        from kivy.uix.camera import Camera
        self.camera = Camera(play=True)
        self.camera.resolution = (640, 480)
        self.ids.card.clear_widgets()
        self.ids.card.add_widget(self.camera)
        self.kivycam=1
        print("Using Kivy Camera")
        print(self.kivycam)
        
    def update_video(self,*args):
        """Continously updating the feed"""
        if self.capture and self.capture.isOpened():
            ret, frame=self.capture.read()
            if ret:
                frame=cv2.flip(frame,0)
                buf=frame.tobytes()
                image_texture=Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
                image_texture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
                self.ids.camera_view.texture=image_texture
            
    def update_status(self, text):
       self.ids.status_label.text = text
    
   
    def _capture_on_main_thread(self, event):
       if not self.camera.texture:
           print("No texture available yet")
           self.capturedFrame=None
       else:
          size = self.camera.texture.size
          pixels = self.camera.texture.pixels

          frame = np.frombuffer(pixels, np.uint8).reshape(size[1], size[0], 4)  # RGBA
          frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
          self.capturedFrame=frame
       event.set()
    
    #this function below is actually collecting faces from images (50) and store it in dataset
    def _train_model_(self):
        threading.Thread(target=self.train_model,daemon=True).start()
    
    def train_model(self):

        name=self.ids.name_input.text.strip()
        if not name:
            print("Please enter a name before training.")
            return
        if not self.kivycam:
           if not self.capture or not self.capture.isOpened():
               print("Camera not started")
               return

        face_samples=[]
        count=0
        while count<50:
            if not(self.kivycam): #using opencv camera
               self.capturedFrame=None
               ret, self.capturedFrame=self.capture.read()   
               if not ret:
                   continue
            else:
                if self.camera.texture:
                    self.capturedFrame = None  # Reset
                    event = threading.Event()
                    Clock.schedule_once(lambda dt:self._capture_on_main_thread(event), 0)
                    event.wait(timeout=1.0)

                else:
                    continue
            gray=cv2.cvtColor(self.capturedFrame,cv2.COLOR_BGR2GRAY)
            faces=self.face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(50,50))

            for(x,y,w,h) in faces:
                face_img = gray[y:y + h, x:x +w] 
                file_path=os.path.join(get_dataset_path(),f"{name}_{count}.jpg")
                cv2.imwrite(file_path,face_img)
                face_samples.append(face_img)
                count+=1       
                print(f"Captured {count} images")
                #self.ids.status_label.text=f"Captured {count} images"
                Clock.schedule_once(lambda dt: self.update_status(f"Captured {count} images"), 0)

            time.sleep(0.2)      
        self.train_face_recognizer() 
        
         
    
    def dataset_deletion(self):
        dataset_path=get_dataset_path()
        if os.path.exists(dataset_path):
            for file in os.listdir(dataset_path):
                file_path=os.path.join(dataset_path,file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("Dataset emptied")       

    def train_face_recognizer(self):
        global name_mapping

        with open(get_label_file_path(), "r") as file:
           content = file.read()
           numbers = [int(num) for num in content.split() if num.isdigit()]
        Last_Label=numbers[0]

        #checking if dataset folder exists
        if os.path.exists(get_dataset_path()):
            dataset_path=get_dataset_path()
        else:
            print("Dataset folder not found")    

        #checking if name_mapping.csv exists
        if os.path.exists(name_mapping):
            print("mapping csv found")
            mapping_file=name_mapping
        else:
            print("name_mapping not found") 
        
        #if no old model exists we will train  it from scratch
        if not os.path.exists(get_trained_model_path()):
            recognizer=cv2.face.LBPHFaceRecognizer_create()
            face_samples=[]
            labels=[]
            label_mapping={}
            current_label=0
            Last_Label=0
            for file in os.listdir(dataset_path):
             if file.endswith(".jpg"):
                name=file.split("_")[0] #extract person name
                img_path=os.path.join(dataset_path,file)
                img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

                face_samples.append(img)
                if name not in label_mapping:
                    label_mapping[name]=current_label
                    current_label+=1
                labels.append(label_mapping[name])
            if len(face_samples)==0:
              print("No face found in dataset")
              return 
            
            Last_Label=current_label-1  #setting last label for future
            with open(get_label_file_path(), "w") as file:
                file.write(str(Last_Label))
            print(f"Last Label Used:{Last_Label}")                


            #Training Model
            recognizer.train(face_samples,np.array(labels))
            recognizer.save(get_trained_model_path())
            print(f"Training complete")
           # self.ids.status_label.text=f"Training Completed"
            Clock.schedule_once(lambda dt: self.update_status("Training Completed"), 0)
            #writing in name_mapping.csv file
            with open(mapping_file, "w", newline="") as f:
              writer=csv.writer(f)
              writer.writerow(["id","Name"])  #Header row
              for name, label in label_mapping.items():
                  writer.writerow([label,name])

            print(f"Labels saved in {mapping_file}")  
            self.dataset_deletion()
     
        else:#Old model already exists
            
            recognizer=cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(get_trained_model_path())  

            #setting new current_label
            #if os.path.exists("name_mapping.csv"):
             # df=pd.read_csv("name_mapping.csv")
              #existing_label_mapping=dict(zip(df["id"],df["Name"]))
              #current_label=(max(existing_label_mapping.keys())+1) #gets last user id 
            #else:
             # current_label=0
            if Last_Label is not None:
               current_label=(Last_Label+1) 
            else:
                print("ERROR-Last label is none while training old model")

            #collecting samples of new person
            face_samples=[]
            labels=[]
            label_mapping={}  

            for file in os.listdir(dataset_path):
                if file.endswith(".jpg"):
                    name=file.split("_")[0] #extract person name
                    img_path=os.path.join(dataset_path,file)
                    img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

                    face_samples.append(img)
                if name not in label_mapping:
                    label_mapping[name]=current_label
                    current_label+=1

                labels.append(label_mapping[name])
                if len(face_samples)==0:
                    print("No face found in dataset")
                    return 
                
            Last_Label=current_label-1    #setting Last Label for Future
            print(f"Last Label Used:{Last_Label}")
            with open(get_label_file_path(), "w") as file:
                file.write(str(Last_Label))            

            #Training Model
            recognizer.update(face_samples,np.array(labels))
            recognizer.save(get_trained_model_path())
            print("Model updated successfully")
            #self.ids.status_label.text=f"Model Updated Successfully"
            Clock.schedule_once(lambda dt: self.update_status("Model Successfully Updated"), 0)

            #writing in name_mapping.csv file
            with open(mapping_file, "a", newline="") as f:
                writer=csv.writer(f)

                for name, label in label_mapping.items():
                    writer.writerow([label,name])
                print(f"Labels saved in {mapping_file}") 
                self.dataset_deletion()   

    def switch_screen(self, screen_name):
        self.manager.current=screen_name   

    def stop_camera(self):
        if not self.kivycam:
           if self.capture and self.capture.isOpened():
               self.capture.release()
               self.capture=None
               print("Camera Released")  
        else:
            if hasattr(self, 'camera') and self.camera:
              self.camera.play = False
              self.ids.card.remove_widget(self.camera) 
              self.camera = None  
              print("Kivy Camera Released")  
              gc.collect()

    def on_leave(self):
        self.stop_camera()   
        


#--------------------------------------------------------------------------------------------------------------------------------------------------
#Recognition & Attendance Page
Builder.load_file("recognition_screen.kv")
class RecognitionScreen(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.capture=None #camera will be inialized when screen is entered
        self.capturedFrame=None
        #loading model
        #haar cascade is not detecting faces properly we will use deep learning
        #self.face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        prototxt = get_model_path("deploy.prototxt")
        weights = get_model_path("res10_300x300_ssd_iter_140000.caffemodel")
        self.net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        self.kivycam=0

    def on_enter(self):
        """Start camera feed when entering the screen"""
        self.name_map=self.load_name_mapping()  #loading CSV once ar startup of screen
       
        try:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                raise Exception("OpenCV camera failed")
            self.use_opencv = True
            print("Using OpenCv camera")
            Clock.schedule_interval(self.update_video, 1.0 / 30)
        except:
            self.use_opencv = False
            self.start_kivy_camera()
        
    def start_kivy_camera(self):
        from kivy.uix.camera import Camera
        self.camera = Camera(play=True)
        self.camera.resolution = (640, 480)
      
        self.kivycam=1
        print("Using Kivy Camera")
        print(self.kivycam)
        Clock.schedule_interval(self.update_video_for_kivyCam, 1.0 / 30)

    def mark_attendance(self,name):
        global attendance_file
        date_str= datetime.now().strftime("%Y-%m-%d")
        time_str= datetime.now().strftime("%H:%M:%S")
    
        already_marked = False
        if os.path.exists(attendance_file):
            with open(attendance_file, "r") as file:
                reader= csv.reader(file)
                for row in reader:
                    if row and row[0] == name and row[1] == date_str:
                        already_marked= True
                        break
        if not already_marked:
            with open(attendance_file,"a",newline="") as file:
                writer= csv.writer(file)
                writer.writerow([name,date_str,time_str])
                print(f"Attendance marked for {name} at {time_str}")


    def detect_faces_dnn(self,frame):
        h,w=frame.shape[:2]
        blob=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
         
        self.net.setInput(blob)
        detections=self.net.forward()
        faces=[]  #storing face coordinates
        

        for i in range(detections.shape[2]):
            confidence= detections[0,0,i,2]
            if confidence>0.4:
              
                box= detections[0,0,i,3:7]* np.array([w,h,w,h])
                (x,y,x1,y1)=box.astype("int")

                faces.append((x,y,x1-x,y1-y)) #(x,y,width,height)

                cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2) #drawing rectangle
       

        return frame,faces #returning frame and detected face coordinates         
 
    #function for loading name_mapping in a dictionary  since then recognizer will not have to open csv again and again 
    def load_name_mapping(self):
        global name_mapping
        name_map={}
        if os.path.exists(name_mapping):
            with open(name_mapping,"r") as file:
                reader=csv.reader(file)
                next(reader,None) #skipping first row

                for row in reader:
                    if len(row)>=2:
                        name_map[int(row[0])]=row[1] #convert label to int   
        return name_map      

    def recognize_faces(self,frame,faces):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        recognized_faces= [] 
        
        face_recognizer=cv2.face.LBPHFaceRecognizer_create()

        #we were detecting face again using haar cascade
        #faces= haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))

        if not os.path.exists(get_trained_model_path()):
            print("No trained model found. Skipping recognition")
            return recognized_faces  #returning empty list
        
        face_recognizer.read(get_trained_model_path())

        for(x,y,w,h) in faces:
            face_roi= gray[y:y+h, x:x+w]   #extracts face region
            label,confidence= face_recognizer.predict(face_roi)       

            if confidence<60:
                name=self.name_map.get(label,"Unknown") 
                if not name=="Unknown":
                  self.mark_attendance(name) #marking attendance
            else:
                name="Unknown"

            recognized_faces.append((x,y,w,h,name))
        
        return recognized_faces            

    def update_video_for_kivyCam(self,dt):
        if not self.camera.texture:
           print("Kivy camera didnt started")
           return 
        
        size = self.camera.texture.size
        pixels = self.camera.texture.pixels

        frame = np.frombuffer(pixels, np.uint8).reshape(size[1], size[0], 4)  # RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
   
        #Detects Face
        Frame,faces=self.detect_faces_dnn(frame) #calling DNN face Detection
        
        #Recognize Face
        recognized_faces= self.recognize_faces(Frame,faces)
        #Drawing rectangle&names
        for (x,y,w,h,name) in recognized_faces:
                    cv2.rectangle(Frame,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(Frame,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
              
        Frame=cv2.flip(Frame,0) #Flip frame for correct orientation
        buf= Frame.tobytes() #convert frame to bytes
                
        image_texture =Texture.create(size=(Frame.shape[1],Frame.shape[0]),colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr',bufferfmt='ubyte')
        if not image_texture:
            print("Waiting for texture")
            return
        self.ids.camera_view.texture=image_texture
        print("Frame shape:", Frame.shape)
        print("Texture size:", image_texture.size)
        print("camera_view widget:", self.ids.camera_view)
        print("Labelled Image is shown")
    
    def update_video(self,dt):
     
        """ Continously updates the video feed"""
        if self.capture is not None:
            ret, frame= self.capture.read()
            
            if ret:
               
               #Detects Face
                Frame,faces=self.detect_faces_dnn(frame) #calling DNN face Detection

                #Recognize Face
                recognized_faces= self.recognize_faces(Frame,faces)
                
                #Drawing rectangle&names
                for (x,y,w,h,name) in recognized_faces:
                    cv2.rectangle(Frame,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(Frame,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
              
                Frame=cv2.flip(Frame,0) #Flip frame for correct orientation
                buf= Frame.tobytes() #convert frame to bytes
                
                image_texture =Texture.create(size=(Frame.shape[1],Frame.shape[0]),colorfmt='bgr')
                image_texture.blit_buffer(buf, colorfmt='bgr',bufferfmt='ubyte')
                self.ids.camera_view.texture=image_texture

    def on_leave(self):
        """Stop camera feed when leavinvg screen""" 

        Clock.unschedule(self.update_video) #stop updating video feed
        Clock.unschedule(self.update_video_for_kivyCam)
        if not self.kivycam:
           if self.capture is not None:
               self.capture.release() #Release camera resource
               self.capture= None #Reset catpture
        else:
            if hasattr(self, 'camera') and self.camera:
              self.camera.play = False
              self.ids.card.remove_widget(self.camera) 
              self.camera = None  
              print("Kivy Camera Released")  
              gc.collect()
 
               
    def switch_screen(self, screen_name):
        self.manager.current=screen_name


#--------------------------------------------------------------------------------------------------------------------------------------------------
     
#Attendance Sheet Page 

Builder.load_file("attendance_screen.kv")
class AttendanceScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dialog=None
        self.Date=datetime.now().strftime("%Y-%m-%d")
        self.table=None

    def on_pre_enter(self):
        today=datetime.now().strftime("%Y-%m-%d")
        self.load_attendance_table(today)

    def load_attendance_table(self,date):
        global attendance_file
        self.ids.table_box.clear_widgets()

        if os.path.exists(attendance_file):
            df=pd.read_csv(attendance_file)
            filtered_df=df[df['Date']==date].copy()
          #  print(filtered_df)                        #debug

            if not filtered_df.empty:
                self.Date=date
                formatted_date=datetime.strptime(date,"%Y-%m-%d").strftime("%d %B %Y")
                self.ids.datelabel.text=(f"Attendance for {formatted_date}")

                filtered_df=filtered_df[['Name','Time']]

                headers=list(filtered_df.columns)
                rows=[tuple(map(str,row)) for row in filtered_df.values]
                print(headers)
                print(rows)

                if self.table:
                    self.ids.table_box.remove_widget(self.table)
                    
                self.table=MDDataTable(
                    column_data=[(col,dp(30)) for col in headers],
                    row_data=rows,
                    use_pagination=True,
                    rows_num=5,
                    size_hint=(1,0.8),               
                )
                self.ids.table_box.add_widget(self.table)
            else:
                self.ids.datelabel.text=f"No attendance found for {datetime.strptime(date, '%Y-%m-%d').strftime('%d %B %Y')}"
        else:
            print("Attendance file not found")            

    def show_filter_dialog(self):
        self.dialog=None
        if not self.dialog:
            self.dialog=MDDialog(
                title="Filter Attendance by date",
                type="custom",
                auto_dismiss=False,
                content_cls=MDTextField(hint_text="Enter date (YYYY-MM-DD))",helper_text_mode="on_focus"),
                buttons=[
                    MDRaisedButton(text="Filter",on_release=self.apply_filter),
                    MDRaisedButton(text="Cancel",on_release=lambda x:self.dialog.dismiss())
                ]
            )            
        self.dialog.open()

    def apply_filter(self,instance):
        global attendance_file
         
        date=self.dialog.content_cls.text.strip()
        try:
        # Try to parse the date
           valid_date = datetime.strptime(date, "%Y-%m-%d").date()
        # Now apply the filter using valid_date
           self.dialog.dismiss()
           self.dialog=None
           self.load_attendance_table(date)

        except ValueError:
           self.dialog.dismiss()
           self.show_invalid_date_popup()

    def show_invalid_date_popup(self):
        self.dialog=None
        self.dialog= MDDialog(
        title="Invalid Date Format",
        auto_dismiss=False,
        text="Please enter the date in YYYY-MM-DD format (e.g., 2024-03-12).",
        buttons=[MDRaisedButton(text="OK", on_release=lambda x: self.dialog.dismiss())]
        )
        self.dialog.open()

    def export_attendance(self):
        global attendance_file
        if os.path.exists(attendance_file):
            now=datetime.now().strftime("%d%m%Y_%H%M%S")  
            export_file=f"Exported_Attendance_{now}.csv"
            df=pd.read_csv(attendance_file) 
            if is_android():
                export_attendance_to_excel(df,attendance_file)
            else:    
               df.to_csv(export_file,index=False)
               print(f"Attendance exported to {export_file}")
        else:
            print("Attendance file not found")

    def show_manual_attendance_dialog(self):
        self.dialog=None
        if not self.dialog:
            self.dialog=MDDialog(
                title="Mark Attendance Manually",
                auto_dismiss=False,
                type="custom",
                content_cls=MDTextField(hint_text="Enter Students Name"),
                buttons=[
                    MDRaisedButton(text="Mark Present",on_release=self.mark_present),
                    MDRaisedButton(text="Cancel",on_release=lambda x:self.dialog.dismiss())
                ]
            )                         
        self.dialog.open()

    def invalidNameMessage(self):
        self.dialog.dismiss()
        self.dialog=None
        self.dialog= MDDialog(
        title="Error",
        auto_dismiss=False,
        text="Name field cannot be empty",
        buttons=[MDRaisedButton(text="OK", on_release=lambda x: self.dialog.dismiss())]
        )
        self.dialog.open()


    def mark_present(self,instance):
        global attendance_file
        if not self.dialog.content_cls.text.strip():
            self.invalidNameMessage()
            return
        name=self.dialog.content_cls.text.strip()
        self.dialog.dismiss()
        self.dialog=None
        if os.path.exists(attendance_file):
            df=pd.read_csv(attendance_file)
            today_date=datetime.now().strftime("%Y-%m-%d")
            current_time=datetime.now().strftime("%I:%M %p") 
             
            already_marked = not df[(df["Name"] == name) & (df["Date"] == self.Date)].empty
            if not already_marked:
               new_entry=pd.DataFrame([[name,self.Date,current_time]],columns=["Name","Date","Time"])
               matching_indices = df.index[df['Date'] == self.Date].tolist()
               if matching_indices:
                  insert_at = matching_indices[-1] + 1  # Insert after the last matching date
                  top = df.iloc[:insert_at]
                  bottom = df.iloc[insert_at:]
                  df = pd.concat([top, new_entry, bottom], ignore_index=True)
               else:
                   df = pd.concat([df, new_entry], ignore_index=True)

               df.to_csv(attendance_file,index=False)
               print(f"{name} marked present for {self.Date} at {current_time}")
            self.load_attendance_table(self.Date)
        else:
            print("Attendance file not found")   

    def on_leave(self):
       if self.table:
           self.ids.table_box.remove_widget(self.table)
           self.table = None

       if self.dialog:
           self.dialog.dismiss()
           self.dialog = None

    def switch_screen(self, screen_name):
       print(f"Switching from AttendanceScreen to {screen_name}...")  # Debugging
    # Switch screen
       self.manager.current = screen_name

 #-------------------------------------------------------------------------------------------------------------------------------------------------       

#Screen Manager
class AttendanceApp(MDApp):
    def build(self):
        if is_android():
           ask_android_permissions()
        Builder.load_file("home_screen.kv")
        sm=ScreenManager()
        sm.transition=NoTransition()
        sm.add_widget(HomeScreen(name="home"))
        sm.add_widget(TrainModelScreen(name="train"))
        sm.add_widget(RecognitionScreen(name="recognition"))
        sm.add_widget(AttendanceScreen(name="attendance"))
        sm.current= "home"
        return sm
       
if __name__=="__main__":
    AttendanceApp().run()    
     
