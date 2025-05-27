Overview
A Python-based attendance system using facial recognition with a user-friendly GUI built in Kivy. It allows training, automatic attendance marking, and attendance management for different subjects and semesters.

Features
Popup to select semester and subject at launch

Train model with webcam images (50 per student)

Mark attendance via live face recognition (Haar Cascade, LBPH, DNN)

View, filter, manually edit, and export attendance

Simple, modular Kivy-based interface

Technologies Used
Python
Kivy
OpenCV
Haar Cascade, LBPH, DNN
CSV for data storage

🚀 How to Run
Clone the repository

Install dependencies:
bash
Copy
Edit
pip install kivy opencv-python
Run the main app:

bash
Copy
Edit
python main.py
Enter semester and subject when prompted.

📁 File Structure
main.py – Entry point with GUI

train.py – For image capture and training

recognize.py – For live attendance marking

data/ – Stores images and CSV files

✅ Requirements
Python 3.6+

Webcam

Required Python packages listed in requirements.txt# Face_Recog
