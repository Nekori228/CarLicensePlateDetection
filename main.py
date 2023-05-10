# Import dependencies
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pytesseract
import requests
import re
import datetime
import serial
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk

# Set tesseract path to where the tesseract exe file is located (Edit this path accordingly based on your own settings)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

carplate_img = cv2.imread('./images/car_image.png')

# Start video capture from default camera
capture = cv2.VideoCapture(0)
# Import Haar Cascade XML file for Russian car plate numbers
carplate_haar_cascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_russian_plate_number.xml')
registeredPlates = []
framesPassed = 0
detectionTime = datetime.datetime.now()
prevPlate = "000000"
textToPut = ""
textColor = (255, 0, 255)

ser = serial.Serial("COM5", 9600)

'''SELECT *, preg.allowed
FROM plates_reports as prep
LEFT JOIN plates_registered as preg 
        ON prep.license LIKE CONCAT (preg.license, '%');'''


def request_plates():
    url = 'https://spktt.ru/plater/api.php?method=get.plates'
    headers = {'Authorization': 'No'}
    payload = {'method': ''}

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        plates_json = response.json()
        registeredPlates.clear()
        for plates in plates_json["plates"]:
            registeredPlates.append(plates["license"])
        print(registeredPlates)
    except requests.exceptions.HTTPError as error:
        print(error)
        # This code will run if there is a 404 error.


def setBarrierState(state):
    ser.write(bytearray(state, 'ascii'))


def report_detection(plate, isAllowed, isRegistered):
    url = 'https://spktt.ru/plater/api.php?method=post.report'
    headers = {'Authorization': 'No'}
    payload = {'method': 'post.report', 'plate': plate, 'isAllowed': isAllowed, 'isRegistered': isRegistered}

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        print(response)
    except requests.exceptions.HTTPError as error:
        print(error)
        # This code will run if there is a 404 error.


request_plates()

'''while capture.isOpened():

    

cv2.imshow('Frame', carplate_img)
capture.release()
cv2.destroyAllWindows()  # destroy all opened windows'''

root = Tk()
root.title('Car license plate detector')
# root.geometry('640x520')
root.minsize(646, 530)
root.maxsize(1080, 720)
root.configure(bg='#303030')

if (capture.isOpened() == False):
    print("Unable to read camera feed")



def exitWindow():
    capture.release()
    cv2.destroyAllWindows()
    root.destroy()
    root.quit()


f1 = LabelFrame(root, bg='red')
f1.pack()
l1 = Label(f1, bg='red')
l1.pack()

l2 = Label(f1, bg='blue')
l2.place(x=0, y=0)

b1 = Button(root, fg='white', bg='green', activebackground='white', activeforeground='black', text='OPEN', relief=GROOVE,
            height=50, width=20, command=lambda: setBarrierState('O'))
b1.pack(side=LEFT, padx=5, pady=5)

b2 = Button(root, fg='white', bg='red', activebackground='white', activeforeground='red', text='CLOSE', relief=GROOVE,
            height=50, width=20, command=lambda: setBarrierState('C'))
b2.pack(side=LEFT, padx=5, pady=5)

b3 = Button(root, bg='blue', fg='white', activebackground='white', activeforeground='blue', text='Update plates',
            relief=GROOVE, height=50, width=20, command=request_plates)
b3.pack(side=RIGHT, padx=5, pady=5)

b4 = Button(root, fg='white', bg='red', activebackground='white', activeforeground='red', text='EXIT ❌ ', relief=GROOVE,
            height=50, width=20, command=exitWindow)
b4.pack(side=BOTTOM, padx=5, pady=5)

while True:
    if framesPassed > 5:
        framesPassed = 0
    else:
        framesPassed += 1
        continue

    ret, carplate_img = capture.read()

    if not ret or ret is None:
        # carplate_img = cv2.imread('./images/car_image.png')
        print("failed to grab frame")
        continue

    '''    
    try:
        ret, carplate_img = capture.read()

    except ValueError:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    '''

    # Read car image and convert color to RGB

    carplate_img_rgb = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)

    carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_img_rgb, scaleFactor=1.1, minNeighbors=5)
    # cv2.imshow('carplate_img_rgb', carplate_img)
    if len(carplate_rects) > 0:
        # print("Plates count: ",len(carplate_rects))

        # Function to enlarge the plt display for user to view more clearly
        def enlarge_plt_display(image, scale_factor):
            width = int(image.shape[1] * scale_factor / 100)
            height = int(image.shape[0] * scale_factor / 100)
            dim = (width, height)
            plt.figure(figsize=dim)
            plt.axis('off')
            plt.close()
            # plt.imshow(image)


        enlarge_plt_display(carplate_img_rgb, 1.2)


        # Setup function to detect car plate
        def carplate_detect(image):
            carplate_overlay = image.copy()  # Create overlay to display red rectangle of detected car plate
            carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_overlay, scaleFactor=1.1, minNeighbors=5)

            for x, y, w, h in carplate_rects:
                cv2.rectangle(carplate_overlay, (x, y), (x + w, y + h), (255, 0, 0), 5)

            return carplate_overlay


        detected_carplate_img = carplate_detect(carplate_img_rgb)
        enlarge_plt_display(detected_carplate_img, 1.2)


        # Function to retrieve only the car plate sub-image itself
        def carplate_extract(image):

            carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

            for x, y, w, h in carplate_rects:
                carplate_img = image[y + 15:y + h - 10, x + 15:x + w - 20]

            return carplate_img


        # Enlarge image for further image processing later on
        def enlarge_img(image, scale_percent):
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)

            if not ret or ret is None:
                return image
            else:
                resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                return resized_image


        # Display extracted car license plate image
        carplate_extract_img = carplate_extract(carplate_img_rgb)
        carplate_extract_img = enlarge_img(carplate_extract_img, 150)
        # plt.imshow(carplate_extract_img);

        # Convert image to grayscale
        carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)
        plt.axis('off')
        # plt.imshow(carplate_extract_img_gray, cmap='gray');

        # Apply median blur + grayscale
        carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img_gray, 3)  # Kernel size 3
        plt.axis('off')
        # plt.imshow(carplate_extract_img_gray_blur, cmap='gray');

        # Display the resulting image
        # cv2.imshow('carplate_extract_img_gray_blur', carplate_extract_img_gray_blur)
        # Display the text extracted from the car plate
        text = pytesseract.image_to_string(carplate_extract_img_gray_blur,
                                           config=f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCEHKMOPTYX0123456789')
        text = text.upper()

        text = re.sub(r"[^ABCEHKMOPTYX0-9]+", '', text)

        # img_roi = img[y: y + h, x:x + w]
        # cv2.imshow("ROI", edged)
        # cv2.putText(img, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        # display the license plate and the output image

        if re.match("[ABCEHKMOPTYX][0-9]{3}[ABCEHKMOPTYX]{2}[0-9]{2,3}", text):
            difference = datetime.datetime.now() - detectionTime
            difference = difference.seconds
            if (difference > 5 and prevPlate[0:6] != text[0:6]):
                prevPlate = text[0:6]
                print(prevPlate)
                detectionTime = datetime.datetime.now()
                if text in registeredPlates or any(text in s for s in registeredPlates):
                    print("REGISTERED PLATE IN ", text)
                    textToPut = text + " - REGISTERED"
                    report_detection(text, 1, 1)
                    setBarrierState('O')
                    textColor = (0, 255, 0)

                    # cv2.rectangle(carplate_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    print(text, " - ATTENDANCE REPORTED ")
                    report_detection(text, 0, 0)
                    setBarrierState('C')
                    textToPut = text + " - UNKNOWN"
                    textColor = (255, 0, 0)

            cv2.putText(carplate_img, textToPut, (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        textColor, 2)
        img2 = ImageTk.PhotoImage(Image.fromarray(carplate_extract_img_gray))
        l2['image'] = img2
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    img = ImageTk.PhotoImage(Image.fromarray(carplate_img))

    l1['image'] = img

    root.update()

capture.release()
ser.close()
