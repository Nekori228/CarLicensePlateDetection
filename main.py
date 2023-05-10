# Import dependencies
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pytesseract
import requests
import re
import datetime
import sys
import glob
import serial
from tkinter import messagebox, ttk
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk

# Set tesseract path to where the tesseract exe file is located (Edit this path accordingly based on your own settings)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

carplate_img = cv2.imread('./images/car_image.png')

# Start video capture from default camera
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 650)


# Import Haar Cascade XML file for Russian car plate numbers
carplate_haar_cascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_russian_plate_number.xml')
registeredPlates = []
framesPassed = 0
detectionTime = datetime.datetime.now()
prevPlate = "000000"
textToPut = ""
textColor = (255, 0, 255)


min_area = 500
count = 0

ser = serial.Serial("COM5", 9600)

def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result
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
    print(serial_ports())

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

root = Tk()
root.title('Car license plate detector')
# root.geometry('640x520')
root.minsize(1080, 720)
root.maxsize(1080, 720)
root.configure(bg='#303030')

mainmenu = Menu(root)
settingsmenu = Menu(mainmenu, tearoff=0)

settingsmenu2 = Menu(settingsmenu, tearoff=0)
settingsmenu2.add_command(label="COM4")
settingsmenu2.add_command(label="COM5")

settingsmenu.add_cascade(label="COM-port", menu=settingsmenu2)

root.config(menu=mainmenu)



if (capture.isOpened() == False):
    print("Unable to read camera feed")



def exitWindow():
    capture.release()
    cv2.destroyAllWindows()
    root.destroy()
    root.quit()


f1 = LabelFrame(root, bg='red')
f1.pack()
videoLabel = Label(f1, bg='black', width=1080, height=650)
videoLabel.pack()

plateLabel = Label(root, bg='black', height=45, width=190)

plateLabel.pack(side=LEFT, padx=5, pady=5 )
b1 = Button(root, fg='white', bg='#54b030', activebackground='white', activeforeground='black', text='OPEN ⬆️', relief=GROOVE,
            height=50, width=30, command=lambda: setBarrierState('O'))
b1.pack(side=LEFT, padx=5, pady=5)

b2 = Button(root, fg='white', bg='#c41d23', activebackground='white', activeforeground='#c41d23', text='CLOSE ⬇️', relief=GROOVE,
            height=50, width=30, command=lambda: setBarrierState('C'))
b2.pack(side=LEFT, padx=5, pady=5)

b3 = Button(root, bg='#3268a8', fg='white', activebackground='white', activeforeground='#3268a8', text='Update plates',
            relief=GROOVE, height=50, width=30, command=request_plates)
b3.pack(side=LEFT, padx=5, pady=5)

b4 = Button(root, fg='white', bg='#c41d23', activebackground='white', activeforeground='#c41d23', text='EXIT ❌', relief=GROOVE,
            height=50, width=30, command=exitWindow)
b4.pack(side=LEFT, padx=5, pady=5)



while True:
    n_plate_cnt = []
    if framesPassed > 5:
        framesPassed = 0
    else:
        framesPassed += 1
        continue
    ret, frame = capture.read()


    if not ret or ret is None:
        # carplate_img = cv2.imread('./images/car_image.png')
        print("failed to grab frame")
        continue

    carplate_rects = carplate_haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    # cv2.imshow('carplate_img_rgb', carplate_img)
    if len(carplate_rects) > 0:
        plate_cascade = carplate_haar_cascade
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        for (x, y, w, h) in plates:
            area = w * h

            if area > min_area:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                img_roi = frame[y: y + h, x:x + w]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                edged = cv2.Canny(gray, 10, 200)
                imagem = cv2.bitwise_not(gray)
                adjusted = cv2.addWeighted(imagem, 3.0, imagem, 0, 0)

                # find the contours, sort them, and keep only the 5 largest ones
                contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

                # loop over the contours
                for c in contours:
                    # approximate each contour
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    # if the contour has 4 points, we can say
                    # that we have found our license plate
                    if len(approx) == 4:
                        n_plate_cnt = approx
                        break
                if len(n_plate_cnt) == 4:
                    # get the bounding box of the contour and
                    # extract the license plate from the image
                    (x, y, w, h) = cv2.boundingRect(n_plate_cnt)
                    license_plate = adjusted[y:y + h + 2, x:x + w + 2]

                    # Display the resulting image
                    # cv2.imshow('carplate_extract_img_gray_blur', carplate_extract_img_gray_blur)
                    # Display the text extracted from the car plate
                    text = pytesseract.image_to_string(license_plate, config=f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCEHKMOPTYX0123456789')
                    text = text.upper()

                    text = re.sub(r"[^ABCEHKMOPTYX0-9]+", '', text)

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
                        cv2.putText(frame, textToPut, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, textColor, 2)
                    img2 = ImageTk.PhotoImage(Image.fromarray(license_plate))
                    plateLabel['image'] = img2

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    img = ImageTk.PhotoImage(Image.fromarray(frame))

    videoLabel['image'] = img

    root.update()

capture.release()
ser.close()
