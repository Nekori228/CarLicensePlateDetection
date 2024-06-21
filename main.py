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
from six import StringIO

# Set tesseract path to where the tesseract exe file is located (Edit this path accordingly based on your own settings)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

carplate_img = cv2.imread('./images/car_image.png')

# Start video capture from default camera
capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 650)


# Import Haar Cascade XML file for Russian car plate numbers
carplate_haar_cascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_russian_plate_number.xml')
registeredPlates = []
restrictedPlates = []
recently_attended = []
framesPassed = 0
detectionTime = datetime.datetime.now()
prevPlate = "000000"
textToPut = ""
textColor = (255, 0, 255)

manualAccessRequested = 0

min_area = 500
count = 0

ser = serial.Serial("COM4", 9600)


def manualOpen():
    global manualAccessRequested
    manualAccessRequested = 1
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
    url = 'http://carplater/api/cars/list'
    headers = {'Authorization': 'Bearer 10|vZhYVeQLQklSZXdcoo1HbJx6YD7zU6BAafjYW2o6'}
    payload = {'method': ''}

    try:
        response = requests.get(url, headers=headers, data=payload)
        response.raise_for_status()
        print(response.content)
        cars_json = response.json()
        print(cars_json)
        registeredPlates.clear()

        for car in cars_json["cars"]:
            if car["owner"]["access_granted"] == 0:
                restrictedPlates.append(car["plate"])
                print("RESTRICTED PLATE "+car["plate"])
            else:
                registeredPlates.append(car["plate"])
        print(registeredPlates)
    except requests.exceptions.HTTPError as error:
        print(error)
        # This code will run if there is a 404 error.


def setBarrierState(state, plate, writeStatus):
    print(serial_ports())

    ser.write(bytearray(state, 'ascii'))
    if state == 'O':
        report_detection(plate, 1, writeStatus)
        #setBarrierState('O');


def report_detection(plate, isAllowed, writeStatus):
    if writeStatus is True:
        print("image written")
        with open('./temp/0.jpg', 'rb') as f:
            img_data = f.read()
            #image = {'image': open('./temp/0.jpg', 'rb')}
            files = {'image': ("plate.jpg", img_data)}
    else:
        print("problem")  # or raise exception, handle problem, etc.
        files = {'image': 0}


    url = 'http://carplater/api/detects'
    headers = {'Authorization': 'Bearer 10|vZhYVeQLQklSZXdcoo1HbJx6YD7zU6BAafjYW2o6'}

    payload = {'plate': plate, 'wasApproved': isAllowed}

    try:
        response = requests.post(url, headers=headers, data=payload, files=files)
        response.raise_for_status()
        print(response)
        print(response.content)
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
settingsmenu2.add_command(label="COM6")
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


f1 = LabelFrame(root)
f1.pack()
videoLabel = Label(f1, bg='black', width=900, height=650)
videoLabel.pack(side=LEFT)


scanned_plates_var = Variable(value=recently_attended)
#scanned_plates_listbox = Listbox(f1, listvariable=scanned_plates_var)
scanned_plates_listbox = Listbox(f1)
scanned_plates_listbox.pack(fill=Y, side=RIGHT, padx=5, pady=5)

plateLabel = Label(root, bg='black', height=45, width=190)

plateLabel.pack(side=LEFT, padx=5, pady=5 )
b1 = Button(root, fg='white', bg='#54b030', activebackground='white', activeforeground='black', text='OPEN ⬆️', relief=GROOVE,
            height=50, width=30, command=lambda: manualOpen())
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
                #cv2.imshow("1 ROI", img_roi)
                gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                #cv2.imshow("2 Gray", gray)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                #cv2.imshow("3 Blur", blur)
                edged = cv2.Canny(gray, 10, 200)
                cv2.imshow("4 edged", edged)
                #imagem = cv2.bitwise_not(edged)
                #cv2.imshow("5 imagem", imagem)

                resize_test_license_plate = cv2.resize(
                    frame, None, fx=2, fy=2,
                    interpolation=cv2.INTER_CUBIC)
                cv2.imshow("resize_test_license_plate", resize_test_license_plate)
                grayscale_resize_test_license_plate = cv2.cvtColor(
                    resize_test_license_plate, cv2.COLOR_BGR2GRAY)
                cv2.imshow("grayscale_resize_test_license_plate", grayscale_resize_test_license_plate)
                gaussian_blur_license_plate = cv2.GaussianBlur(
                    grayscale_resize_test_license_plate, (5, 5), 0)
                cv2.imshow("gaussian_blur_license_plate", gaussian_blur_license_plate)


                adjusted = gray #cv2.addWeighted(imagem, 3.0, imagem, 0, 0)

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
                    #text = pytesseract.image_to_string(license_plate, config=f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCEHKMOPTYX0123456789')
                    text = pytesseract.image_to_string(license_plate, lang='eng', config='--oem 3 -l eng --psm 6 -c tessedit_char_whitelist=ABCEHKMOPTYX0123456789')
                    text = text.upper()

                    text = re.sub(r"[^ABCEHKMOPTYX0-9]+", '', text)

                    if re.match("[ABCEHKMOPTYX][0-9]{3}[ABCEHKMOPTYX]{2}[0-9]{2,3}", text):
                        print(text)
                        difference = datetime.datetime.now() - detectionTime
                        difference = difference.seconds
                        if (difference > 5 and prevPlate[0:6] != text[0:6]):
                            prevPlate = text[0:6]
                            print(prevPlate)
                            detectionTime = datetime.datetime.now()

                            writeStatus = cv2.imwrite('./temp/0.jpg', frame)
                            if text in registeredPlates or any(text in s for s in registeredPlates):
                                print("REGISTERED PLATE IN ", text)
                                recently_attended.append(text)
                                textToPut = text + " - REGISTERED"

                                #frame_im = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
                                #pil_im = Image.fromarray(frame_im)
                                #stream = StringIO()
                                #pil_im.save(stream, format="JPEG")
                                #stream.seek(0)
                                #img_for_post = stream.read()

                                #report_detection()
                                setBarrierState('O', text, writeStatus)
                                textColor = (0, 255, 0)
                                # cv2.rectangle(carplate_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            else:
                                print(text, " - UNKNOWN PLATE ")

                                #frame_im = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
                                #pil_im = Image.fromarray(frame_im)
                                #stream = StringIO()
                                #pil_im.save(stream, format="JPEG")
                                #stream.seek(0)
                                #img_for_post = stream.read()

                                recently_attended.append(text)

                                #report_detection(text, 0, writeStatus)
                                if text in restrictedPlates or any(text in s for s in restrictedPlates):
                                    textToPut = text + " - RESTRICTED"
                                    setBarrierState('C', 0, 0)
                                else:
                                    if manualAccessRequested:
                                        print("Opening manually...")
                                        setBarrierState('O', text, writeStatus)
                                        manualAccessRequested = 0
                                    else:
                                        setBarrierState('C', 0, 0)
                                    textToPut = text + " - UNKNOWN"
                                textColor = (255, 0, 0)
                            print(recently_attended)
                            scanned_plates_listbox.insert(len(recently_attended), textToPut)
                            scanned_plates_listbox.pack()
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
