#
# Name: cw4.py
#
# Author: C.-H. Dominic HUNG <dominic@teli.hku.hk / chdhung@hku.hk>
#  Technology-Enriched Learning Initiative,
#  The University of Hong Kong
#
# Description: Demonstration program for classwork 4 of CCST9015 offered in
#  Spring 2019. The program showcasts realtime identification of objects in
#  image stream by Google Vision API and highlight distinguished objects and
#  relevant descriptions in a video stream.
#

import image, gvision
import time, signal
import pygame
import tkinter as tk
import smtplib,ssl
from email.mime.multipart import MIMEMultipart  
from email.mime.base import MIMEBase  
from email.mime.text import MIMEText  
from email.utils import formatdate  
from email import encoders


# __exit
# ====================
#
# Perform clean up tasks upon receiving termination signals and exit gracefully.
#
# @param signum  Signal dispatched to current process and passed to this handler
# @param curstackCurrent stack frame (Not usually used)
#
def __exit(signum, curstack) :
    if __senhat is not None :
        __senhat.clear()

    quit()

signal.signal(signal.SIGINT, __exit)

def callpolice():
    print("Alerting Police....")
    print("3....2.....1....")
    pygame.mixer.init()
    pygame.mixer.music.load("/home/student/Desktop/Classwork/CW4/siren.wav")
    pygame.mixer.music.play()

def send_an_email():
    toaddr = 'amruth1999@gmail.com'      # To id 
    me = 'amruth1999@gmail.com'          # your id
    subject = "Weapon Detection"              # Subject
    msg = MIMEMultipart()  
    msg['Subject'] = subject  
    msg['From'] = me  
    msg['To'] = toaddr  
    msg.preamble = "test "   
    #msg.attach(MIMEText(text))  
    part = MIMEBase('application', "octet-stream")  
    part.set_payload(open("/home/student/Desktop/Classwork/CW4/detectedgun.jpg", "rb").read())  
    encoders.encode_base64(part)  
    part.add_header('Content-Disposition', 'attachment; filename="detectedgun.jpg"')   # File name and format name
    msg.attach(part)  
    try:  
       s = smtplib.SMTP('smtp.gmail.com', 587)  # Protocol
       s.ehlo()  
       s.starttls()  
       s.ehlo()  
       s.login(user = 'amruth1999@gmail.com', password = 'amruthraghav')  # User id & password
       #s.send_message(msg)  
       s.sendmail(me, toaddr, msg.as_string())  
       s.quit()  
    #except:  
    #   print ("Error: unable to send email")    
    except SMTPException as error:  
          print ("Error")                # Exception
   
#
# USER LOGIC
#

# Declare and define a variable here so when the variable is read later in the
#  program, there is a definition.
#
# Analogy, what can you do if your parents asks your boyfriend or girlfriend
#  to come for family reunion dinner when you have none? (Forever alone!) When
#  you can not handle the case, why would you think your computer knows how to
#  handle non-existing objects? (wink)
#
SAMPLE_SEP = 20
detect_sel = 0

ret = []

agg = 0
ent = time.time()

# A loop that runs forever without stopping
#
while True :
	# Capture a video frame by calling `record_video()' in `image' module.
	#
    logoimg=image.load_image("/home/student/Desktop/Classwork/CW4/lifelogo.jpg")
    frm = image.record_video()
    frm = image.overlay_text(frm,"LifeLine 1.0 ",(10,20),(0,0,0),False)

    # If the counter `detect_sel' is divisible by 20, i.e., 0, 20, 40, ...,
    #  call `gvision()' in the `gvision' module and gives the video frame as an
    #  argument for `object' identification.
    #
    # As the submission of a video frame over the WiFi, through the Internet to
    #  distant Google Vision API incurs heavy latency, aka., delay, the choice
    #  of sending 1 video frame in every 20 frames captured is an engineering
    #  compromise for perceived smoother user experience over actual smoothness
    #  in captioning following identifiable objects.
    #
    # !!!Engineering is an art of compromise between theoretical science and
    #  realistics constraints!!!
    #
    # The result returned from the `gvision' function is a long list of objects
    #  distinguishable by the Google Vision with their locations and recognised
    #  object type descriptions.
    #
    # Recall above, if we have not give meaning to `detect_sel', how possible
    #  the computer can divide non-existence with 20 and know the results? It is
    #  non-existence! Not ZERO!
    #
    
    if detect_sel % SAMPLE_SEP == 0 :
        ret = gvision.gvision(frm, "object")
        #print("Number of Objects Found : " .format(len(ret)))
        for object_ in ret:
            name1= format(object_.name)
            if name1 == "Handgun" :
                a = image.record_image()
                a = image.overlay_text(a,"GUN Detected", (10,20), (0,0,0) , False)
                image.replay_image(a)
                print("Gun Detected...")
                image.save_image(a,"/home/student/Desktop/LifeLine/detectedgun.jpg")
                root = tk.Tk()
                logo = tk.PhotoImage(file="/home/student/Desktop/Classwork/CW4/gun.png")
                one = tk.Label(root,text="GUN ALERT!!! A gun has been detected in LE4 HKU. Click the button to call for Backup!!",bg="red", fg="white",font ="Times 28 bold")
                button= tk.Button(root,text="Call Backup", command=callpolice, font ="Times 28 bold")
                theLabel = tk.Label(root, image=logo)
                one.pack()
                button.pack()
                theLabel.pack()
                tk.mainloop()
            if  name1 == "Rifle" or name1 == "Shot gun":
                a = image.record_image()
                a = image.overlay_text(a,"Rifle/SHotgun Detected", (10,20), (0,0,0) , False)
                image.replay_image(a)
                print("Rifle/ShotGun Detected...")
                image.save_image(a,"/home/student/Desktop/LifeLine/detectedshotgun.jpg")
                root = tk.Tk()
                logo = tk.PhotoImage(file="/home/student/Desktop/Classwork/CW4/rifle.png")
                one = tk.Label(root,text="Rifle/ShotGun ALERT!!! A Rifle/ShotGun has been detected in LE4 HKU. Click the button to call for Backup!!",bg="red", fg="white",font ="Times 28 bold")
                button= tk.Button(root,text="Call Backup", command=callpolice, font ="Times 28 bold")
                theLabel = tk.Label(root, image=logo)
                one.pack()
                button.pack()
                theLabel.pack()
                tk.mainloop()
                #send_an_email() 
            if name1 == "Kitchen knife" or name1 == "Knife" or name1 == "Tableware knife":
                print("Knife Detected")
                a = image.record_image()
                a = image.overlay_text(a,"Knife Detected", (10,20), (0,0,0) , False)
                image.replay_image(a)
                image.save_image(a,"/home/student/Desktop/Classwork/CW4/detectedknife.jpg")
                root = tk.Tk()
                button= tk.Button(root,text="Call Backup", command=callpolice, font ="Times 28 bold")
                logo = tk.PhotoImage(file="/home/student/Desktop/Classwork/CW4/knife.png")
                one = tk.Label(root,text="Knife Alert. A Knife has been detected in LE4 HKU. Click the Button to call for Backup!!",bg="red", fg="white", font ="Times 28 bold")
                theLabel = tk.Label(root, image=logo)
                one.pack()
                button.pack()
                theLabel.pack()
                tk.mainloop()
                

    # The original captured video frame with the Google Vision result list is
    #  put as argument to `highlight_image' function in `image' module for
    #  image rendering, i.e., framing identified objects and caption the type
    #  an object is identified as. The result from `highlight_image' is directly
    #  feed as an argument to `replay_video' function in `image' module to
    #  trigger a display window or update an existing triggered display window.
    #
    # Note that, if `gvision' is not run in current step, e.g., in counter 1, 2,
    #  ..., 19 and 21, 22, ..., 39 and ..., the stagnant result is used to frame
    #  and tag objects. Therefore, if the up-to-date image shifted from previous
    #  frame too much, the framing will mis-align for sure. But it is again,
    #  a compromise for tolerable user experience.
    #
    image.replay_video(image.highlight_image(frm, ret, txttag = "name"))
        

    # We advance the counter by 1, this counter will overflow if the program is
    #  executed for infinite time. But for sure, this will not happen, you have
    #  to return this Raspberry Pi to us! =]
    #
    # What is overflow? Imagine we have an A4 paper, write mathematical number
    #  Pi on it.  3.14? The paper can hold 4 characters for sure! What about...
    #  3.141592654 a result from calculator, just 11 characters, fine!
    #
    # But what about I want accurate Pi! And you have to infinitely expand, more
    #  and more digits coming in. The dimension of A4 paper is an example of
    #  physical constraint and there is limit to how many things be written to
    #  an A4. Pi is a real number that can be infinitely expanded depending on
    #  the accuracy we want.
    #
    # Computer space for holding a number is finite, therefore, we we keep
    #  incrementing a number, we will hit a point the computer can no longer
    #  keep all digits and the computer crashes.
    #
    detect_sel += 1

    end = time.time()
    dlt = end - ent
    agg += dlt

    #print("Frame Rate: " + str(1 / dlt) + " FPS (Average: " + str(1 / (agg / detect_sel)) + " FPS)")

    ent = time.time()

# Destroy opened video windows.
#
cv2.destroyAllWindows()

#
# (END OF) USER LOGIC
#