#
# Name: cw4-face.py
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
import os, time, signal
import numpy, cv2, random
from PIL import Image

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
pjoy = None
psup = None
psad = None
pang = None
njoy = None

__easter = False

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

ret = []

predicate = dict()
predicate["joy"] = False
predicate["sur"] = False
predicate["sad"] = False
predicate["ang"] = False

agg = 0
ent = time.time()

# A loop that runs forever without stopping
#
while True :
    # Capture a video frame by calling `record_video()' in `image' module.
    #
    frm = image.record_video()

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
        ret = gvision.gvision(frm, "face")

        # Extract recognised faces that is identified to be joyful
        #
        pjoy = gvision.face_extract(ret, emotion = "joy", negate = False)
        njoy = gvision.face_extract(ret, emotion = "joy", negate = True)

        # Extract recognised faces that is identified to be surprise
        #
        psup = gvision.face_extract(ret, emotion = "surprise", negate = False)

        # Extract recognised faces that is identified to be sorrow
        #
        psad = gvision.face_extract(ret, emotion = "sorrow", negate = False)

        # Extract recognised faces that is identified to be angry
        #
        pang = gvision.face_extract(ret, emotion = "anger", negate = False)

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
    # frm = image.overlay_text(frm, "Faces Detected: " + str(len(pjoy)) + " / " + str(len(ret)), anchor = (-20, -20), colour = (255, 255, 255))
    frm = image.overlay_text(frm, "=D " + str(len(pjoy)), anchor = (-20, -120), colour = (0, 255, 0))
    frm = image.overlay_text(frm, "=[ " + str(len(psad)), anchor = (-20, -96), colour = (0, 0, 0))
    frm = image.overlay_text(frm, "=O " + str(len(psup)), anchor = (-20, -68), colour = (0, 0, 255))
    frm = image.overlay_text(frm, ">( " + str(len(pang)), anchor = (-20, -44), colour = (255, 0, 0))
    frm = image.overlay_text(frm, "Faces: " + str(len(ret)), anchor = (-20, -20), colour = ((255, 255, 255) if __easter is False else (172, 128, 255)))

    frm = image.overlay_image(frm, image.load_image(BASE_PATH + "/TELI_Logo-24px.jpg"), anchor = (-20, 5))

    frm = image.highlight_image(frm, pjoy, colour=(0, 255, 0), txttag = None, norm = False)

    frm = image.highlight_image(frm, njoy, colour=(255, 0, 0), txttag = None, norm = False)

    if __easter is True :
        BLOCK_LENGTH = 16

        for obj in psad :
            if isinstance(frm, numpy.ndarray) is not True :
                box = [(vertex.x, vertex.y) for vertex in obj.bounding_poly.vertices]

                anchor_tl = (min([x[0] for x in box]), min([y[1] for y in box]))
                anchor_br = (max([x[0] for x in box]), max([y[1] for y in box]))

                for i in range(anchor_tl[0], anchor_br[0] - BLOCK_LENGTH, BLOCK_LENGTH) :
                    for j in range(anchor_tl[1], anchor_br[1] - BLOCK_LENGTH, BLOCK_LENGTH) :
                        _ = random.randrange(127, 255, 16)

                        mos = Image.new("RGB", (BLOCK_LENGTH, BLOCK_LENGTH), (_, _, _))

                        frm = image.overlay_image(frm, mos, anchor = (i, j), paste = True)
            else :
                box = numpy.array([(vertex.x, vertex.y) for vertex in obj.bounding_poly.vertices], numpy.int32)

                anchor_tl = (min(box.swapaxes(0, 1)[0]), min(box.swapaxes(0, 1)[1]))
                anchor_br = (max(box.swapaxes(0, 1)[0]), max(box.swapaxes(0, 1)[1]))

                for i in range(anchor_tl[0], anchor_br[0] - BLOCK_LENGTH, BLOCK_LENGTH) :
                    for j in range(anchor_tl[1], anchor_br[1] - BLOCK_LENGTH, BLOCK_LENGTH) :
                        _ = random.randrange(127, 255, 16)
    
                        mos = numpy.empty((BLOCK_LENGTH, BLOCK_LENGTH, 3), dtype = numpy.uint8)
    
                        cv2.rectangle(mos, (0, 0), (BLOCK_LENGTH, BLOCK_LENGTH), (_, _, _), -1)
    
                        frm = image.overlay_image(frm, mos, anchor = (i, j), paste = True)

        BLOCK_LENGTH = 4

        for obj in pang :
            if isinstance(frm, numpy.ndarray) is not True :
                anchor_tl = [0, 0]
                anchor_br = [0, frm.size[1]]

                for landmark in obj.landmarks :
                    if landmark.type == gvision.FACE_LDMARK["LEFT_OF_LEFT_EYEBROW"] :
                        anchor_tl[0] = landmark.position.x
                    elif landmark.type == gvision.FACE_LDMARK["RIGHT_OF_RIGHT_EYEBROW"] :
                        anchor_br[0] = landmark.position.x
                    elif landmark.type == gvision.FACE_LDMARK["LEFT_EYEBROW_UPPER_MIDPOINT"] :
                        if landmark.position.y > anchor_tl[1] :
                            anchor_tl[1] = landmark.position.y
                    elif landmark.type == gvision.FACE_LDMARK["LEFT_EAR_TRAGION"] :
                        if landmark.position.y < anchor_br[1] :
                            anchor_br[1] = landmark.position.y
                    elif landmark.type == gvision.FACE_LDMARK["RIGHT_EYEBROW_UPPER_MIDPOINT"] :
                        if landmark.position.y > anchor_tl[1] :
                            anchor_tl[1] = landmark.position.y
                    elif landmark.type == gvision.FACE_LDMARK["RIGHT_EAR_TRAGION"] :
                        if landmark.position.y < anchor_br[1] :
                            anchor_br[1] = landmark.position.y

                for i in range(int(anchor_tl[0]), int(anchor_br[0]) - BLOCK_LENGTH, BLOCK_LENGTH) :
                    for j in range(int(anchor_tl[1]), int(anchor_br[1]) - BLOCK_LENGTH, BLOCK_LENGTH) :
                        _ = random.randrange(127, 255, 16)

                        mos = Image.new("RGB", (BLOCK_LENGTH, BLOCK_LENGTH), (_, _, _))

                        frm = image.overlay_image(frm, mos, anchor = (i, j), paste = True)
            else :
                anchor_tl = [0, 0]
                anchor_br = [0, frm.shape[0]]

                for landmark in obj.landmarks :
                    if landmark.type == gvision.FACE_LDMARK["LEFT_OF_LEFT_EYEBROW"] :
                        anchor_tl[0] = landmark.position.x
                    elif landmark.type == gvision.FACE_LDMARK["RIGHT_OF_RIGHT_EYEBROW"] :
                        anchor_br[0] = landmark.position.x
                    elif landmark.type == gvision.FACE_LDMARK["LEFT_EYEBROW_UPPER_MIDPOINT"] :
                        if landmark.position.y > anchor_tl[1] :
                            anchor_tl[1] = landmark.position.y
                    elif landmark.type == gvision.FACE_LDMARK["LEFT_EAR_TRAGION"] :
                        if landmark.position.y < anchor_br[1] :
                            anchor_br[1] = landmark.position.y
                    elif landmark.type == gvision.FACE_LDMARK["RIGHT_EYEBROW_UPPER_MIDPOINT"] :
                        if landmark.position.y > anchor_tl[1] :
                            anchor_tl[1] = landmark.position.y
                    elif landmark.type == gvision.FACE_LDMARK["RIGHT_EAR_TRAGION"] :
                        if landmark.position.y < anchor_br[1] :
                            anchor_br[1] = landmark.position.y

                for i in range(int(anchor_tl[0]), int(anchor_br[0]) - BLOCK_LENGTH, BLOCK_LENGTH) :
                    for j in range(int(anchor_tl[1]), int(anchor_br[1]) - BLOCK_LENGTH, BLOCK_LENGTH) :
                        _ = random.randrange(127, 255, 16)
    
                        mos = numpy.empty((BLOCK_LENGTH, BLOCK_LENGTH, 3), dtype = numpy.uint8)
    
                        cv2.rectangle(mos, (0, 0), (BLOCK_LENGTH, BLOCK_LENGTH), (_, _, _), -1)
    
                        frm = image.overlay_image(frm, mos, anchor = (i, j), paste = True)

    image.replay_video(frm)

    # Special Feature #0
    #
    if len(ret) is not 0 and len(ret) == len(pjoy) and __easter is False :
        if predicate["ang"] is True :
            pass
        else :
            prescale = image.SCALE

            image.resolution_rescale(1)

            img = image.record_image()

            image.resolution_rescale(prescale)

            img = image.overlay_image(img, image.load_image(BASE_PATH + "/TELI_Logo-24px.jpg"), anchor = (-20, 5))

            img = image.overlay_text(img, time.strftime("%Y-%M-%d %H:%M"), anchor = (-20, -20), colour = (255, 255, 255))

            image.replay_image(img)

            image.save_image(img, BASE_PATH + "/IMG-" + time.strftime("%Y-%M-%d %H:%M:%S") + ".jpg")

            predicate["ang"] = True
    else :
        predicate["ang"] = False

    # # Special Feature #0 [Alternative Code]
    # #
    # if len(ret) is not 0 and len(ret) == len(pjoy) :
    #     img = image.record_image()
    #
    #     img = image.overlay_image(img, image.load_image(BASE_PATH + "/TELI_Logo-24px.jpg"), anchor = (-20, 5))
    #
    #     img = image.overlay_text(img, time.strftime("%Y-%M-%d %H:%M"), anchor = (-20, -20), colour = (255, 255, 255))
    #
    #     image.replay_image(img)
    #
    #     image.save_image(img, BASE_PATH + "/IMG-" + time.strftime("%Y-%M-%d %H:%M:%S") + ".jpg")

    # Special Feature #1
    #
    if len(ret) > 1 and len(psup) > 1 and __easter is False:
        if predicate["sad"] is True :
            pass
        else :
            prescale = image.SCALE

            for _ in range(2, 3) :
                image.replay_video(image.load_image(BASE_PATH + "/.image/" + str(image.SCALE) + "/Sequel-{}.jpg".format(_ + 1)))

                time.sleep(0.3)

            image.replay_video(image.load_image(BASE_PATH + "/.image/" + str(image.SCALE) + "/Sequel-{}.jpg".format(4)))

            image.resolution_rescale(1)

            img = image.record_image()

            image.resolution_rescale(prescale)

            img = image.overlay_image(img, image.load_image(BASE_PATH + "/TELI_Logo-24px.jpg"), anchor = (-20, 5))

            img = image.overlay_text(img, time.strftime("%Y-%M-%d %H:%M"), anchor = (-20, -20), colour = (236, 78, 53))

            image.replay_image(img)

            image.save_image(img, BASE_PATH + "/Gotcha-" + time.strftime("%Y-%M-%d %H:%M:%S") + ".jpg")

            predicate["sad"] = True

            __easter = True
    else :
        predicate["sad"] = False

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