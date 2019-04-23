#
# Name: gvision.py
#
# Author: C.-H. Dominic HUNG <dominic@teli.hku.hk / chdhung@hku.hk>
#  Technology-Enriched Learning Initiative,
#  The University of Hong Kong
#
# Description: Library wrapping Google Vision API in support of Classwork 4
#  of CCST9015 offered in Spring 2019.
#

# TODO: REPLACE WITH A SHARED HEADER FILE THAT HOLDS THE PATH
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/usr/share/ccst9015/gcloud_ccst9015_sp19.json"

import io, numpy, cv2
from PIL import Image

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

#
# GOOGLE CLOUD SERVICES WRAPPING ROUTINES
#

# GVISION
# ====================
#
# The function will transfer the input image to Google Vision API to obtain Google
#  Cloud processing results according to selected processing strategy.
#
# @param img     A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 presented for Google Vision image processing service
# @param op_type A string for selecting the image process offered by Google Vision
#                 API. Supported options are 1) "text" 2) "label" 3) "face" 4) "object".
# @param max     An integer specifying the maximum number of objects or best matches
#         _detect to be returned by the image processing services. The parameter is not
#                  applicable to "text" identification service.
#
# @ret           A selected subset of Google Vision API rendered JSON object containing
#                 results returned by image processing services, selected according to
#                 op_type specified to provide most relevant returns.
#
def gvision(img, op_type = "text", max_detect = 10) :
    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    if img is not None and isinstance(img, Image.Image) :
        rawfile = io.BytesIO()

        img.save(rawfile, format='JPEG')

        rawfile = rawfile.getvalue()
    elif type(img) is io.BytesIO :
        img.seek(0)

        rawfile = img.getvalue()
    elif img is not None and isinstance(img, numpy.ndarray) :
        _, rawfile = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])

        rawfile = rawfile.tostring()
    else :
        raise TypeError

    image = types.Image(content=rawfile)

    OPTYPE_ERR = "[ERROR] Invalid Google Vision Detection Type."

    if op_type is not None and type(op_type) is str :
        if op_type == "text" :
            response = client.text_detection(image=image) # max_results has no effect to text detection

            # DETAILS OF RETURN OBJECT
            #
            # locale: "en"
            # description: String
            # bounding_poly:
            #  vertices:
            #   x: Integer 0-Max(RESOLUTION_X)
            #   y: Integer 0-Max(RESOLUTION_Y)
            #
            ret = response.text_annotations
        elif op_type == "label" :
            response = client.label_detection(image=image, max_results=max_detect)

            # DETAILS OF RETURN OBJECT
            #
            # description: String
            # score: Float 0-1
            # topicality: Float 0-1
            #
            ret = response.label_annotations
        elif op_type == "face" :
            response = client.face_detection(image=image, max_results=max_detect)

            # DETAILS OF RETURN OBJECT
            #
            # bounding_poly:
            #  vertices:
            #   x: Integer 0-MAX(RESOLUTION_X)
            #   y: Integer 0-MAX(RESOLUTION_Y)
            # fd_bounding_poly:
            #  vertices:
            #   x: Integer 0-MAX(RESOLUTION_X)
            #   y: Integer 0-MAX(RESOLUTION_Y)
            # landmarks:
            #  type: [LEFT/RIGHT]_EYE, [LEFT/RIGHT]_OF_[LEFT/RIGHT]_EYEBROW, MIDPOINT_BETWEEN_EYES, NOSE_TIP, [UPPER/LOWER]_LIP, MOUTH_[LEFT/CENTER/RIGHT], NOSE_BOTTOM_[LEFT/CENTER/RIGHT], [LEFT/RIGHT]_EYE_[TOP/BOTTOM]_BOUNDARY, [LEFT/RIGHT]_EYE_[LEFT/RIGHT]_CORNER, [LEFT/RIGHT]_EYE_PUPIL, [LEFT/RIGHT]_EYEBROW_UPPER_MIDPOINT, [LEFT/RIGHT]_EAR_TRAGION, FOREHEAD_GLABELLA, CHIN_GNATHION, CHIN_[LEFT/RIGHT]_GONION
            #  position:
            #   x: Flaot 0-MAX(RESOLUTION_X)
            #   Y: Float 0-MAX(RESOLUTION_Y)
            #   z: Float 0-
            # roll_angle
            # pan_angle
            # tilt_angle
            # detection_confidence
            # landmarking_confidence
            # [joysorrow/anger/surprise/under_exposed/blurred/headwear]_likelihood: int 1(VERY_UNLIKELY), 2(UNLIKELY), 3(POSSIBLE), 4(LIKELY), 5(VERY_LIKELY)
            #
            ret = response.face_annotations
        elif op_type == "object" :
            response = client.object_localization(image=image)

            # DETAILS OF RETURN OBJECT
            #
            # name: Identified Object Name
            # score: Flaot 0-1
            # bounding_poly:
            #  normalized_vertices:
            #   x: Float 0-1
            #   y: Float 0-1
            #
            ret = response.localized_object_annotations
        else :
            raise TypeError(OPTYPE_ERR)
    else :
        raise TypeError(OPTYPE_ERR)

    return ret

#
# (END OF) GOOGLE CLOUD SERVICES WRAPPING ROUTINES
#

#
# GOOGLE CLOUD VISION AUXILIARY ROUTINES
#

# FACE_EXTRACT
# ====================
#
# The function supports the filtering of face detection results returned from Google
#  Cloud Vision according to predicates.
#
# @param gvres   A list of objects found by Google Vision API, as in the format
#                 returned from Google Vision face detection API.
# @param emotion A string for sieving faces according to emotions detected by Google
#                 Vision. Supported emptions are 1) "joy" 2) "sorrow" 3) "anger" and
#                 4) "surprise".
# @param negate  A boolean value to dictate if the predicate should be inverted, e.g.,
#                 if emotion parameter is passed in with "joy", when negate is passed
#                 True, faces not identified as joyful would be returned as result.
#
# @ret           A selected subset of Google Vision API rendered JSON object containing
#                 results returned by image processing services, filtered according to
#                 predicates specified.
#
def face_extract(faces, emotion = "joy", negate = False) :
    # TODO: Check Instance Type of faces

    if emotion is not None and type(emotion) is str :
        if emotion == "joy" or emotion == "sorrow" \
            or emotion == "anger" or emotion == "surprise" :
            emotion += "_likelihood"
        else :
            raise ValueError
    else :
        raise TypeError

    if negate is not None and type(negate) is bool :
        pass
    else :
        raise TypeError

    ret = []

    for i in faces :
        if eval("i." + emotion) > 3 :
            if negate is False :
                ret.append(i)
        elif eval("i." + emotion) == 3 :
            ret.append(i)
        else :
            if negate is True :
                ret.append(i)

    return ret

FACE_LDMARK = dict()
FACE_LDMARK["LEFT_EYE"] = 1
FACE_LDMARK["RIGHT_EYE"] = 2
FACE_LDMARK["LEFT_OF_LEFT_EYEBROW"] = 3
FACE_LDMARK["RIGHT_OF_LEFT_EYEBROW"] = 4
FACE_LDMARK["LEFT_OF_RIGHT_EYEBROW"] = 5
FACE_LDMARK["RIGHT_OF_RIGHT_EYEBROW"] = 6
FACE_LDMARK["MIDPOINT_BETWEEN_EYES"] = 7
FACE_LDMARK["NOSE_TIP"] = 8
FACE_LDMARK["UPPER_LIP"] = 9
FACE_LDMARK["LOWER_LIP"] = 10
FACE_LDMARK["MOUTH_LEFT"] = 11
FACE_LDMARK["MOUTH_RIGHT"] = 12
FACE_LDMARK["MOUTH_CENTER"] = 13
FACE_LDMARK["NOSE_BOTTOM_RIGHT"] = 14
FACE_LDMARK["NOSE_BOTTOM_LEFT"] = 15
FACE_LDMARK["NOSE_BOTTOM_CENTER"] = 16
FACE_LDMARK["LEFT_EYE_TOP_BOUNDARY"] = 17
FACE_LDMARK["LEFT_EYE_RIGHT_CORNER"] = 18
FACE_LDMARK["LEFT_EYE_BOTTOM_BOUNDARY"] = 19
FACE_LDMARK["LEFT_EYE_LEFT_CORNER"] = 20
FACE_LDMARK["LEFT_EYE_PUPIL"] = 29
FACE_LDMARK["RIGHT_EYE_TOP_BOUNDARY"] = 21
FACE_LDMARK["RIGHT_EYE_RIGHT_CORNER"] = 22
FACE_LDMARK["RIGHT_EYE_BOTTOM_BOUNDARY"] = 23
FACE_LDMARK["RIGHT_EYE_LEFT_CORNER"] = 24
FACE_LDMARK["RIGHT_EYE_PUPIL"] = 30
FACE_LDMARK["LEFT_EYEBROW_UPPER_MIDPOINT"] = 25
FACE_LDMARK["RIGHT_EYEBROW_UPPER_MIDPOINT"] = 26
FACE_LDMARK["LEFT_EAR_TRAGION"] = 27
FACE_LDMARK["RIGHT_EAR_TRAGION"] = 28
FACE_LDMARK["FOREHEAD_GLABELLA"] = 31
FACE_LDMARK["CHIN_GNATHION"] = 32
FACE_LDMARK["CHIN_LEFT_GONION"] = 33
FACE_LDMARK["CHIN_RIGHT_GONION"] = 34

#
# (END OF) GOOGLE CLOUD VISION AUXILIARY ROUTINES
#

#
# USER ROUTINES
#

# MAIN
# ====================
#
# Main demonstration routine called when the module is ran in standalone mode.
#
# The default showcase routine is to present to Google API 4 individual images
#  for 4 supported services.
#
if __name__ == "__main__" :
    import image
    import os, time

    # Setting the common base path where input images can be found.
    #
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

    # [Demonstration 1] Text Detection by Google Cloud API
    #
    textf = BASE_PATH + "/cw4-input-text.jpg"
    # textf = BASE_PATH + "/cw4-input-object.jpg"

    print('Text Recognition:', end=' ')

    srcimg = Image.open(textf)

    ent = time.time()

    ret = gvision(srcimg, "text")

    print("(Latency: " + str(time.time() - ent) + "s)")

    for det in ret :
        print(det.description)

    # [Demonstration 2] Label Detection by Google Cloud API
    #
    lablf = BASE_PATH + "/cw4-input-label.jpg"
    # lablf = BASE_PATH + "/cw4-input-object.jpg"

    print('Image Labeling:', end=' ')

    srcimg = Image.open(lablf)

    ent = time.time()

    ret = gvision(srcimg, "label")

    print("(Latency: " + str(time.time() - ent) + "s)")

    for det in ret :
        print(det.description)

    # [Demonstration 3] Face Detection by Google Cloud API
    #
    facef = BASE_PATH + "/cw4-{}-face.jpg"

    print('Face Recognition:', end=' ')

    srcimg = Image.open(facef.format("input"))

    ent = time.time()

    ret = gvision(srcimg, "face")

    print("(Latency: " + str(time.time() - ent) + "s)")

    image.highlight_image(srcimg, ret, txttag = None, norm = False).save(facef.format("output"))

    print('saved to', facef.format("output"))

    # [Demonstration 4] Object Detection by Google Cloud API
    #
    # objf = BASE_PATH + "/cw4-{}-object.jpg"
    objf = BASE_PATH + "/cw4-{}-object-overlap.jpg"

    print('Object Recognition:', end=' ')

    srcimg = Image.open(objf.format("input"))

    ent = time.time()

    ret = gvision(srcimg, "object")

    print("(Latency: " + str(time.time() - ent) + "s)")

    image.highlight_image(srcimg, ret, txttag = "name").save(objf.format("output"))

    print('saved to', objf.format("output"))

#
# (END OF) USER ROUTINES
#