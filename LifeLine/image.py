#
# Name: image.py
#
# Author: C.-H. Dominic HUNG <dominic@teli.hku.hk / chdhung@hku.hk>
#  Technology-Enriched Learning Initiative,
#  The University of Hong Kong
#
# Description: Library supporting image acquisition/replay, load/save and
#  other additional image rendering capabilities, e.g., object framing.
#
#  The library is prepared in support of Classwork 4 of CCST9015 offered in
#  Spring 2019.
#

from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy, cv2
import io, tempfile, copy

# PARAMETERS
# ====================
#
CAMH = 1920                # HORIZONTAL RESOLUTION (IN PIXEL) OF CAMERA
CAMV = 1080                # VERTICAL RESOLUTION (IN PIXEL) OF CAMERA
SCALE = 4                  # DOWN SCALE FACTOR FOR BANDWIDTH CONSERVATION
LINEWIDTH = 5              # WIDTH OF LINES (IN PIXEL) FOR FRAMING IDENTIFIED OBJECTS

#
# IMAGE RENDERING ROUTINES
#

# HIGHLIGHT_IMAGE
# ====================
#
# Given an image, this function will frame objects according to user-provided locations
#  markers and tag the objects with text labels if they are presented altogether.
#
# @param img     A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 presented for highlighting objects within. For NumPy Array object,
#                 OpenCV drawing is used instead of PIL ImageDraw routines.
# @param objs    A list of objects found by Google Vision API, as in the format
#                 returned from Google Vision API
# @param colour  A tuple of three 8-bit integer elements of individual component
#                 colour Red, Green and Blue that represents the colour intended to
#                 be used for framing objects and underlining text (if applicable).
# @param txttag  None or a string for matching the tag associated with purposeful
#                 descriptions of objects as determined by Google Vision API
# @param tag     A function for matching # TODO:
# @param norm    A boolean variable informing of the function whether the position
#                 as presented by Google Vision API is normalised against pictoral
#                 dimension or an absolute pixel position. Normalised locations
#                 provide x-, y- position within the range of 0-1 and requires
#                 factoring with the dimension of the associated axes.
#
# @ret           A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 with required highlighting frames and (if applicable) relevant
#                 taggings as decided by input. The type of object returned would
#                 be the same as the img object type.
#
def highlight_image(img, objs, colour = (255, 0, 0), txttag = "description", norm = True) :
    if img is not None and isinstance(img, Image.Image) is True :
        pilimg = copy.copy(img)

        draw = ImageDraw.Draw(pilimg)
    elif img is not None and type(img) is io.BytesIO :
        img.seek(0)

        pilimg = Image.open(img)

        draw = ImageDraw.Draw(pilimg)
    elif img is not None and isinstance(img, numpy.ndarray) is True :
        img = copy.copy(img)
    else :
        raise TypeError

    # Specify the text font style and size to be used for labelling descriptions/
    #  taggings in the picture.
    #
    if isinstance(img, numpy.ndarray) is not True :
        font = ImageFont.truetype(font="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size=24)
    else :
        font = (cv2.FONT_HERSHEY_SIMPLEX, 0.6)

    # TODO: Check Instance Type of objs

    if colour is not None and (type(colour) is tuple or type(colour) is list) and len(colour) is 3 :
        for i in 0, 1, 2 :
            if colour[i] < 0 or colour[i] > 255 :
                raise ValueError
            else :
                pass

        if (colour[0] * 0.299 + colour[1] * 0.587 + colour[2] * 0.114) / 255 > 0.5 :
            txtclr = (0, 0, 0)
        else :
            txtclr = (255, 255, 255)
    else :
        raise TypeError

    if norm is not None and type(norm) is bool :
        pass
    else :
        raise TypeError

    for obj in objs:
        if txttag is not None and type(txttag) is str :
            captxt = eval("obj." + txttag)
        elif txttag is None :
            captxt = None
        else :
            raise TypeError

        if isinstance(img, numpy.ndarray) is not True :
            if norm == False :
                box = [(vertex.x, vertex.y) for vertex in obj.bounding_poly.vertices]
            else :
                box = [(pilimg.size[0] * vertex.x, pilimg.size[1] * vertex.y) for vertex in obj.bounding_poly.normalized_vertices]

            if captxt is not None :
                anchor = (min([x[0] for x in box]), min([y[1] for y in box]))

                txtbox = draw.textsize(captxt, font=font)
                txtbox = (anchor[0] + LINEWIDTH / 2 + txtbox[0] + LINEWIDTH, anchor[1] + txtbox[1] + LINEWIDTH / 2)

            draw.line(box + [box[0]], width=LINEWIDTH, fill=colour)

            if captxt is not None :
                draw.rectangle([anchor, txtbox], fill=colour)
                draw.text((anchor[0] + LINEWIDTH / 2, anchor[1]), captxt, font=font, fill=txtclr)
        else :
            if norm == False :
                box = numpy.array([(vertex.x, vertex.y) for vertex in obj.bounding_poly.vertices], numpy.int32)
            else :
                box = numpy.array([(img.shape[1] * vertex.x, img.shape[0] * vertex.y) for vertex in obj.bounding_poly.normalized_vertices], numpy.int32)

            if captxt is not None :
                anchor = (min(box.swapaxes(0, 1)[0]), min(box.swapaxes(0, 1)[1]))

                txtbox = cv2.getTextSize(captxt, font[0], font[1], 1)[0]

            cv2.polylines(img, [box.reshape(-1, 1, 2)], True, colour, LINEWIDTH)

            if captxt is not None :
                cv2.rectangle(img, (anchor[0], anchor[1]), (int(anchor[0] + LINEWIDTH / 2 + txtbox[0] + LINEWIDTH), int(anchor[1] + txtbox[1] + LINEWIDTH / 2)), colour, -1)
                cv2.putText(img, captxt, (int(anchor[0] + LINEWIDTH / 2), int(anchor[1] + txtbox[1])), font[0], font[1], txtclr)

    if type(img) is io.BytesIO :
        img.seek(0)

        ret = io.BytesIO()

        pilimg.save(ret, "JPEG")

        return ret
    elif type(img) is numpy.ndarray :
        return img
    else :
        return pilimg

# OVERLAY_IMAGE
# ====================
#
# This function superimposes an image, e.g., a TV station logo, over a master image.
#  The overlaying object must be smaller than the master and has to be placed with
#  the canvas, which is of the dimension of the master.
#
# @param img     A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 presented for pasting another image atop. For NumPy Array object,
#                 OpenCV drawing is used instead of PIL ImageDraw routines.
# @param ovly_imgA PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 presented to be pasted on master image. For NumPy Array object,
#                 OpenCV drawing is used instead of PIL ImageDraw routines. The type
#                 of the object has to agree with the type of the object presented
#                 as img.
# @param anchor  A tuple of two integer elements of x-, y-axis coordinates to place
#                 the overlaying image, ovly_img. The origin of the x-, y-pane is
#                 the upper left corner of the master for positive x-, y-component
#                 values with first pixel as +1. The bound of the x-, y-coordinates
#                 to be presented is limited to the width and height of the master
#                 image, img, in the number of pixels. The function accepts negative
#                 numbers provided to be x-, y-coordinate components, signifying the
#                 desire for such individual component coordinate to be considered
#                 from a new origin that is from opposite edge of the corresponding
#                 axis. E.g., A coordinate of (-1, 20) denotes the first pixel from
#                 the right along the x-axis and the twentieth pixel from the top
#                 along the y-axis. The top left corner pixel of the overlaying
#                 image will be on the coordinate dictated by this anchor parameter.
# @param paste   A boolean variable to dictate the function to skip removing the
#                 background from the overlaying image. Paste directly the raw
#                 overlaying image to the background image.
#
# @ret           A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 with required overlay. The type of object returned would be the
#                 same as the img object type.
#
def overlay_image(img, ovly_img, anchor = (1, 1), paste = False) :
    if img is not None and ovly_img is not None and type(img) is type(ovly_img) :
        pass
    else :
        raise TypeError

    if img is not None and isinstance(img, numpy.ndarray) is True :
        img = copy.copy(img)
    elif img is not None and isinstance(img, Image.Image) is True :
        pilimg = copy.copy(img)
        ovly_pilimg = ovly_img
    elif img is not None and type(img) is io.BytesIO :
        img.seek(0)

        pilimg = Image.open(img)

        ovly_img.seek(0)

        ovly_pilimg = Image.open(ovly_img)
    else :
        raise TypeError

    if paste is not None and type(paste) is bool :
        pass
    else :
        raise TypeError

    if anchor is not None and (type(anchor) is tuple or type(anchor) is list) and len(anchor) is 2 :
        if isinstance(img, numpy.ndarray) is True :
            if img.shape[0] > ovly_img.shape[0] and img.shape[1] > ovly_img.shape[1] :
                pass
            else :
                raise IndexError("Overlay Image is Larger than the Canvas of the Base Image")

            if anchor[0] - ovly_img.shape[1] + 1 < -img.shape[1] or anchor[0] + ovly_img.shape[1] - 1 > img.shape[1] or anchor[0] == 0 :
                raise ValueError
            else :
                pass

            if anchor[1] - ovly_img.shape[0] + 1 < -img.shape[0] or anchor[1] + ovly_img.shape[0] - 1 > img.shape[0] or anchor[1] == 0 :
                raise ValueError
            else :
                pass

            if anchor[0] > 0 :
                _ = anchor[0] - 1
            else :
                _ = img.shape[1] + anchor[0] - (ovly_img.shape[1] - 1)

            if anchor[1] > 0 :
                anchor = (_, anchor[1] - 1)
            else :
                anchor = (_, img.shape[0] + anchor[1] - (ovly_img.shape[0] - 1))

            if paste is not True :
                ovly_cnvs = img[anchor[1] : anchor[1] + ovly_img.shape[0], anchor[0] : anchor[0] + ovly_img.shape[1]]
                _, ovly_mask = cv2.threshold(cv2.cvtColor(ovly_img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

                img[anchor[1] : anchor[1] + ovly_img.shape[0], anchor[0] : anchor[0] + ovly_img.shape[1]] = \
                    cv2.add(cv2.bitwise_and(ovly_cnvs, ovly_cnvs, mask = ovly_mask), \
                    cv2.bitwise_and(ovly_img, ovly_img, mask = cv2.bitwise_not(ovly_mask)))
            else :
                img[anchor[1] : anchor[1] + ovly_img.shape[0], anchor[0] : anchor[0] + ovly_img.shape[1]] = ovly_img
        else :
            if pilimg.size[0] > ovly_pilimg.size[0] and pilimg.size[1] > ovly_pilimg.size[1] :
                pass
            else :
                raise IndexError("Overlay Image is Larger than the Canvas of the Base Image")

            if anchor[0] - ovly_pilimg.size[0] + 1 < -pilimg.size[0] or anchor[0] + ovly_pilimg.size[0] - 1 > pilimg.size[0] or anchor[0] == 0 :
                raise ValueError
            else :
                pass

            if anchor[1] - ovly_pilimg.size[1] + 1 < -pilimg.size[1] or anchor[1] + ovly_pilimg.size[1] - 1 > pilimg.size[1] or anchor[1] == 0 :
                raise ValueError
            else :
                pass

            if anchor[0] > 0 :
                _ = anchor[0] - 1
            else :
                _ = pilimg.size[0] + anchor[0] - (ovly_pilimg.size[0] - 1)

            if anchor[1] > 0 :
                anchor = (_, anchor[1] - 1)
            else :
                anchor = (_, pilimg.size[1] + anchor[1] - (ovly_pilimg.size[1] - 1))

            if paste is not True :
                pilimg.paste(ovly_pilimg, anchor, ImageOps.invert(ovly_pilimg).convert("1", dither=0))
            else :
                pilimg.paste(ovly_pilimg, anchor)
    else :
        raise TypeError

    if type(img) is io.BytesIO :
        img.seek(0)

        ret = io.BytesIO()

        pilimg.save(ret, "JPEG")

        return ret
    elif type(img) is numpy.ndarray :
        return img
    else :
        return pilimg

# OVERLAY_TEXT
# ====================
#
# This function overlays text marking on the given image.
#
# @param img     A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 presented for placing text on top. For NumPy Array object, OpenCV
#                 drawing is used instead of PIL ImageDraw routines.
# @param ovly_txtA string object containing the text to be glued on the master.
# @param anchor  A tuple of two integer elements of x-, y-axis coordinates to place
#                 the overlaying image, ovly_img. The origin of the x-, y-pane is
#                 the upper left corner of the master for positive x-, y-component
#                 values with first pixel as +1. The bound of the x-, y-coordinates
#                 to be presented is limited to the width and height of the master
#                 image, img, in the number of pixels. The function accepts negative
#                 numbers provided to be x-, y-coordinate components, signifying the
#                 desire for such individual component coordinate to be considered
#                 from a new origin that is from opposite edge of the corresponding
#                 axis. E.g., A coordinate of (-1, 20) denotes the first pixel from
#                 the right along the x-axis and the twentieth pixel from the top
#                 along the y-axis. The top left corner pixel of the overlaying text
#                 will be on the coordinate dictated by this anchor parameter.
# @param colour  A tuple of three 8-bit integer elements of individual component
#                 colour Red, Green and Blue that represents the colour of the text
#                 or if label parameter, label is passed True, represents the colour
#                 intended as the label colour contrasting the text. The text colour
#                 if label option is enabled, will be either white or black that is
#                 automcatically determined by the function.
# @param label   A boolean value that dictates whether a text box with background
#                 colour supporting the text should be used or just raw text should
#                 be stuck on the master image.
#
# @ret           A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 with required overlay. The type of object returned would be the
#                 same as the img object type.
#
def overlay_text(img, ovly_txt, anchor = (1, 1), colour = (255, 0, 0), label = False) :
    if img is not None and isinstance(img, Image.Image) is True :
        pilimg = copy.copy(img)

        draw = ImageDraw.Draw(pilimg)
    elif img is not None and type(img) is io.BytesIO :
        img.seek(0)

        pilimg = Image.open(img)

        draw = ImageDraw.Draw(pilimg)
    elif img is not None and isinstance(img, numpy.ndarray) is True :
        img = copy.copy(img)
    else :
        raise TypeError

    # Specify the text font style and size to be used for labelling descriptions/
    #  taggings in the picture.
    #
    if isinstance(img, numpy.ndarray) is not True :
        font = ImageFont.truetype(font="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size=24)
    else :
        font = (cv2.FONT_HERSHEY_SIMPLEX, 0.6)

    if colour is not None and (type(colour) is tuple or type(colour) is list) and len(colour) is 3 :
        for i in 0, 1, 2 :
            if colour[i] < 0 or colour[i] > 255 :
                raise ValueError
            else :
                pass

        if (colour[0] * 0.299 + colour[1] * 0.587 + colour[2] * 0.114) / 255 > 0.5 :
            txtclr = (0, 0, 0)
        else :
            txtclr = (255, 255, 255)
    else :
        raise TypeError

    if anchor is not None and (type(anchor) is tuple or type(anchor) is list) and len(anchor) is 2 :
        if isinstance(img, numpy.ndarray) is True :
            lbl_size = cv2.getTextSize(ovly_txt, font[0], font[1], 1)[0]

            if anchor[0] > 0 :
                _ = anchor[0] - 1
            else :
                if label is True :
                    _ = img.shape[1] + anchor[0] - (lbl_size[0] + LINEWIDTH - 1)
                else :
                    _ = img.shape[1] + anchor[0] - (lbl_size[0] - 1)

            if anchor[1] > 0 :
                if label is True :
                    location = (_, anchor[1] - 1 + lbl_size[1] + LINEWIDTH - 1)
                else :
                    location = (_, anchor[1] - 1 + lbl_size[1] - 1)
            else :
                location = (_, img.shape[0] + anchor[1])

            if label is not None and label is True :
                boundry = (location[0] + lbl_size[0] + LINEWIDTH - 1, location[1] - lbl_size[1] - LINEWIDTH + 1)

                cv2.rectangle(img, location, boundry, colour, -1)

                # The bottom-left corner of an object in OpenCV is reference point to the anchorage, i.e., min(x), max(y).
                #
                cv2.putText(img, ovly_txt, (int(location[0] + LINEWIDTH / 2), int(location[1] - LINEWIDTH / 2)), font[0], font[1], txtclr)
            else :
                cv2.putText(img, ovly_txt, location, font[0], font[1], colour)
        else :
            lbl_size = draw.textsize(ovly_txt, font=font)

            if anchor[0] > 0 :
                _ = anchor[0] - 1
            else :
                if label is True :
                    _ = pilimg.size[0] + anchor[0] - (lbl_size[0] + LINEWIDTH - 1)
                else :
                    _ = pilimg.size[0] + anchor[0] - (lbl_size[0] - 1)

            if anchor[1] > 0 :
                if label is True :
                    location = (_, anchor[1] - 1 + lbl_size[1] + LINEWIDTH - 1)
                else :
                    location = (_, anchor[1] - 1 + lbl_size[1] - 1)
            else :
                location = (_, pilimg.size[1] + anchor[1])

            boundry = (location[0] + lbl_size[0] + LINEWIDTH - 1, location[1] - lbl_size[1] - LINEWIDTH + 1)

            if label is not None and label is True :
                draw.rectangle([location, boundry], fill=colour)

                # The top-left corner of an object in PIL is reference point to the anchorage, i.e., min(x), min(y).
                #
                draw.text((location[0] + LINEWIDTH / 2, boundry[1] + LINEWIDTH / 2), ovly_txt, font=font, fill=txtclr)
            else :
                draw.text((location[0], boundry[1] + LINEWIDTH), ovly_txt, font=font, fill=colour)
    else :
        raise TypeError

    if type(img) is io.BytesIO :
        img.seek(0)

        ret = io.BytesIO()

        pilimg.save(ret, "JPEG")

        return ret
    elif type(img) is numpy.ndarray :
        return img
    else :
        return pilimg

#
# (END OF) IMAGE RENDERING ROUTINES
#

#
# IMAGE PRESENTATION ROUTINES
#

# __GENERIC_OIMAGE
# ====================
#
# Generic routine for handling image output to filesystem or display system.
#
# @param img     A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 to be output
# @param ofile   None or a string of file location path for saving a JPEG image.
#                 If the argument was passed with None, it is assumed the image
#                 should be shown instead of stored in the filesystem and be
#                 presented to user according to additional argument static_ofile
# @param static  A boolean variable dictating when ofile is None, whether the image
#         _ofile  should be presented to user as image (static) or video.
#
def __generic_oimage(img, ofile = None, static_ofile = True) :
    if img is not None and isinstance(img, Image.Image) is True :
        pass
    elif img is not None and isinstance(img, numpy.ndarray) is True :
        pass
    elif img is not None and isinstance(img, io.BytesIO) is True :
        pass
    else :
        raise TypeError

    if static_ofile is not None and type(static_ofile) is bool :
        pass
    else :
        raise TypeError

    if ofile is None :
        if static_ofile is False :
            if isinstance(img, Image.Image) is True :
                cv2.imshow('Video', numpy.swapaxes(numpy.array(img.convert('RGB'), dtype=numpy.uint8).reshape((img.size[1], img.size[0], 3)), 0, 0)[:, :, ::-1])
            elif isinstance(img, numpy.ndarray) is True :
                cv2.imshow('Video', img[:, :, ::-1])
            elif isinstance(img, io.BytesIO) is True :
                img.seek(0)

                pilimg = Image.open(img)

                cv2.imshow('Video', numpy.swapaxes(numpy.array(pilimg.convert('RGB'), dtype=numpy.uint8).reshape((pilimg.size[1], pilimg.size[0], 3)), 0, 0)[:, :, ::-1])

            # [REQUIRED] https://stackoverflow.com/a/22277285
            cv2.waitKey(1)
        else :
            if isinstance(img, Image.Image) is True :
                img.show(title = "Transient Processed Output")
            elif isinstance(img, numpy.ndarray) is True :
                Image.fromarray(img).show()
            elif isinstance(img, io.BytesIO) is True :
                img.seek(0)

                Image.open(img).show()
    elif ofile is not None and type(ofile) is str :
        if isinstance(img, Image.Image) is True :
            img.save(ofile)
        elif isinstance(img, numpy.ndarray) is True :
            Image.fromarray(img).save(ofile)
        elif isinstance(img, io.BytesIO) is True :
            img.seek(0)

            Image.open(img).save(ofile)
    else :
        raise TypeError

# REPLAY_VIDEO
# ====================
#
# Routine for outputting a single video frame to the video window in graphical
#  user interface.
#
# @param frame   A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 to be output to graphical user interface as a constituent frame
#                 in the form of video
#
def replay_video(frame = None) :
    __generic_oimage(frame, None, False)

# REPLAY_IMAGE
# ====================
#
# Routine for outputting a picture in graphical user interface.
#
# @param image   A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 to be output to graphical user interface in the form of picture
#
def replay_image(image = None) :
    __generic_oimage(image, None, True)

# SAVE_IMAGE
# ====================
#
# Storing an image output to local filesystem.
#
# @param image   A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 to be output
# @param ofile   A string of file location path for saving a JPEG image.
#
def save_image(image, ofile) :
    __generic_oimage(image, ofile)

#
# (END OF) IMAGE PRESENTATION ROUTINES
#

#
# IMAGE ACQUISITION ROUTINES
#

import picamera, math

__cam_inst = None
__usenumpy = not False
__intl_mem = False

# __GENERIC_IIMAGE
# ====================
#
# Generic routine for handling image input from filesystem or image acquisition device.
#  There is no particular difference in acquiring an image input as single image or
#  as a single frame in a video.
#
# @param ifile   None or a string of file location path for retrieving a JPEG image.
#                 If the argument was passed with None, it is assumed the image
#                 should be acquired by local camera and the mode of capture as
#                 a single frame of video or picture should be indicated by the
#                 extra argument static_ifile
# @param static  A boolean variable dictating when ifile is None, whether the image
#         _ifile  should be acquired as image (static) or as a frame in a video.
#
# @ret           A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 captured. The exact object type returned is to be determined by
#                 module internal variable __usenumpy and __intl_mem. __usenumpy
#                 dictates the use of using NumPy Array for image storage and the
#                 flag takes precedence. The flag __intl_mem indicates the use of
#                 in-memory BytesIO object as image container. If both flags are
#                 not asserted, the return object would be a PIL image object.
#
def __generic_iimage(ifile = None, static_ifile = True) :
    if ifile is not None and type(ifile) is str :
        if __usenumpy is False :
            if __intl_mem is False :
                return Image.open(ifile).convert(mode = "RGB")
            else :
                ret = io.BytesIO()

                ret.write(open(ifile, "rb").read())

                return ret
        else :
            img = Image.open(ifile)

            ret = numpy.swapaxes(numpy.array(img.convert('RGB'), dtype=numpy.uint8).reshape((img.size[1], img.size[0], 3)), 0, 0)

            return ret
    elif ifile is None :
        global __cam_inst

        if static_ifile is not None and type(static_ifile) is bool :
            if __cam_inst is None :
                __cam_inst = picamera.PiCamera()
                __cam_inst.resolution = (int(CAMH / SCALE), int(CAMV / SCALE))

            if __usenumpy is False :
                # NO DIFFERENCE IN PERFORMANCE USING BYTESIO VERSUS MKTIME BY EXPERIMENT
                #  FOR RESOLUTION OF (480, 272). PERFORMANCE IMPACT FOR USING BYTESIO FOR
                #  RESOLUTION OF (1988, 1020)
                #
                if __intl_mem is False :
                    ret = tempfile.mktemp()

                    # use_video_port = True can make 10 times frame rate boost
                    # (site: https://pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/)
                    #
                    __cam_inst.capture(ret, format="jpeg", use_video_port = (static_ifile is False))

                    return Image.open(ret).convert(mode = "RGB")
                else :
                    ret = io.BytesIO()

                    __cam_inst.capture(ret, format="jpeg", use_video_port = (static_ifile is False))

                return ret
            else :
                # [WARNING] Camera resolution is aligned to 16 pixel and is automatically uplifted by the resolution set step.
                #  If the container is not uplifted to equivalent size, segmentation fault will occur.
                RESOALIGN = 16

                ret = numpy.empty((int(math.ceil(CAMV / SCALE / RESOALIGN) * RESOALIGN), int(math.ceil(CAMH / SCALE / RESOALIGN) * RESOALIGN), 3), dtype=numpy.uint8)

                __cam_inst.capture(ret, format="rgb", use_video_port = (static_ifile is False))

                return ret
        else :
            raise TypeError

# RECORD_VIDEO
# ====================
#
# Routine for capturing a single video frame of a video stream by the local
#  camera.
#
# @ret           A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 captured by the local camera as a constituent frame in a video
#                 stream.  The exact object type returned is to be determined by
#                 module internal variable __usenumpy and __intl_mem. __usenumpy
#                 dictates the use of using NumPy Array for image storage and the
#                 flag takes precedence. The flag __intl_mem indicates the use of
#                 in-memory BytesIO object as image container. If both flags are
#                 not asserted, the return object would be a PIL image object.
#
def record_video() :
    return __generic_iimage(None, False)

# RECORD_IMAGE
#
# ====================
#
# Routine for capturing a picture by the local camera.
#
# @ret           A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 captured by the local camera as a picture. The exact object type
#                 returned is to be determined by module internal variable __usenumpy
#                 and __intl_mem. __usenumpy dictates the use of using NumPy Array
#                 for image storage and the flag takes precedence. The flag
#                 __intl_mem indicates the use of in-memory BytesIO object as image
#                 container. If both flags are not asserted, the return object would
#                 be a PIL image object.
#
def record_image() :
    return __generic_iimage(None, True)

# LOAD_IMAGE
# ====================
#
# Retrieving an image from the local filesystem.
#
# @param ifile   A string of file location path for retrieving a JPEG image.
#
# @ret           A PIL image object or NumPy Array object containing RGB format
#                 image or a BytesIO streaming object containing JPEG format image
#                 loaded from the local filesystem.  The exact object type returned is
#                 to be determined by module internal variable __usenumpy and
#                 __intl_mem. __usenumpy dictates the use of using NumPy Array for
#                 image storage and the flag takes precedence. The flag __intl_mem
#                 indicates the use of in-memory BytesIO object as image container. If
#                 both flags are not asserted, the return object would be a PIL image
#                 object.
#
def load_image(ifile) :
    return __generic_iimage(ifile)

#
# (END OF) IMAGE ACQUISITION ROUTINES
#

#
# HARDWARE MANIPULATION ROUTINES
#

def resolution_rescale(factor = 4) :
    global SCALE
    global __cam_inst

    if factor is not None and type(factor) is int :
        if factor > 0 :
            SCALE = factor

            if __cam_inst is not None :
                __cam_inst.resolution = (int(CAMH / SCALE), int(CAMV / SCALE))
        else :
            raise ValueError
    else :
        raise TypeError

#
# (END OF) HARDWARE MANIPULATION ROUTINES
#

#
# USER ROUTINES
#

# MAIN
# ====================
#
# Main demonstration routine called when the module is ran in standalone mode.
#
# The default showcase routine streams reatime captured images in form of video and
#  overlay the image with TELI logo and current time.
#
if __name__ == "__main__" :
    import os, time, signal

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

    # __exit
    # ====================
    #
    # Perform clean up tasks upon receiving termination signals and exit
    #  gracefully.
    #
    # @param signum  A signal.Signals enums of the signal dispatched to current
    #                 process and passed to this handler
    # @param curstackCurrent stack frame (Not usually used)
    #
    def __exit(signum, curstack) :
        quit()

    signal.signal(signal.SIGINT, __exit)

    SCALE = 2

    agg = 0
    cnt = 0
    ent = time.time()

    while True :
        # Base layer (Master) of the video
        #
        layer_base = record_video()

        # Layer with text overlayed on the previous frame layer, i.e., Base Layer.
        #
        layer_text = overlay_text(layer_base, time.strftime("%H:%M"), anchor = (20, 20), colour = (255, 255, 255))

        # Layer with an. image overlayed on the previous frame layers, i.e., Text Layer
        #. + Base Layer.
        #
        layer_logo = overlay_image(layer_text, load_image(BASE_PATH + "/TELI_Logo-24px.jpg"), anchor = (-20, 5))

        # Leading the finished frame with all overlays to output display.
        #
        replay_video(layer_logo)

        cnt += 1

        end = time.time()
        dlt = end - ent
        agg += dlt

        print("Frame Rate: " + str(1 / dlt) + " FPS (Average: " + str(1 / (agg / cnt)) + " FPS)")

        ent = time.time()

    cv2.destroyAllWindows()
#
# (END OF) USER ROUTINES
#