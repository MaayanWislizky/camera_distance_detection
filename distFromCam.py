#!/usr/bin/env python3
#----------------------------------------------------------------------------
# Created By  : Maayan Wislizky
# Created Date: 28/8/21
# version ='1.0'
# ---------------------------------------------------------------------------
"""This script is for calculating "face from the camera" per each frame with a single camera.
This script is based on ML and trained to find faces (haarcascade_frontalface). For better results,
I'd use the calibrated stereo camera and Epipolar geometry to extract a disparity map.
for cases with a different target, we can use: Canny- for edges(may require filter), then,
find contours or methods like keypoint detection, local invariant descriptors, and keypoint matching to find the target.
"""
import cv2

# distance from camera to face when taking single image (cm)
head_to_cam_distance = 35

# width of face in the real world (cm)
face_width = 12.3

# colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# fontScale
fontScale = 0.5

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# face detector object
# https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
face_detector = cv2.CascadeClassifier(r"./haarcascade_frontalface_default.xml")

singleImage = cv2.imread(r"./WIN_20210828_21_20_56_Pro.jpg")
# focal length is calculated from triangle similarity
# F = (P x D) / W
def focalLength(measured_distance, width, width_in_image):
    # finding the focal length
    focal_length = (width_in_image * measured_distance) / width
    return focal_length


# estimated distance
def distanceCalc(focalLength, faceWidth, imageFaceWidth):
    distance = (faceWidth * focalLength) / imageFaceWidth

    return distance


# type the distance to the screen
def distToScreen(dist, frame):
    # draw line as background of text
    cv2.line(frame, (30, 30), (230, 30), RED, 32)
    cv2.line(frame, (30, 30), (230, 30), BLACK, 28)

    cv2.putText(
        frame, f"dist: {round(dist, 2)} CM", (30, 35), font,
        fontScale, color, thickness, cv2.LINE_AA)


def faceDetection(image):
    # init
    face_width = 0

    # converting color image ot gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detecting face in the image
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

    # looping through the face detected in the image
    # getting coordinates x, y , width and height
    for (x, y, h, w) in faces:
        # draw the rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)

        # getting face width in the pixels
        face_width = w

    # return the face width in pixel
    return face_width




# find the face width(pixels) in single image
single_img_face_width = faceDetection(singleImage)

# get the focal by calling "focalLength"
# face width in single image (pixels),
# distance from camera in single image(cm),
# face width(cm)
focal_length_found = focalLength(
    head_to_cam_distance, face_width, single_img_face_width)

print("focal length:", focal_length_found)


# initialize the camera object
cap = cv2.VideoCapture(0)


while True:

    # reading the frame from camera
    _, frame = cap.read()

    # calling face detection function to find
    # the width of face(pixels) for each frame
    face_width_in_frame = faceDetection(frame)

    # check if the face is not zero
    if face_width_in_frame != 0:
        # calculating the distance by calling function
        distance = distanceCalc(
            focal_length_found, face_width, face_width_in_frame)

        distToScreen(distance, frame)

    # show the frame on the screen
    cv2.imshow("frame", frame)

    # quit the program if you press 'q' on keyboard
    if cv2.waitKey(1) == ord("q"):
        break

# closing the camera
cap.release()

# closing all the windows that are opened
cv2.destroyAllWindows()
