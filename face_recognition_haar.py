import cv2
import matplotlib.pyplot as plt
import time
import sys


class FaceRecognition(object):
    def __init__(self, method=None):
        """
        :param method: 'haar' or 'lbp'
        """
        self.method = 'haar'
        if method:
            self.method = method
        self.scaleFactor = 1.1
        self.minNeighbors = 5

    @staticmethod
    def __convertToRGB(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __find_faces(self, img_gray):
        if self.method == 'haar':
            face_cascade = cv2.CascadeClassifier(
                '/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
        elif self.method == 'lbp':
            face_cascade = cv2.CascadeClassifier(
                '/usr/local/opt/opencv/share/OpenCV/lbpcascades/lbpcascade_frontalface.xml')
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors)
        return faces

    def __mark_faces(self, img, faces):
        marked_im = img
        for (x, y, w, h) in faces:
            cv2.rectangle(marked_im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return marked_im

    def detect_face(self, img):
        img_copy = img
        gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

        faces = self.__find_faces(gray_img)
        marked = self.__mark_faces(img_copy, faces)
        return marked
        return self.__convertToRGB(marked)


def display_gray(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def display_RGB(img):
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    img = cv2.imread('alba.jpg')
    fr = FaceRecognition(img)
    marked = fr.detect_face()
    display_RGB(marked)
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# faces = find_faces(gray_img)
# marked = mark_faces(faces, img)
# display_RGB(convertToRGB(marked))
