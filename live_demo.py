import cv2
from face_recognition_haar import FaceRecognition

cam = cv2.VideoCapture(0)

cv2.namedWindow("haar")

img_counter = 0

fr = FaceRecognition()
while True:
    ret, frame = cam.read()

    frame = fr.detect_face(frame)
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
