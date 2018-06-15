import os
import cv2

from OpenCVDemo.face_recog_dir.face_normalisation import get_normalised_faces

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('C:\\Users\\anuj5\AppData\Local\conda\conda\envs\\face_recog_env\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

number_of_images_for_training = 20

folder = "people/" + input('Name: ').lower()
if not os.path.exists(folder):
    os.mkdir(folder)
    counter = 0
    timer = 0

    while counter < number_of_images_for_training:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_coord = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for i, face in enumerate(faces_coord):

            if len(faces_coord) > 0 and timer % 100 == 0:
                if len(faces_coord) == 1:
                    cv2.rectangle(frame, (faces_coord[i][0], faces_coord[i][1]),
                                  (faces_coord[i][0] + faces_coord[i][2], faces_coord[i][1] + faces_coord[i][3]), (0, 255, 0), 1)
                    faces = get_normalised_faces(gray, faces_coord)
                    image_path = folder + "/img_" + str(counter) + ".jpg"
                    cv2.imwrite(image_path, faces[0])
                    print(counter, image_path)
                    counter += 1
                else:
                    cv2.rectangle(frame, (faces_coord[i][0], faces_coord[i][1]),
                                  (faces_coord[i][0] + faces_coord[i][2], faces_coord[i][1] + faces_coord[i][3]),
                                  (0, 0, 255), 1)

        cv2.imshow('frame', frame)
        cv2.waitKey(10)
        timer += 10
else:
    print("folder " + folder + " already exists")

cap.release()
cv2.destroyAllWindows()
