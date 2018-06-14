import cv2

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('C:\\Users\\anuj5\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2'
                                    '\data\haarcascade_frontalface_alt2.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    count = 0
    for (x, y, w, h) in faces:
        roi_gray = gray[y: y+h, x: x+w]
        roi_color = frame[y: y+h, x: x+w]

        color = (255, 0, 0)
        stroke = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
        img = 'img_'+str(count)+".png"
        count += 1
        cv2.imwrite(img, roi_color)
        print(x, y, w, h)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()