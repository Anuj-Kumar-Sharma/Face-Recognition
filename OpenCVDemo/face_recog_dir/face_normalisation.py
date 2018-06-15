import cv2


def cut_face(image, face_coord):
    faces = []
    for (x, y, w, h) in face_coord:
        per = int(0.2 * w / 2)
        faces.append(image[y: y+h, x+per: x+w-per])
    return faces


def normalise_intensity(images):
    norm_image = []
    for image in images:
        if len(image) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        norm_image.append(cv2.equalizeHist(image))
    return norm_image


def resize(images, size=(128, 128)):
    norm_image = []
    for image in images:
        if image.shape < size:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        norm_image.append(image)
    return norm_image


def get_normalised_faces(frame, face_coord):
    faces = cut_face(frame, face_coord)
    faces = normalise_intensity(faces)
    faces = resize(faces)

    return faces
