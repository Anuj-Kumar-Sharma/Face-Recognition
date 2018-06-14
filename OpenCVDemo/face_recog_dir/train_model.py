import cv2
from OpenCVDemo.face_recog_dir.get_training_dataset import collect_dataset

images, labels, labels_dict = collect_dataset()

# Eigen faces recogniser
rec_eigen = cv2.face.EigenFaceRecognizer_create()
rec_eigen.train(images, labels)

# Fisher faces recogniser
rec_fisher = cv2.face.FisherFaceRecognizer_create()
rec_fisher.train(images, labels)

# LBPH face recogniser
rec_lbph = cv2.face.LBPHFaceRecognizer_create()
rec_lbph.train(images, labels)

print("trained successfully.")