import cv2
import numpy
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar = dict()
for i in range(1,6):
    cifar.update(unpickle("cifar-10-batches-py/data_batch_%d" % i))
#print(type(cifar[b'data']))

cam = cv2.VideoCapture(0)
cv2.namedWindow("frame")
img_counter = 0
img_list = []

while True:
    ret, frame = cam.read()
    cv2.imshow('frame', frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        img_list.append(img_name)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

for pic in img_list:
    img_name = "opencv_frame_{}.png".format(img_counter)
    im = Image.open(img_name)
    im.show()
