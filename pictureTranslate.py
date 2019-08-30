import cv2
import numpy

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar = dict()
for i in range(1,6):
    cifar.update(unpickle("cifar-10-batches-py/data_batch_%d" % i))
#print(type(cifar[b'data']))
