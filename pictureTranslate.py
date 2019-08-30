import cv2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar = dict()
for i in range(1,6):
    cifar.update(unpickle("cifar-10-batches-py/data_batch_%d" % i))
print(type(cifar[b'data']))
