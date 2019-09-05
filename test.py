import cv2
import numpy
from googletrans import Translator

translator = Translator()
print(type(translator.translate('Hello World!', 'es', 'en')))


