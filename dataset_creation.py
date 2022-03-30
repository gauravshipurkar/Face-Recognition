import numpy as np
import cv2
from pathlib import Path

print(Path('.').absolute())

capture = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = capture.read()
    Image = frame
    if ret == True:
        Image = cv2.resize(Image, (700, 700))
        file_path = 'C:\Users\gaura\OneDrive\Desktop\Ongoing\Face\DataSet\Sample'
        cv2.imwrite(file_path, Image)
        cv2.putText(Image, str(count), (70, 70),
                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('DataSet Development', Image)
        count = count + 1

    if cv2.waitKey(2) & 0xFF == ord('q') or count == 200:
        break

capture.release()
cv2.destroyAllWindows()
