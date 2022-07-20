import cv2
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

cap = cv2.VideoCapture(0)
interrupt = -1
minValue = 70

model = load_model("my_model")

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Coordinates of the handSquare
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])

    handSquare = frame[10:410, 220:520]

    frame = cv2.rectangle(frame, (220-1, 9), (620+1, 419), (255,0,0) ,1)
    
    size = 100
    
    handSquare_gray = cv2.cvtColor(handSquare, cv2.COLOR_BGR2GRAY)

    handSquare_gray = cv2.GaussianBlur(handSquare_gray,(5,5),2)
    # #blur = cv2.bilateralFilter(handSquare,9,75,75)
    
    handSquare_gray = cv2.adaptiveThreshold(handSquare_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, handSquare_gray = cv2.threshold(handSquare_gray, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    test_image = cv2.resize(handSquare_gray, (size, size))
    img_array = image.img_to_array(test_image)
    
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch)
    predicted = np.argmax(predictions, axis=1)
    # print("\n\n *** ", predictions)
    print("\n---- ", chr(int(65 + predicted[0])))
    
    cv2.imshow("Frame", frame)
    cv2.imshow("handSquare", cv2.resize(handSquare_gray, (400, 400)))
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()


