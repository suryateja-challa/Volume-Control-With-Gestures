import cv2
import time
import numpy as np
import mediapipe as mp

wcam, hcam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

pTime = 0
cTime = 0
#mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)



    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                if id == 4 or id == 8:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mp.solutions.drawing_utils.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 25, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()