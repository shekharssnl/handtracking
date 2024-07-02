import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Custom drawing specifications
landmark_drawing_spec = mpDraw.DrawingSpec(color=(255, 55, 255), thickness=2, circle_radius=2)
connection_drawing_spec = mpDraw.DrawingSpec(color=(2, 255, 2), thickness=1, circle_radius=2)  # White connections

cTime = 0
pTime = 0


while(True):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                #print(id,lm)
                h,w,c = img.shape
                cx,cy =int(lm.x * w), int(lm.y * h)
                print(id,cx,cy)
                #if id ==4:
                    #cv2.circle(img, (cx,cy), 20,(255,0,255),cv2.FILLED)
                #cv2.putText(img, str(id), (cx,cy), cv2.FONT_HERSHEY_PLAIN, 2,
                                #(255, 0, 255), 3, )

            mpDraw.draw_landmarks(img , handLms , mpHands.HAND_CONNECTIONS
                                  ,landmark_drawing_spec,connection_drawing_spec)

    cTime = time.time()
    fps= 1/(cTime - pTime)
    pTime =cTime

    cv2.putText(img, str(int(fps)) ,(10,70),cv2.FONT_HERSHEY_PLAIN,2,
                (255,0,255),3, )

    cv2.imshow("Image", img)
    cv2.waitKey(1)