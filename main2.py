import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame,results.pose_landmarks,mpPose.POSE_CONNECTIONS,mpDraw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),mpDraw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))

    cv2.imshow("cam",frame)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
