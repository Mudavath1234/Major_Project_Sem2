import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
from math import dist
from tracker import *

model = YOLO('yolov8s.pt') 
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('TMS')
cv2.setMouseCallback('TMS', RGB)

# read Video
cap = cv2.VideoCapture('Videos/video1.mp4')

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 output
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1020, 500))  # Save as .mp4 file


my_file = open("coco.txt", "r")  # Class File
data = my_file.read()
class_list = data.split("\n")

count = 0
area = [(314, 297), (742, 297), (805, 323), (248, 323)]  # Starting area (entry)
area2 = [(171, 359), (890, 359), (1019, 422), (15, 422)]  # Destination area (exit)

area_c = set()
tracker = Tracker()
speed_limit = 120

# Vehicles tracking data
vehicles_entering = {}
vehicles_elapsed_time = {}
vehicles_entering_backward = {}

# Forward and Backward vehicle counts
forward_vehicle_count = 0
backward_vehicle_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))  # [Modified] -- Resizing frame

    results = model.predict(frame, verbose=False)  # [Modified] -- Turn off extra printing
    a = results[0].boxes.data

    px = pd.DataFrame(a).astype("float")

    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c or 'motorcycle' in c or 'truck' in c or 'bus' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        results = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        results2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)

        if results >= 0:
            if id not in vehicles_entering_backward:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX,
                            0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                Init_time = time.time()

                if id not in vehicles_entering:
                    vehicles_entering[id] = Init_time
            else:
                try:
                    elapsed_time = time.time() - vehicles_entering_backward[id]
                except KeyError:
                    pass

                if id not in vehicles_elapsed_time:
                    vehicles_elapsed_time[id] = elapsed_time
                else:
                    try:
                        elapsed_time = vehicles_elapsed_time[id]
                        dist_val = 45
                        speed_KH = (dist_val / elapsed_time) * 3.6
                        cv2.putText(frame, str(int(speed_KH)) + 'Km/h', (x4, y4),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                        if speed_KH >= speed_limit:
                            cv2.putText(frame, "Speed limit violated!", (440, 112),
                                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(frame, 'Detected', (cx, cy),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    except ZeroDivisionError:
                        pass

        if results2 >= 0:
            if id not in vehicles_entering:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX,
                            0.8, (0, 255, 255), 2, cv2.LINE_AA)
                area_c.add(id)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                Init_time = time.time()

                if id not in vehicles_entering_backward:
                    vehicles_entering_backward[id] = Init_time
                forward_vehicle_count += 1  # Increment forward vehicle count
            else:
                try:
                    elapsed_time = time.time() - vehicles_entering[id]
                except KeyError:
                    pass

                if id not in vehicles_elapsed_time:
                    vehicles_elapsed_time[id] = elapsed_time
                else:
                    try:
                        elapsed_time = vehicles_elapsed_time[id]
                        dist_val = 45
                        speed_KH = (dist_val / elapsed_time) * 3.6
                        cv2.putText(frame, str(int(speed_KH)) + 'Km/h', (x4, y4),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                        if speed_KH >= speed_limit:
                            cv2.putText(frame, "Speed limit violated!", (440, 112),
                                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(frame, 'Detected', (cx, cy),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    except ZeroDivisionError:
                        pass
                backward_vehicle_count += 1  # Increment backward vehicle count

    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)

    cnt = len(area_c)
    cv2.putText(frame, ('Vehicle-Count:-') + str(cnt), (452, 50),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (102, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("TMS", frame)

    out.write(frame)  # [Modified] -- Save the current frame to output video

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()  # [Modified] -- Release the VideoWriter
cv2.destroyAllWindows()


# After processing is done

# Total vehicles counted
total_vehicles = len(area_c)

# Count of speed violated vehicles
violated_vehicles = 0

# Check in vehicles_elapsed_time for violations
for id, elapsed_time in vehicles_elapsed_time.items():
    try:
        dist = 45  # Distance between regions
        speed_KH = (dist / elapsed_time) * 3.6
        if speed_KH >= speed_limit:
            violated_vehicles += 1
    except ZeroDivisionError:
        pass

# Print the final summary
print("\n--- Traffic Monitoring Summary ---")
print(f"Total Vehicles Counted: {total_vehicles}")
print(f"Vehicles Violating Speed Limit: {violated_vehicles}")
print(f"Forward Vehicles Count: {forward_vehicle_count}")
print(f"Backward Vehicles Count: {backward_vehicle_count}")
print("\nAdvice to Speed Violators:")
print("⚠️  Please follow speed limits to ensure road safety!")
print("⚠️  High speed increases the risk of accidents. Drive responsibly.")
print("-----------------------------------\n")

# Save the summary to a text file
with open("summary.txt", "w", encoding='utf-8') as summary_file:
    summary_file.write("--- Traffic Monitoring Summary ---\n")
    summary_file.write(f"Total Vehicles Counted: {total_vehicles}\n")
    summary_file.write(f"Vehicles Violating Speed Limit: {violated_vehicles}\n")
    summary_file.write(f"Forward Vehicles Count: {forward_vehicle_count}\n")
    summary_file.write(f"Backward Vehicles Count: {backward_vehicle_count}\n")
    summary_file.write("\nAdvice to Speed Violators:\n")
    summary_file.write("⚠️  Please follow speed limits to ensure road safety!\n")
    summary_file.write("⚠️  High speed increases the risk of accidents. Drive responsibly.\n")

# Save the summary to a CSV file
summary_df = pd.DataFrame({
    'Total Vehicles Counted': [total_vehicles],
    'Vehicles Violating Speed Limit': [violated_vehicles],
    'Forward Vehicles Count': [forward_vehicle_count],
    'Backward Vehicles Count': [backward_vehicle_count]
})
summary_df.to_csv('summary.csv', index=False)


