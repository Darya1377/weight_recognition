from ultralytics import YOLO
import cv2
import time
from collections import Counter
import numpy
import os
import csv
from datetime import datetime, timedelta
from collections import defaultdict

def most_common_length(strings):
    length_counts = defaultdict(int)
    for s in strings:
        length = len(s)
        length_counts[length] += 1
    
    if not length_counts:
        return None  # or handle empty list case as needed
    
    max_count = max(length_counts.values())
    most_common_lengths = [length for length, count in length_counts.items() if count == max_count]
    
    return most_common_lengths[0]

start_time = time.time()

# Load your YOLOv8 model
model = YOLO(r"C:\Users\user\Desktop\weight_recognition\runs\detect\train3\weights\best.pt")  # replace with your model path

# Open video file
video_path = r"C:\Users\user\Desktop\weight_recognition\recordings\recording_20250617_073150.mp4" # replace with your video path
cap = cv2.VideoCapture(video_path)

# Get video properties for the output writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_path = "output_video_weight3.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'avc1' for H.264
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Define text display properties
font = cv2.FONT_HERSHEY_SIMPLEX 
font_scale = 0.7
font_thickness = 2
text_color = (0, 255, 0)  # Green color
text_position = (10, 30)  # Top-left position
weight_all = []
avg = 0
csv_filename = "detected_weight.csv"
file_exists = os.path.isfile(csv_filename)

with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Time', 'Weight']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if not file_exists:
        writer.writeheader()
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        
        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, conf = 0.65)
            
            # Get the first result (assuming single image inference)
            result = results[0]
            
            # Get bounding boxes, classes, and names
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            classes = result.boxes.cls.cpu().numpy()
            class_names = result.names
            
            # Combine boxes with their class names and sort from left to right
            detections = []
            for box, cls in zip(boxes, classes):
                x1 = box[0]  # left coordinate
                class_name = class_names[int(cls)]
                detections.append((x1, class_name))
            
            # Sort detections by x-coordinate (left to right)
            detections.sort(key=lambda x: x[0])
            
            # Extract just the class names in order
            sorted_class_names = [name for (x1, name) in detections]
            
            # Concatenate into a single string
            class_string = " ".join(sorted_class_names)
            weight = class_string.replace(' ', '')
            # Print the result for this frame to terminal
            
            
            # Get the annotated frame with bounding boxes
            annotated_frame = results[0].plot()
            # if len(weight)==1 or len(weight)==5:
            #     print(weight)
            #     # Add the class string to the frame
            #     cv2.putText(annotated_frame, f"Objects (L→R): {weight}", 
            #             text_position, font, font_scale, 
            #             text_color, font_thickness, cv2.LINE_AA)
            # print(type(weight))
            if len(weight) != 1:
                
                if len(weight)!=0:
                    weight_all.append(weight)
                    # print(weight_all)

            elif weight_all and avg == 0 and weight == '0':
                
                # print("AAAAAAAAAAAAAAAAAAA")
                max_length = most_common_length(weight_all)
                common_length_list = [num for num in weight_all if len(num) == max_length]
                # print(common_length_list)

                first_digits = [num[0] for num in common_length_list]
                digit_counts = Counter(first_digits)
                most_common_digit = digit_counts.most_common(1)[0][0]
                filtered_numbers = [num for num in common_length_list if num[0] == most_common_digit]
                filtered_numbers_int = [int(x) for x in filtered_numbers]
                avg = numpy.average(filtered_numbers_int)
                current_time_now = datetime.now()
                writer.writerow({
                        'Time': current_time_now.strftime("%Y-%m-%d %H:%M:%S"),
                        'Weight': int(avg)
                    })
                csvfile.flush()
                # current_time = time.time()
                # elapsed_time = current_time - start_time
                
                # if elapsed_time >= 20 and elapsed_time <= 140:
                #     print(weight)
                #     cv2.putText(annotated_frame, f"Objects (L->R): {weight}, {elapsed_time}", 
                #             text_position, font, font_scale, 
                #             text_color, font_thickness, cv2.LINE_AA)
                #     if len(weight) == 5:
                #         weight_all.append(weight)

                # elif elapsed_time > 140 and elapsed_time < 160 and weight_all:  # Check if weight_all is not empty
                #     if avg == 0:
                #         first_digits = [num[0] for num in weight_all]
                #         digit_counts = Counter(first_digits)
                #         most_common_digit = digit_counts.most_common(1)[0][0]
                #         filtered_numbers = [num for num in weight_all if num[0] == most_common_digit]
                #         filtered_numbers_int = [int(x) for x in filtered_numbers]
                #         avg = numpy.average(filtered_numbers_int)
                        
                #         current_time_now = datetime.now()
                #         writer.writerow({
                #                 'Time': current_time_now.strftime("%Y-%m-%d %H:%M:%S"),
                #                 'Weight': int(avg)
                #             })
                #         csvfile.flush()
            else:
                avg = 0
                start_time = time.time()
                weight_all = []
                    

            # break  # Exit the loop or perform an action
            # Write the frame to the output video
            out.write(annotated_frame)
            
            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output video saved to: {output_path}")