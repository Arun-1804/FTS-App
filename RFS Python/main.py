import cv2
import numpy as np
import os
import pandas as pd
from scipy.spatial.distance import euclidean
from imutils import perspective, contours
import imutils
import xlsxwriter



from flask import Flask, request, jsonify

app = Flask(__name__)



'''def resize_image(image, max_width=800, max_height=600):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_dimensions = (int(width * ratio), int(height * ratio))
        resized = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
        return resized
    return image'''

def pixel_to_cm(pixel_length, pixel_per_cm):
    return pixel_length / pixel_per_cm

def annotate_image(image, box, pixel_per_cm, contour_idx, crack_data, crack_name):
    width_cm = pixel_to_cm(euclidean(box[0], box[1]), pixel_per_cm)
    height_cm = pixel_to_cm(euclidean(box[1], box[2]), pixel_per_cm)

    M = cv2.moments(box)
    cX = int(M["m10"] / M["m00"]) if M["m00"] else 0
    cY = int(M["m01"] / M["m00"]) if M["m00"] else 0

    cv2.putText(image, f"{width_cm:.1f}cm x {height_cm:.1f}cm - {crack_name}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)

    crack_data.append([crack_name, width_cm, height_cm])

def process_video_and_save_data(video_path, output_folder, pixel_per_cm):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(os.path.join(output_folder, "annotated_video.avi"),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    crack_data = []
    frame_images = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        edged = cv2.Canny(blur, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = [x for x in cnts if cv2.contourArea(x) > 100]

        for i, cnt in enumerate(cnts):
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)

            # Assume crack names are in the format "Crack 1", "Crack 2", etc.
            crack_name = f"Crack {i + 1}"
            annotate_image(frame, box, pixel_per_cm, i + 1, crack_data, crack_name)

        annotated_frame = frame
        out.write(annotated_frame)

        frame_images.append(annotated_frame)

    cap.release()
    out.release()

    # Save crack dimensions to CSV
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_folder, f"{video_name}_crack_dimensions.csv")
    crack_df = pd.DataFrame(crack_data, columns=["Crack", "Length (cm)", "Width (cm)"])
    crack_df.to_csv(csv_path, index=False)

    # Create an Excel workbook and insert the data and frames
    workbook = xlsxwriter.Workbook(os.path.join(output_folder, f"{video_name}_data_and_frames.xlsx"))
    worksheet_data = workbook.add_worksheet("Crack Data")
    worksheet_images = workbook.add_worksheet("Frames")

    # Insert data into the workbook
    worksheet_data.write("A1", "Crack")
    worksheet_data.write("B1", "Length (cm)")
    worksheet_data.write("C1", "Width (cm)")

    for i, (crack, length, width) in enumerate(crack_data, start=1):
        worksheet_data.write(f"A{i + 1}", crack)
        worksheet_data.write(f"B{i + 1}", length)
        worksheet_data.write(f"C{i + 1}", width)

    # Insert frames into the workbook
    for i, frame_image in enumerate(frame_images):
        frame_path = os.path.join(output_folder, f"frame_{i}.png")
        cv2.imwrite(frame_path, frame_image)
        worksheet_images.insert_image(i, 0, frame_path, {'x_scale': 0.5, 'y_scale': 0.5})

    workbook.close()

    print(f"Data, annotated video, and frames saved for {video_name}.")


# Specify the paths and conversion ratio
video_path = r'http://192.168.251.58:8080/video' # Update with your video path
output_folder = r"D:\Buisness Apps\MERN\Vid"# Update with your output folder path
# video_path = r"C:\Desktop\PJT\Data Valdation\45cm video\crack5.mp4"# Update with your video path
# output_folder = r"C:\Desktop\PJT\Data Valdation\45cm\45cm video data crack-5"# Update with your output folder path
cap = cv2.VideoCapture(video_path)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
pixel_per_cm = width*height/(80*260) # Example conversion rate, adjust based on your reference
# process_video_and_save_data(video_path, output_folder , pixel_per_cm)




@app.route('/sendIp', methods=['GET', 'POST'])
def handle_request():
    # Extract JSON data from the request
    if request.method == 'GET':
        Ip = request.args.get('ip')
        process_video_and_save_data(Ip, output_folder , pixel_per_cm)
        response =  f'Started Detecting..'
        return jsonify(response)
    elif request.method == 'POST':
        data = request.get_json()
        response =  f'received a post request : {data}'
    # Process the data (this is just an example)
    # response = {
    #     'message': 'Data received successfully!',
    #     'received_data': data
    # }

    # Return a JSON response
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)



# URL of the video stream
# url = 'http://192.168.26.48:8080'
# url = r"C:\Users\bala1\movies\ProjectVideo\test.mp4"

# # Create a VideoCapture object
# cap = cv2.VideoCapture(url)

# if not cap.isOpened():
#     print('Error opening video stream')
# else:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print('Failed to grab frame')
#             break

#         # Display the resulting frame
#         cv2.imshow('Frame', frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the capture and close windows
#     cap.release()
#     cv2.destroyAllWindows()