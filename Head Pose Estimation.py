import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.io import loadmat

# Initialize the Face Mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Function to load images and labels from a folder
def load_images_and_labels(folder):
    images = []
    labels = []
    
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            # Load the image
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)
            images.append(image)

            # Load the corresponding .mat file for labels
            mat_filename = filename.replace('.jpg', '.mat')
            mat_path = os.path.join(folder, mat_filename)
            
            if os.path.exists(mat_path):
                mat_data = loadmat(mat_path)
                # Assuming label is stored in a field called 'label'
                label = mat_data.get('label', None)
                if label is not None:
                    labels.append(label[0][0])  # Append the label value
                else:
                    labels.append(None)  # In case label is not found
            else:
                labels.append(None)  # If no .mat file exists

    return images, labels

# Folder path containing images and .mat files
folder_path = r'task ml\AFLW2000'

# Load images and labels
images, labels = load_images_and_labels(folder_path)

# Function to calculate yaw, pitch, and roll based on key landmarks
def calculate_angles(landmarks, image_width, image_height):
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    chin = landmarks[152]

    nose_x, nose_y = int(nose_tip.x * image_width), int(nose_tip.y * image_height)
    left_eye_x, left_eye_y = int(left_eye.x * image_width), int(left_eye.y * image_height)
    right_eye_x, right_eye_y = int(right_eye.x * image_width), int(right_eye.y * image_height)
    chin_x, chin_y = int(chin.x * image_width), int(chin.y * image_height)

    yaw = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x) * 180 / np.pi
    pitch = np.arctan2(chin_y - nose_y, chin_x - nose_x) * 180 / np.pi
    roll = np.arctan2(left_eye_y - right_eye_y, left_eye_x - right_eye_x) * 180 / np.pi

    return yaw, pitch, roll

# Function to draw vectors on image
def draw_vector(image, start, angle, length, color, thickness=2):
    end_x = int(start[0] + length * np.cos(np.deg2rad(angle)))
    end_y = int(start[1] - length * np.sin(np.deg2rad(angle)))
    cv2.arrowedLine(image, start, (end_x, end_y), color, thickness)

# Loop through each image
for idx, image in enumerate(images):
    if image is None:
        print(f"Could not read image at index {idx}")
        continue

    # Convert the image to RGB as required by Mediapipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face mesh detection
    results = face_mesh.process(image_rgb)

    # Draw the face mesh landmarks if detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = image.shape
            landmarks = face_landmarks.landmark

            # Calculate yaw, pitch, and roll
            yaw, pitch, roll = calculate_angles(landmarks, iw, ih)

            # Calculate center of the face
            nose_tip = landmarks[1]
            cx, cy = int(nose_tip.x * iw), int(nose_tip.y * ih)

            # Define vector length
            vector_length = 100

            # Draw vectors
            draw_vector(image, (cx, cy), yaw, vector_length, (0, 0, 255))  # Red for yaw
            draw_vector(image, (cx, cy), pitch, vector_length, (0, 255, 0))  # Green for pitch
            draw_vector(image, (cx, cy), roll, vector_length, (255, 0, 0))  # Blue for roll

            # Print yaw, pitch, and roll
            print(f"Image {idx}: Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}")

        # Show the image with vectors
        cv2.imshow(f'Image {idx} - Face Mesh with Vectors', image)
        cv2.waitKey(0)  # Wait for key press to continue to the next image
        cv2.destroyAllWindows()



















# ==================================================================================================================
# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# from scipy.io import loadmat

# # Initialize the Face Mesh module
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh()

# # Function to load images and labels from a folder
# def load_images_and_labels(folder):
#     images = []
#     labels = []
    
#     for filename in os.listdir(folder):
#         if filename.endswith('.jpg'):
#             # Load the image
#             image_path = os.path.join(folder, filename)
#             image = cv2.imread(image_path)
#             images.append(image)

#             # Load the corresponding .mat file for labels
#             mat_filename = filename.replace('.jpg', '.mat')
#             mat_path = os.path.join(folder, mat_filename)
            
#             if os.path.exists(mat_path):
#                 mat_data = loadmat(mat_path)
#                 # Assuming label is stored in a field called 'label'
#                 label = mat_data.get('label', None)
#                 if label is not None:
#                     labels.append(label[0][0])  # Append the label value
#                 else:
#                     labels.append(None)  # In case label is not found
#             else:
#                 labels.append(None)  # If no .mat file exists

#     return images, labels

# # Folder path containing images and .mat files
# folder_path = r'task ml\AFLW2000'

# # Load images and labels
# images, labels = load_images_and_labels(folder_path)

# # Function to calculate yaw, pitch, and roll based on key landmarks
# def calculate_angles(landmarks, image_width, image_height):
#     nose_tip = landmarks[1]
#     left_eye = landmarks[33]
#     right_eye = landmarks[263]
#     chin = landmarks[152]

#     nose_x, nose_y = int(nose_tip.x * image_width), int(nose_tip.y * image_height)
#     left_eye_x, left_eye_y = int(left_eye.x * image_width), int(left_eye.y * image_height)
#     right_eye_x, right_eye_y = int(right_eye.x * image_width), int(right_eye.y * image_height)
#     chin_x, chin_y = int(chin.x * image_width), int(chin.y * image_height)

#     yaw = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x) * 180 / np.pi
#     pitch = np.arctan2(chin_y - nose_y, chin_x - nose_x) * 180 / np.pi
#     roll = np.arctan2(left_eye_y - right_eye_y, left_eye_x - right_eye_x) * 180 / np.pi

#     return yaw, pitch, roll

# # Loop through each image
# for idx, image in enumerate(images):
#     if image is None:
#         print(f"Could not read image at index {idx}")
#         continue

#     # Convert the image to RGB as required by Mediapipe
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Perform face mesh detection
#     results = face_mesh.process(image_rgb)

#     # Draw the face mesh landmarks if detected
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             ih, iw, _ = image.shape
#             landmarks = face_landmarks.landmark

#             # Calculate yaw, pitch, and roll
#             yaw, pitch, roll = calculate_angles(landmarks, iw, ih)

#             # Print yaw, pitch, and roll
#             print(f"Image {idx}: Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}")

#             # Display the image with the results
#             cv2.putText(image, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(image, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(image, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # Draw face landmarks
#             for landmark in face_landmarks.landmark:
#                 x, y = int(landmark.x * iw), int(landmark.y * ih)
#                 cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

#         # Show the image with landmarks
#         cv2.imshow(f'Image {idx} - Face Mesh with Yaw, Pitch, Roll', image)
#         cv2.waitKey(0)  # Wait for key press to continue to the next image
#         cv2.destroyAllWindows()


# ===============================================



# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize the Face Mesh module
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh()

# # Load the image
# image = cv2.imread('task ml\AFLW2000')

# # Convert the image to RGB because Mediapipe works with RGB images
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Perform face mesh detection
# results = face_mesh.process(image_rgb)

# # Function to calculate yaw, pitch, and roll based on key landmarks
# def calculate_angles(landmarks, image_width, image_height):
#     # Get key landmarks (nose tip and eyes)
#     nose_tip = landmarks[1]  # Nose tip landmark (index 1 in Face Mesh)
#     left_eye = landmarks[33]  # Left eye landmark (index 33)
#     right_eye = landmarks[263]  # Right eye landmark (index 263)
#     chin = landmarks[152]  # Chin landmark (index 152)

#     # Convert normalized landmarks to pixel coordinates
#     nose_x, nose_y = int(nose_tip.x * image_width), int(nose_tip.y * image_height)
#     left_eye_x, left_eye_y = int(left_eye.x * image_width), int(left_eye.y * image_height)
#     right_eye_x, right_eye_y = int(right_eye.x * image_width), int(right_eye.y * image_height)
#     chin_x, chin_y = int(chin.x * image_width), int(chin.y * image_height)

#     # Calculate Yaw (left-right rotation) based on the horizontal distance between the eyes
#     yaw = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x) * 180 / np.pi

#     # Calculate Pitch (up-down rotation) based on the distance between the nose and the chin
#     pitch = np.arctan2(chin_y - nose_y, chin_x - nose_x) * 180 / np.pi

#     # Calculate Roll (tilt) based on the vertical position of the eyes
#     roll = np.arctan2(left_eye_y - right_eye_y, left_eye_x - right_eye_x) * 180 / np.pi

#     return yaw, pitch, roll

# # Draw the face mesh landmarks if detected
# if results.multi_face_landmarks:
#     for face_landmarks in results.multi_face_landmarks:
#         # Convert landmarks to pixel coordinates
#         ih, iw, _ = image.shape
#         landmarks = face_landmarks.landmark

#         # Calculate yaw, pitch, and roll
#         yaw, pitch, roll = calculate_angles(landmarks, iw, ih)

#         # Print yaw, pitch, and roll to the terminal
#         print(f"\nYaw: {yaw:.2f},\nPitch: {pitch:.2f},\nRoll: {roll:.2f}")

#         # Display yaw, pitch, and roll on the image
#         cv2.putText(image, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.putText(image, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.putText(image, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Loop through the face landmarks and draw them (blue points)
#         for landmark in face_landmarks.landmark:
#             x, y = int(landmark.x * iw), int(landmark.y * ih)
#             cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Blue color (BGR format)

# # Display the image with face mesh landmarks and yaw/pitch/roll values
# cv2.imshow('Face Mesh with Yaw, Pitch, Roll', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
