import os,math
import warnings
import cv2
import torch
import numpy as np
from datetime import datetime
from torchvision import transforms
import tensorflow as tf
from flask import Flask, render_template, jsonify, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import mediapipe as mp
from dotenv import load_dotenv  # For environment variables
from collections import deque
from flask_cors import CORS


# Suppress TenpythoFlow and PyTorch warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Load environment variables
load_dotenv()
# Initialize deque with a fixed size
activity_window = deque(maxlen=10)  
# Flask app setup
app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'mp4,avi,mov').split(','))

# Ensure required folders exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists('static/fall_images'):
    os.makedirs('static/fall_images')

# Load YOLOv8 model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = YOLO('models/yolov8n-pose.pt')  # Use appropriate YOLOv8 model
model.to(device)

# Load CNN-LSTM model
cnn_lstm_model_path = os.path.join(os.getcwd(), 'Activity_detection_CNN_LSTM.h5')
cnn_lstm_model = tf.keras.models.load_model(cnn_lstm_model_path)
cnn_lstm_model.trainable = False

# Global variables
fall_status = "Not Fall"
activity_status = "Updating"
fall_image_path = None
timeline = []

# MediaPipe Pose Initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def smooth_activity(predicted_label):
    activity_window.append(predicted_label)
    # Return the most common activity in the window
    return max(set(activity_window), key=activity_window.count)  # Majority vote

def preprocess_frame(frame):
    # Apply Gaussian blur to smoothen the frame
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # Optional: Apply sharpening for deblurring
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_frame = cv2.filter2D(blurred_frame, -1, kernel)
    return sharpened_frame

def draw_annotations(frame, activity, bbox, fall_detected=False):
    """
    Draw bounding box and activity status on the video frame.
    Args:
        frame: The frame to annotate.
        activity: The smoothed activity to display.
        bbox: Tuple (x1, y1, x2, y2) for bounding box coordinates.
        fall_detected: Boolean indicating if a fall was detected.
    """
    x1, y1, x2, y2 = bbox
    color = (0, 0, 255) if fall_detected else (0, 255, 0)  # Red for fall, green otherwise
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Display activity
    text = f"Activity: {activity}"
    if fall_detected:
        text = "Fall Detected!"

    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def detect_activity_status(keypoints, prev_keypoints=None,frame_height=1.0):
    """
    Detect activity status based on pose keypoints.
    Args:
        keypoints: Dictionary of normalized keypoints with landmark names as keys and (x, y) tuples as values.
        prev_keypoints: Dictionary of keypoints from the previous frame for movement detection (optional).
        frame_height: Height of the frame to normalize distances (default is 1.0 for normalized keypoints).
    
    Returns:
        str: Detected activity ("Standing", "Sitting", "Walking", "Lying Down", or "Unknown").
    """
    def calculate_angle(keypoints, point1, point2, point3):
        """Calculate the angle between three landmarks."""
        value = keypoints.get(point1)
        if isinstance(value, (tuple, list)) and len(value) >= 2:
        # Extract x and y coordinates if value is a tuple or list with length 2
            x1,y1 = value[0],value[1]
            
        value = keypoints.get(point2)
        if isinstance(value, (tuple, list)) and len(value) >= 2:
        # Extract x and y coordinates if value is a tuple or list with length 2
            x2, y2 = value[0],value[1]   
            print(x2, y2)
        value = keypoints.get(point3)
        if isinstance(value, (tuple, list)) and len(value) >= 2:
        # Extract x and y coordinates if value is a tuple or list with length 2
            x3, y3 = value[0],value[1]
            print(x3, y3)

        
        vec1 = (x1 - x2, y1 - y2)
        vec2 = (x3 - x2, y3 - y2)
        
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        magnitude1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
        magnitude2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        cosine_angle = dot_product / (magnitude1 * magnitude2)
        cosine_angle = max(min(cosine_angle, 1), -1)
        return math.degrees(math.acos(cosine_angle))

    
    def euclidean_distance(keypoints, point1, point2):
        """Calculate the Euclidean distance between two landmarks."""
        x1, y1 = keypoints.get(point1, (0, 0))
        x2, y2 = keypoints.get(point2, (0, 0))
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def track_vertical_movement(keypoints, prev_keypoints, landmark_name):
        """Track vertical movement of a specific landmark between frames."""
        current_y = keypoints.get(landmark_name, (0, 0))[1]
        prev_y = prev_keypoints.get(landmark_name, (0, 0))[1]
        return abs(current_y - prev_y)

    # Extract y-coordinates of key landmarks
    head_y = keypoints.get('NOSE', (0, 0))[1]
    left_shoulder_y = keypoints.get('LEFT_SHOULDER', (0, 0))[1]
    right_shoulder_y = keypoints.get('RIGHT_SHOULDER', (0, 0))[1]
    left_hip_y = keypoints.get('LEFT_HIP', (0, 0))[1]
    right_hip_y = keypoints.get('RIGHT_HIP', (0, 0))[1]
    left_knee_y = keypoints.get('LEFT_KNEE', (0, 0))[1]
    right_knee_y = keypoints.get('RIGHT_KNEE', (0, 0))[1]
    left_ankle_y = keypoints.get('LEFT_ANKLE', (0, 0))[1]
    right_ankle_y = keypoints.get('RIGHT_ANKLE', (0, 0))[1]
 
    shoulders_y_avg = (left_shoulder_y + right_shoulder_y) / 2
    hips_y_avg = (left_hip_y + right_hip_y) / 2
    knees_y_avg = (left_knee_y + right_knee_y) / 2
    ankles_y_avg = (left_ankle_y + right_ankle_y) / 2

    print(f"Head Y: {head_y}, Shoulders Avg Y: {shoulders_y_avg}, Hips Avg Y: {hips_y_avg}")  # Debugging output

    
# Thresholds for movement detection
    horizontal_threshold = 0.05 * frame_height  # Tolerance for horizontal alignment
    standing_knee_hip_diff = 0.15 * frame_height
    lying_angle_threshold = 160  # Angle to determine lying down
    sitting_angle_threshold = 100  # Angle to determine sitting
    walking_movement_threshold = 0.02 * frame_height

    # Lying Down Detection
    if abs(head_y - shoulders_y_avg) < horizontal_threshold and abs(shoulders_y_avg - hips_y_avg) < horizontal_threshold:
        return "Lying Down"

    # Calculate key angles for activity detection
    hip_knee_angle = calculate_angle(keypoints, 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE')
    shoulder_angle = calculate_angle(keypoints, 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'NOSE')

    # Sitting Detection
    if hip_knee_angle < sitting_angle_threshold and abs(hips_y_avg - shoulders_y_avg) > horizontal_threshold:
        return "Sitting"

    # Standing Detection
    if abs(knees_y_avg - hips_y_avg) > standing_knee_hip_diff and abs(ankles_y_avg - knees_y_avg) > standing_knee_hip_diff:
        return "Standing"

    # Walking Detection
    if prev_keypoints:
        ankle_movement = track_vertical_movement(keypoints, prev_keypoints, 'LEFT_ANKLE')
        if ankle_movement > walking_movement_threshold:
            return "Walking"

    # Fall Detection
    if shoulder_angle > lying_angle_threshold:
        return "Fall Down"

    # Unknown Activity
    return "Updating"


def extract_keypoints_normalized(results_pose):
    keypoints = {}
    for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
        if landmark.visibility > 0.0:  # Only use landmarks with high confidence
            keypoints[mp_pose.PoseLandmark(idx).name] = (landmark.x, landmark.y, landmark.z)
    return keypoints


def detect_fall(keypoints):
# Check if critical landmarks are available before proceeding
    required_landmarks = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    # Ensure all required landmarks are in keypoints
    for landmark in required_landmarks:
        if landmark not in keypoints:
            print(f"Missing landmark: {landmark}")
            return False  # Return False if any required landmark is missing

    # Safely access landmarks using .get() to avoid KeyErrors
    head_y = keypoints.get(mp_pose.PoseLandmark.NOSE, None)
    left_shoulder_y = keypoints.get(mp_pose.PoseLandmark.LEFT_SHOULDER, None)
    right_shoulder_y = keypoints.get(mp_pose.PoseLandmark.RIGHT_SHOULDER, None)

    # Check if we got valid coordinates, if not, return False
    if head_y is None or left_shoulder_y is None or right_shoulder_y is None:
        print("Error: Missing or invalid landmarks.")
        return False

    # Fall detection logic based on pose keypoints
    if head_y.y < min(left_shoulder_y.y, right_shoulder_y.y):
        return True  # Fall detected based on head position relative to shoulders

    return False

def predict_activity(frame_sequence):
    # Preprocess each frame
    preprocessed_frames = []
    for frame in frame_sequence:
        # Resize the frame to match model's expected size (e.g., 64x64)
        frame_resized = cv2.resize(frame.cpu().numpy().transpose(1, 2, 0), (64, 64))  # Resize frame
        preprocessed_frames.append(frame_resized)
    
    # Stack the frames to form the input array (shape: batch_size, time_steps, height, width, channels)
    input_array = np.array(preprocessed_frames)
    input_array = np.expand_dims(input_array, axis=0)  # Add batch dimension

    # Normalize the frames if required (optional)
    input_array = input_array / 255.0  # Normalize to [0, 1]

    # Use the CNN-LSTM model for prediction
    activity_output = cnn_lstm_model.predict(input_array)
    activity_labels = ["Fall", "Walking", "Walking Upstairs", "Walking Downstairs", "Sitting", "Standing"]
    return activity_labels[np.argmax(activity_output)]

# Process uploaded video
def process_video(video_filename):
    global fall_status, activity_status, fall_image_path,processed_video_filename
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    cap = cv2.VideoCapture(video_path)
    frame_sequence = []
    seq_len = 5
    frame_cnt = 0
    # Dynamically set the skip frame rate based on the video's FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frm = max(1, int(video_fps / 10))
    confidence_threshold = 0.25
    timeline = []
    # Output video file setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
    output_video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{video_filename}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
            # Apply deblurring
        frame = preprocess_frame(frame)
        
        frame_cnt += 1
        if frame_cnt % skip_frm != 0:
            continue

        frame_resized = cv2.resize(frame, (640, 640))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = transforms.ToTensor()(frame_rgb).unsqueeze(0).to(device)

        results = model(frame_rgb)
        boxes = results[0].boxes.xyxy
        confidences = results[0].boxes.conf
        class_ids = results[0].boxes.cls.int()

        for box, score, class_id in zip(boxes, confidences, class_ids):
            if class_id != 0 or score < confidence_threshold:
                continue
            x1, y1, x2, y2 = map(int, box)
            bbox =(x1, y1, x2, y2)
            label = "Person Detected"
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Extract pose keypoints for the detected person
            person_frame = frame[y1:y2, x1:x2]
            prev_keypoints = None
            # Check if person_frame is valid before processing
            if person_frame is not None and person_frame.size != 0:
                results_pose = pose.process(cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB))

                if results_pose.pose_landmarks:
                    # Create keypoints dictionary with pose landmarks
                    keypoints = {}
                    keypoints = extract_keypoints_normalized(results_pose)
                    # Detect activity
                    activity_status = detect_activity_status(keypoints, prev_keypoints)

                    # Update previous keypoints
                    prev_keypoints = keypoints
                    # Smooth activity
                    smoothed_label = smooth_activity(activity_status)
                    # Fall detection based on pose keypoints
                    fall_detected = detect_fall(keypoints)
                    # Annotate frame with activity status
                    cv2.putText(frame, f"Activity: {activity_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    # Fall detection based CNN-LSTM model
                    frame_sequence.append(frame_tensor.squeeze(0))
                    if len(frame_sequence) > seq_len:
                        frame_sequence.pop(0)
                    if len(frame_sequence) == seq_len:
                        label = predict_activity(frame_sequence)
                        smoothed_label = smooth_activity(label)
                        # Update the global activity status with the smoothed label
                        activity_status = smoothed_label
                        cv2.putText(frame, f"Activity: {activity_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        
                    # Fall detection based on pose keypoints
                    if label or fall_detected:
                        fall_status = "Fall Detected"
                        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                        fall_image_path = f"static/fall_images/fall_{timestamp}.jpg"
                        cv2.imwrite(fall_image_path, frame)
                        timeline.append({'time': timestamp, 'status': 'Fall Detected'})
                    # Annotate frame
                    draw_annotations(frame, smoothed_label, bbox, fall_detected)
            else:
                print(f"Skipping frame: person_frame is empty. Bounding box: {x1}, {y1}, {x2}, {y2}")    
        # Write the annotated frame to the output video
        out.write(frame)   
        # _, buffer = cv2.imencode('.jpg', frame)
        # yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()
    out.release()
    # Store the processed video filename
    processed_video_filename = f"output_{video_filename}"
    # Return the path of the output video for rendering or download
    return output_video_path

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Process the video after upload
        process_video(filename)
        return redirect(url_for('result'))
    return "Invalid file format", 400

# Route to display results after video processing
@app.route('/result')
def result():
    output_video_url = f"/uploads/output_{processed_video_filename}"  
    return render_template('result.html', fall_status=fall_status, activity_status=activity_status, 
                           timeline=timeline, fall_image_path=fall_image_path, output_video_url=output_video_url)

# Route to fetch fall image (if available)
@app.route('/fall_image')
def fall_image():
    if fall_image_path:
        return jsonify({"image_url": fall_image_path, "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    return jsonify({"image_url": "", "timestamp": ""})


# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
