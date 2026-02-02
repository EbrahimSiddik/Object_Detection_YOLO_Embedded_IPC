# consumer_queue.py
import mmap
import struct
import cv2
import numpy as np
import win32event
import win32con
import time

# ---------------- CONFIG ----------------
YOLO_INPUT_WIDTH = 640
YOLO_INPUT_HEIGHT = 640
IMAGE_CHANNELS = 3

MAX_DETECTIONS_PER_FRAME = 200
SHARED_MEMORY_QUEUE_SIZE = 5

# Shared Memory & Sync Names
SHARED_MEMORY_NAME   = "Local\\YOLO_QUEUE_SHM"
SEMAPHORE_EMPTY_SLOTS  = "Local\\YOLO_EMPTY"
SEMAPHORE_FULL_SLOTS   = "Local\\YOLO_FULL"
MUTEX_QUEUE_ACCESS = "Local\\YOLO_MUTEX"

SEMAPHORE_ALL_ACCESS = 0x1F0003 # access level
MUTEX_ALL_ACCESS = 0x1F0001  # access level

# ---------------- SIZE CALCULATIONS ----------------
DETECTION_RECORD_SIZE = 24  # int + float + 4*int = 4 + 4 + 16 = 24 bytes
FRAME_HEADER_SIZE = 20
QUEUE_CONTROL_BLOCK_SIZE = 12

# Slot Size calculation must match C++ exactly
QUEUE_SLOT_SIZE = FRAME_HEADER_SIZE + (MAX_DETECTIONS_PER_FRAME * DETECTION_RECORD_SIZE) + (YOLO_INPUT_WIDTH * YOLO_INPUT_HEIGHT * IMAGE_CHANNELS)
SHARED_MEMORY_TOTAL_SIZE = QUEUE_CONTROL_BLOCK_SIZE + (SHARED_MEMORY_QUEUE_SIZE * QUEUE_SLOT_SIZE)

# Load COCO class names
def load_coco_object_classes(filename="coco-classes.txt"):
    object_classes = []
    try:
        with open(filename, 'r') as f:
            object_classes = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(object_classes)} classes from {filename}")
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default COCO classes.")
        # Default COCO classes (first 20 for example)
        object_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
    return object_classes

# Load classes
OBJECT_CLASSES = load_coco_object_classes()

# Color palette for different classes
CLASS_COLORS = np.random.uniform(0, 255, size=(len(OBJECT_CLASSES), 3))

# ---------------- INITIALIZATION ----------------
print("Waiting for Producer...")
while True:
    try:
        shared_memory = mmap.mmap(-1, SHARED_MEMORY_TOTAL_SIZE, tagname=SHARED_MEMORY_NAME, access=mmap.ACCESS_WRITE)
        semaphore_empty_slots = win32event.OpenSemaphore(SEMAPHORE_ALL_ACCESS, False, SEMAPHORE_EMPTY_SLOTS) # how many slots are avaialbe for the producer to write
        semaphore_full_slots  = win32event.OpenSemaphore(SEMAPHORE_ALL_ACCESS, False, SEMAPHORE_FULL_SLOTS) # how many slots are filled with data for the consumer to read 
        mutex_queue_access    = win32event.OpenMutex(MUTEX_ALL_ACCESS, False, MUTEX_QUEUE_ACCESS)
        
        print(f"Connected! SHM Size: {SHARED_MEMORY_TOTAL_SIZE} bytes")
        print(f"Detection record size: {DETECTION_RECORD_SIZE} bytes")
        print(f"Slot data size: {QUEUE_SLOT_SIZE} bytes")
        break
    except Exception as e:
        print(f"Waiting for producer... Error: {e}")
        time.sleep(0.5)

# ---------------- READ FRAME ----------------
def read_frame_from_shared_memory():
    win32event.WaitForSingleObject(semaphore_full_slots, win32event.INFINITE) # if greater than 0 than proceed, else wait till > 0
    win32event.WaitForSingleObject(mutex_queue_access, win32event.INFINITE)  # wait till mutex is available

    try:
        # 1. Read Control Block
        shared_memory.seek(0)
        producer_write_index, consumer_read_index, filled_slots_count = struct.unpack("iii", shared_memory.read(QUEUE_CONTROL_BLOCK_SIZE))

        # 2. Calculate Offset
        current_slot_start = QUEUE_CONTROL_BLOCK_SIZE + (consumer_read_index * QUEUE_SLOT_SIZE)
        
        # 3. Read Slot Header
        shared_memory.seek(current_slot_start)
        frame_id, frame_width, frame_height, channels, num_detections = struct.unpack("iiiii", shared_memory.read(FRAME_HEADER_SIZE))

        if num_detections > MAX_DETECTIONS_PER_FRAME: 
            num_detections = MAX_DETECTIONS_PER_FRAME
        
        # 4. Skip to detections section
        detections_start_offset = current_slot_start + FRAME_HEADER_SIZE
        shared_memory.seek(detections_start_offset)
        
        # 5. Read detections
        detected_objects = []
        for i in range(MAX_DETECTIONS_PER_FRAME):
            detection_data = shared_memory.read(DETECTION_RECORD_SIZE)
            if i < num_detections:
                # Use little-endian to match Windows
                class_id, confidence, bbox_x, bbox_y, bbox_width, bbox_height = struct.unpack("<ifiiii", detection_data)
                detected_objects.append((class_id, confidence, bbox_x, bbox_y, bbox_width, bbox_height))
        
        # 6. Read Image
        image_data_offset = detections_start_offset + (MAX_DETECTIONS_PER_FRAME * DETECTION_RECORD_SIZE)
        shared_memory.seek(image_data_offset)
        
        image_size = YOLO_INPUT_WIDTH * YOLO_INPUT_HEIGHT * IMAGE_CHANNELS
        raw_image_data = shared_memory.read(image_size)
        
        if len(raw_image_data) != image_size:
            print(f"Error: Image data size mismatch. Got {len(raw_image_data)}, expected {image_size}")
            processed_frame = np.zeros((YOLO_INPUT_HEIGHT, YOLO_INPUT_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
        else:
            processed_frame = np.frombuffer(raw_image_data, dtype=np.uint8).reshape((YOLO_INPUT_HEIGHT, YOLO_INPUT_WIDTH, IMAGE_CHANNELS)).copy()

        # 7. Update indices
        next_read_index = (consumer_read_index + 1) % SHARED_MEMORY_QUEUE_SIZE
        shared_memory.seek(4)
        shared_memory.write(struct.pack("i", next_read_index))  # the next read index
        shared_memory.seek(8)
        shared_memory.write(struct.pack("i", filled_slots_count - 1))     # decrement count to indicate the no of frames in the buffer that can be consumed

    finally:  
        win32event.ReleaseMutex(mutex_queue_access)             # release mutex so that producer can write again
        win32event.ReleaseSemaphore(semaphore_empty_slots, 1)   # indicate that one more slot is available for producer to write

    return processed_frame, detected_objects, frame_id

# ---------------- MAIN LOOP ----------------
cv2.namedWindow("YOLO Real-Time Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Real-Time Detection", 800, 800)

print("Starting Consumer Loop...")
print("Press ESC to exit")

total_frames_processed = 0
fps_calculation_start_time = time.time()
fps_calculation_frame_count = 0
current_display_fps = 0

while True:
    try:
        # Calculate FPS
        fps_calculation_frame_count += 1
        if fps_calculation_frame_count >= 30:
            current_display_fps = fps_calculation_frame_count / (time.time() - fps_calculation_start_time)
            fps_calculation_start_time = time.time()
            fps_calculation_frame_count = 0
        
        # Read frame from shared memory
        processed_frame, detected_objects, current_frame_id = read_frame_from_shared_memory()
        print(f"Read frame ID: {current_frame_id} with {len(detected_objects)} detections")
        total_frames_processed += 1
        
        # Frame is already in BGR format from C++ (OpenCV default)
        display_frame = processed_frame.copy()
        
        # Draw detections
        for class_id, confidence, bbox_x, bbox_y, bbox_width, bbox_height in detected_objects:
            # Get class name and color
            object_class_name = OBJECT_CLASSES[class_id] if class_id < len(OBJECT_CLASSES) else f"Class_{class_id}"
            object_color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
            
            # Convert color from RGB to BGR for OpenCV
            bgr_color = (int(object_color[2]), int(object_color[1]), int(object_color[0]))
            
            # Draw bounding box
            cv2.rectangle(display_frame, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), bgr_color, 2)
            
            # Create label
            detection_label = f"{object_class_name}: {confidence:.2f}"
            
            # Calculate text size
            (label_width, label_height), label_baseline = cv2.getTextSize(
                detection_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                display_frame,
                (bbox_x, bbox_y - label_height - label_baseline - 5),
                (bbox_x + label_width, bbox_y),
                bgr_color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                display_frame,
                detection_label,
                (bbox_x, bbox_y - label_baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Add info overlay
        cv2.putText(
            display_frame,
            f"Frame: {current_frame_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            display_frame,
            f"FPS: {current_display_fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            display_frame,
            f"Detections: {len(detected_objects)}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Display
        cv2.imshow("YOLO Real-Time Detection", display_frame)
        
        # Print debug info occasionally
        if total_frames_processed % 30 == 0:
            print(f"Processed {total_frames_processed} frames. Last frame: {current_frame_id} with {len(detected_objects)} detections")
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("ESC pressed, exiting...")
            break
        elif key == ord(' '):  # Space to pause
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
        break
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
        break

print("Cleaning up...")
cv2.destroyAllWindows()
try:
    shared_memory.close()
except:
    pass

print(f"Total frames processed: {total_frames_processed}")