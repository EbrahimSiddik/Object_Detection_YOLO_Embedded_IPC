# How to Run YOLO Producer-Consumer Project

## Quick Start Guide

### Step 1: Start the Producer (C++)

Open a **PowerShell** terminal and run:

```powershell
cd c:\Users\MN\Desktop\Embedded\consumer\producer
.\x64\Debug\assignment_cpp.exe
```

**Expected Output:**
```
Loaded 80 class names
Video resolution: 1920x1080, FPS: 30
Loaded YOLO model from yolov5s.onnx
Frame 1: Detected 24 objects
  - person (0.89) at [x,y,w,h]
  - car (0.85) at [x,y,w,h]
  ...
Written frame 1 with 24 detections
Frame 2: Detected 26 objects
...
```

⚠️ **Keep this terminal running!** Don't close it.

---

### Step 2: Start the Consumer (Python)

Open a **SECOND PowerShell** terminal (keep the first one running) and run:

```powershell
cd c:\Users\MN\Desktop\Embedded\consumer\consumer
C:\Users\MN\AppData\Local\Programs\Python\Python312\python.exe consumer_shm.py
```

**Note:** Use the full path to Python312 to avoid conflicts with Anaconda.

**Expected Output:**
```
Loaded 80 classes from coco-classes.txt
Waiting for Producer...
Connected! SHM Size: 6168112 bytes
Starting Consumer Loop...
Press ESC to exit
Read frame ID: 1 with 24 detections
Read frame ID: 2 with 26 detections
...
```

**A video window will appear** showing:
- ✅ Live video with detected objects
- ✅ Colored bounding boxes around objects
- ✅ Labels (e.g., "person: 0.89", "car: 0.85")
- ✅ Frame counter (top-left)
- ✅ FPS counter (top-left)
- ✅ Detection count (top-left)

---

## Using the Helper Scripts

### Alternative Method (Easier):

**Terminal 1 - Producer:**
```powershell
cd c:\Users\MN\Desktop\Embedded\consumer\producer
.\run_producer.bat
```

**Terminal 2 - Consumer:**
```powershell
cd c:\Users\MN\Desktop\Embedded\consumer\consumer
.\run_consumer.bat
```

These batch files check for required files before running.

---

## Controls

While the video window is active:

- **ESC** - Stop and exit the consumer
- **SPACE** - Pause/resume video playback

---

## Troubleshooting

### Problem: "Waiting for Producer..." keeps repeating

**Solution:** Start the **producer FIRST**, then start the consumer.

### Problem: "Cannot open video" error in producer

**Solution:** The producer will automatically fallback to webcam (camera 0). To use the video file, ensure `video.mp4` exists in the producer directory.

### Problem: Window doesn't appear

**Possible causes:**
1. Window is minimized or behind other windows - check taskbar
2. Alt+Tab to find the "YOLO Real-Time Detection" window
3. OpenCV not installed: `pip install opencv-python`

### Problem: Consumer exits with error

**Solution:** Ensure Python dependencies are installed:
```powershell
pip install opencv-python numpy pywin32
```

---

## Complete Step-by-Step (From Scratch)

### 1. Open TWO PowerShell terminals in VS Code

Press `` Ctrl+Shift+` `` twice to open two terminal windows, or click the "+" icon in the terminal panel.

### 2. In Terminal 1 (Producer):

```powershell
# Navigate to producer directory
cd c:\Users\MN\Desktop\Embedded\consumer\producer

# Run the producer
.\x64\Debug\assignment_cpp.exe
```

**Wait for output:** "Loaded YOLO model from yolov5s.onnx"

### 3. In Terminal 2 (Consumer):

```powershell
# Navigate to consumer directory  
cd c:\Users\MN\Desktop\Embedded\consumer\consumer

# Run the consumer
python consumer_shm.py
```

**Look for:** "Connected! SHM Size: 6168112 bytes"

### 4. Watch the Magic! 🎉

A window titled **"YOLO Real-Time Detection"** will appear showing real-time object detection with bounding boxes and labels!

---

## Output Examples

### Producer Console Output:
```
Frame 100: Detected 28 objects
  - person (0.865243) at [8,636,109,297]
  - car (0.903548) at [508,543,285,86]
  - traffic light (0.737593) at [1430,377,64,147]
  ...
Written frame 100 with 28 detections
```

### Consumer Console Output:
```
Read frame ID: 100 with 28 detections
Read frame ID: 101 with 29 detections
Processed 30 frames. Last frame: 101 with 29 detections
```

### Visual Output (Window):
- Video playing at ~30 FPS
- Colored boxes around: persons, cars, traffic lights, etc.
- Each box has a label showing class name and confidence
- Top-left corner shows frame number, FPS, and detection count

---

## Architecture Overview

```
┌──────────────────┐         Shared Memory          ┌──────────────────┐
│   Producer.exe   │  ────────────────────────────► │ consumer_shm.py  │
│   (C++ YOLO)     │    Semaphores + Mutex          │  (Python OpenCV) │
│                  │                                 │                  │
│ - Read video     │   Queue: 5 slots               │ - Display video  │
│ - YOLO inference │   Each: 640x640 frame          │ - Draw boxes     │
│ - Write frames   │   + detection data             │ - Show labels    │
└──────────────────┘                                 └──────────────────┘
```

---

## Performance Tips

- Producer processes ~15-30 FPS (depends on CPU/GPU)
- Consumer should match producer FPS
- If consumer is slower, it will skip frames (by design)
- Both programs print status every 30 frames

---

## Stopping the Programs

1. **Click on the video window**
2. **Press ESC** - This stops the consumer
3. **In the producer terminal**, press `Ctrl+C` to stop it

Or close both terminal windows.

---

## File Locations

```
consumer/
├── producer/
│   ├── x64/Debug/
│   │   └── assignment_cpp.exe    ← Producer executable
│   ├── video.mp4                 ← Input video
│   ├── yolov5s.onnx             ← YOLO model
│   ├── coco-classes.txt         ← Class labels
│   └── run_producer.bat         ← Helper script
│
└── consumer/
    ├── consumer_shm.py          ← Consumer script
    ├── coco-classes.txt         ← Class labels
    └── run_consumer.bat         ← Helper script
```

---

## Summary

1. **Open two terminals**
2. **Terminal 1:** `cd producer` → run `assignment_cpp.exe`
3. **Terminal 2:** `cd consumer` → run `python consumer_shm.py`
4. **Watch the video window** with real-time object detection!
5. **Press ESC** when done

That's it! 🚀
