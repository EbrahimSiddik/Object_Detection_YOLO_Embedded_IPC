// producer_yolo_queue.cpp
#include <windows.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstdint>

// Define structure WITHOUT #pragma pack to avoid Windows header issues
struct ObjectDetectionRecord {
    int class_id;
    float confidence;
    int x;
    int y;
    int width;
    int height;
};

// ---------------- CONFIG ----------------
constexpr int YOLO_INPUT_WIDTH = 640;
constexpr int YOLO_INPUT_HEIGHT = 640;
constexpr int IMAGE_CHANNELS = 3;

constexpr int MAX_DETECTIONS_PER_FRAME = 200;
constexpr int SHARED_MEMORY_QUEUE_SIZE = 5;

// YOLO parameters
constexpr float CONFIDENCE_THRESHOLD = 0.25f;
constexpr float NMS_IOU_THRESHOLD = 0.45f;

// Names
constexpr wchar_t SHARED_MEMORY_NAME[] = L"Local\\YOLO_QUEUE_SHM";
constexpr wchar_t SEMAPHORE_EMPTY_SLOTS[] = L"Local\\YOLO_EMPTY";
constexpr wchar_t SEMAPHORE_FULL_SLOTS[] = L"Local\\YOLO_FULL";
constexpr wchar_t MUTEX_QUEUE_ACCESS[] = L"Local\\YOLO_MUTEX";

// ---------------- SIZE CALC ----------------
constexpr size_t DETECTION_RECORD_SIZE = 24; // int + float + 4*int

constexpr size_t QUEUE_SLOT_SIZE =
    5 * sizeof(int) +                              // header
    MAX_DETECTIONS_PER_FRAME * DETECTION_RECORD_SIZE +             // detections
    YOLO_INPUT_WIDTH * YOLO_INPUT_HEIGHT * IMAGE_CHANNELS;         // image

constexpr size_t SHARED_MEMORY_TOTAL_SIZE =
    3 * sizeof(int) +                              // write_idx, read_idx, count
    SHARED_MEMORY_QUEUE_SIZE * QUEUE_SLOT_SIZE;

// ---------------- FUNCTION DECLARATIONS ----------------
void write_frame_to_shared_memory(
    uint8_t* shared_memory_ptr,
    HANDLE semaphore_empty_slots,
    HANDLE semaphore_full_slots,
    HANDLE mutex_queue_access,
    int frame_id,
    const cv::Mat& processed_frame,
    const std::vector<int>& class_ids,
    const std::vector<float>& confidences,
    const std::vector<cv::Rect>& bounding_boxes
);

std::vector<std::string> load_coco_object_classes(const std::string& path);
cv::Mat letterbox_resize(const cv::Mat& src, float& scale);

// ---------------- LOAD CLASSES ----------------
std::vector<std::string> load_coco_object_classes(const std::string& path) {
    std::vector<std::string> object_classes;
    std::ifstream ifs(path);
    std::string line;
    while (std::getline(ifs, line))
        object_classes.push_back(line);
    return object_classes;
}

// ---------------- LETTERBOX RESIZE ----------------
cv::Mat letterbox_resize(const cv::Mat& src, float& scale) {
    int original_width = src.cols;
    int original_height = src.rows;

    scale = std::min(
        static_cast<float>(YOLO_INPUT_WIDTH) / original_width,
        static_cast<float>(YOLO_INPUT_HEIGHT) / original_height
    );

    int scaled_width = static_cast<int>(original_width * scale);
    int scaled_height = static_cast<int>(original_height * scale);

    cv::Mat resized_image;
    cv::resize(src, resized_image, { scaled_width, scaled_height });

    cv::Mat padded_output(YOLO_INPUT_HEIGHT, YOLO_INPUT_WIDTH, CV_8UC3, cv::Scalar(114, 114, 114));

    resized_image.copyTo(padded_output(cv::Rect(0, 0, scaled_width, scaled_height)));
    return padded_output;
}

// ---------------- WRITE ONE FRAME ----------------
void write_frame_to_shared_memory(
    uint8_t* shared_memory_ptr,
    HANDLE semaphore_empty_slots,
    HANDLE semaphore_full_slots,
    HANDLE mutex_queue_access,
    int frame_id,
    const cv::Mat& processed_frame,
    const std::vector<int>& class_ids,
    const std::vector<float>& confidences,
    const std::vector<cv::Rect>& bounding_boxes
) {
    WaitForSingleObject(semaphore_empty_slots, INFINITE); // waits till setEmpty value becomes > 0
    WaitForSingleObject(mutex_queue_access, INFINITE);   // if mutex is free then lock it to this process for reading and writing

    int* control_block = reinterpret_cast<int*>(shared_memory_ptr);
    int producer_write_index = control_block[0];                // the index of the queue we need to write on in logical level

	uint8_t* current_slot_ptr = shared_memory_ptr + 3 * sizeof(int) + producer_write_index * QUEUE_SLOT_SIZE; // calculate physical slot address which is a circular queue 
    uint8_t* data_ptr = current_slot_ptr;

    int num_detections = static_cast<int>(bounding_boxes.size());
    if (num_detections > MAX_DETECTIONS_PER_FRAME) num_detections = MAX_DETECTIONS_PER_FRAME;

    // Header
    memcpy(data_ptr, &frame_id, 4); data_ptr += 4;
    memcpy(data_ptr, &YOLO_INPUT_WIDTH, 4);  data_ptr += 4;
    memcpy(data_ptr, &YOLO_INPUT_HEIGHT, 4);  data_ptr += 4;
    memcpy(data_ptr, &IMAGE_CHANNELS, 4); data_ptr += 4;
    memcpy(data_ptr, &num_detections, 4);      data_ptr += 4;

    // Detections - write each field separately
    for (int i = 0; i < MAX_DETECTIONS_PER_FRAME; i++) {
        if (i < num_detections) {
            memcpy(data_ptr, &class_ids[i], 4);           data_ptr += 4;
            memcpy(data_ptr, &confidences[i], 4);               data_ptr += 4;
            memcpy(data_ptr, &bounding_boxes[i].x, 4);             data_ptr += 4;
            memcpy(data_ptr, &bounding_boxes[i].y, 4);             data_ptr += 4;
            memcpy(data_ptr, &bounding_boxes[i].width, 4);         data_ptr += 4;
            memcpy(data_ptr, &bounding_boxes[i].height, 4);        data_ptr += 4;
        }
        else {
            memset(data_ptr, 0, 24);  // 24 bytes per detection record
            data_ptr += 24;
        }
    }

    // Image (BGR 640x640)
    size_t image_size = YOLO_INPUT_WIDTH * YOLO_INPUT_HEIGHT * IMAGE_CHANNELS;
    if (processed_frame.total() * processed_frame.elemSize() == image_size) {
        memcpy(data_ptr, processed_frame.data, image_size);
    }
    else {
        std::cerr << "Error: Image size mismatch!" << std::endl;
        // Write zeros if size doesn't match
        memset(data_ptr, 0, image_size);
    }

    // Advance queue
	control_block[0] = (producer_write_index + 1) % SHARED_MEMORY_QUEUE_SIZE; // update write index for next write
	control_block[2]++;  // update count

    std::cout << "Written frame " << frame_id << " with " << num_detections << " detections" << std::endl;

	ReleaseMutex(mutex_queue_access);                  // release mutex so that consumer can read
	ReleaseSemaphore(semaphore_full_slots, 1, nullptr);  // increment semFull count by 1 to indicate that a new slot is available fo the consumer to read 
}

// ---------------- MAIN ----------------
int main() {
    // -------- Load classes --------
    auto object_class_names = load_coco_object_classes("coco-classes.txt");
    if (object_class_names.empty()) {
        std::cerr << "ERROR: coco-classes.txt not found\n";
        return -1;
    }
    std::cout << "Loaded " << object_class_names.size() << " class names" << std::endl;

    // -------- Create shared memory --------
    HANDLE shared_memory_handle = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        nullptr,
        PAGE_READWRITE,
        0,
        SHARED_MEMORY_TOTAL_SIZE,
        SHARED_MEMORY_NAME
    );
    if (!shared_memory_handle) {
        std::cerr << "CreateFileMapping failed\n";
        return -1;
    }

    
    uint8_t* shared_memory_ptr = static_cast<uint8_t*>(
        MapViewOfFile(shared_memory_handle, FILE_MAP_ALL_ACCESS, 0, 0, SHARED_MEMORY_TOTAL_SIZE)
        );

    // -------- Create sync objects --------
    HANDLE semaphore_empty_slots = CreateSemaphore(nullptr, SHARED_MEMORY_QUEUE_SIZE, SHARED_MEMORY_QUEUE_SIZE, SEMAPHORE_EMPTY_SLOTS); // how many slots are empty if > 0 then producer can write
	HANDLE semaphore_full_slots = CreateSemaphore(nullptr, 0, SHARED_MEMORY_QUEUE_SIZE, SEMAPHORE_FULL_SLOTS);            // how many slots are full if > 0 then consumer can read
	HANDLE mutex_queue_access = CreateMutex(nullptr, FALSE, MUTEX_QUEUE_ACCESS);                        // mutual exclusion ensures that when when consumer or producer is reading/writing the other process can read/write

    // -------- Init control block --------
    int* control_block = reinterpret_cast<int*>(shared_memory_ptr);
    control_block[0] = 0; // write_idx
    control_block[1] = 0; // read_idx
    control_block[2] = 0; // count

    // -------- Open video --------
    cv::VideoCapture cap("video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open video\n";

        // Try to open webcam as fallback
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video or webcam" << std::endl;
            return -1;
        }
        std::cout << "Using webcam instead of video file" << std::endl;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "Video resolution: " << frame_width << "x" << frame_height << ", FPS: " << fps << std::endl;

    // -------- ONNX Runtime --------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    try {
        Ort::Session session(env, L"yolov5s.onnx", opts);
        std::cout << "Loaded YOLO model from yolov5s.onnx" << std::endl;

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);

        const char* input_names[] = { input_name.get() };
        const char* output_names[] = { output_name.get() };

        std::array<int64_t, 4> input_shape{ 1, 3, YOLO_INPUT_HEIGHT, YOLO_INPUT_WIDTH };
        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // -------- Video loop --------
        cv::Mat current_frame;
        int frame_id = 0;

        while (cap.read(current_frame)) {
            frame_id++;

            // Store original frame for transmission
            cv::Mat resized_frame;
            cv::resize(current_frame, resized_frame, cv::Size(YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT));

            // -------- Preprocess --------
            float letterbox_scale = 1.0f;
            cv::Mat preprocessed_input = letterbox_resize(current_frame, letterbox_scale);

            cv::cvtColor(preprocessed_input, preprocessed_input, cv::COLOR_BGR2RGB);
            preprocessed_input.convertTo(preprocessed_input, CV_32F, 1.0 / 255.0);

            // -------- NCHW tensor --------
            std::vector<float> input_tensor_data(YOLO_INPUT_WIDTH * YOLO_INPUT_HEIGHT * 3);
            int tensor_idx = 0;
            for (int channel = 0; channel < 3; channel++)
                for (int y = 0; y < YOLO_INPUT_HEIGHT; y++)
                    for (int x = 0; x < YOLO_INPUT_WIDTH; x++)
                        input_tensor_data[tensor_idx++] = preprocessed_input.at<cv::Vec3f>(y, x)[channel];

            Ort::Value input_tensor_ort =
                Ort::Value::CreateTensor<float>(
                    mem_info,
                    input_tensor_data.data(),
                    input_tensor_data.size(),
                    input_shape.data(),
                    input_shape.size()
                );

            // -------- Inference --------
            auto model_outputs = session.Run(
                Ort::RunOptions{ nullptr },
                input_names,
                &input_tensor_ort,
                1,
                output_names,
                1
            );

            float* output_data = model_outputs[0].GetTensorMutableData<float>();

            std::vector<int> detected_class_ids;
            std::vector<float> detection_confidences;
            std::vector<cv::Rect> detection_bounding_boxes;

            for (int i = 0; i < 25200; i++) {
                float objectness_confidence = output_data[4];
                if (objectness_confidence < CONFIDENCE_THRESHOLD) {
                    output_data += 85;
                    continue;
                }

                float* class_scores = output_data + 5;
                cv::Mat score_mat(1, object_class_names.size(), CV_32FC1, class_scores);

                cv::Point max_class_id;
                double max_class_score;
                cv::minMaxLoc(score_mat, nullptr, &max_class_score, nullptr, &max_class_id);

                float final_confidence = objectness_confidence * static_cast<float>(max_class_score);
                if (final_confidence >= CONFIDENCE_THRESHOLD) {
                    float center_x = output_data[0];
                    float center_y = output_data[1];
                    float bbox_width = output_data[2];
                    float bbox_height = output_data[3];

                    // Scale back to original image coordinates
                    int bbox_left = static_cast<int>((center_x - 0.5f * bbox_width) / letterbox_scale);
                    int bbox_top = static_cast<int>((center_y - 0.5f * bbox_height) / letterbox_scale);
                    int bbox_final_width = static_cast<int>(bbox_width / letterbox_scale);
                    int bbox_final_height = static_cast<int>(bbox_height / letterbox_scale);

                    // Clip to image boundaries
                    bbox_left = std::max(0, std::min(bbox_left, frame_width - 1));
                    bbox_top = std::max(0, std::min(bbox_top, frame_height - 1));
                    bbox_final_width = std::max(1, std::min(bbox_final_width, frame_width - bbox_left));
                    bbox_final_height = std::max(1, std::min(bbox_final_height, frame_height - bbox_top));

                    detected_class_ids.push_back(max_class_id.x);
                    detection_confidences.push_back(final_confidence);
                    detection_bounding_boxes.emplace_back(bbox_left, bbox_top, bbox_final_width, bbox_final_height);
                }

                output_data += 85;
            }

            // -------- NMS --------
            std::vector<int> nms_indices;
            cv::dnn::NMSBoxes(
                detection_bounding_boxes, detection_confidences,
                CONFIDENCE_THRESHOLD,
                NMS_IOU_THRESHOLD,
                nms_indices
            );

            // -------- Keep only NMS results --------
            std::vector<int> final_class_ids;
            std::vector<float> final_confidences;
            std::vector<cv::Rect> final_bounding_boxes;

            for (int idx : nms_indices) {
                final_class_ids.push_back(detected_class_ids[idx]);
                final_confidences.push_back(detection_confidences[idx]);
                final_bounding_boxes.push_back(detection_bounding_boxes[idx]);
            }

            // -------- Debug output --------
            std::cout << "Frame " << frame_id << ": Detected " << final_bounding_boxes.size() << " objects" << std::endl;
            for (size_t i = 0; i < final_bounding_boxes.size(); ++i) {
                int current_class_id = final_class_ids[i];
                std::string object_class_name = (current_class_id < object_class_names.size()) ?
                    object_class_names[current_class_id] : "Unknown";
                std::cout << "  - " << object_class_name << " (" << final_confidences[i]
                    << ") at [" << final_bounding_boxes[i].x << "," << final_bounding_boxes[i].y
                    << "," << final_bounding_boxes[i].width << "," << final_bounding_boxes[i].height << "]" << std::endl;
            }

            // -------- Scale boxes to 640x640 for transmission --------
            std::vector<cv::Rect> scaled_bounding_boxes;
            float scale_x = static_cast<float>(YOLO_INPUT_WIDTH) / frame_width;
            float scale_y = static_cast<float>(YOLO_INPUT_HEIGHT) / frame_height;

            for (const auto& bbox : final_bounding_boxes) {
                int scaled_x = static_cast<int>(bbox.x * scale_x);
                int scaled_y = static_cast<int>(bbox.y * scale_y);
                int scaled_width = static_cast<int>(bbox.width * scale_x);
                int scaled_height = static_cast<int>(bbox.height * scale_y);

                // Clip to 640x640 boundaries
                scaled_x = std::max(0, std::min(scaled_x, YOLO_INPUT_WIDTH - 1));
                scaled_y = std::max(0, std::min(scaled_y, YOLO_INPUT_HEIGHT - 1));
                scaled_width = std::max(1, std::min(scaled_width, YOLO_INPUT_WIDTH - scaled_x));
                scaled_height = std::max(1, std::min(scaled_height, YOLO_INPUT_HEIGHT - scaled_y));

                scaled_bounding_boxes.emplace_back(scaled_x, scaled_y, scaled_width, scaled_height);
            }

            // -------- Write to shared memory --------
            write_frame_to_shared_memory(
                shared_memory_ptr, semaphore_empty_slots, semaphore_full_slots, mutex_queue_access,
                frame_id, resized_frame, final_class_ids, final_confidences, scaled_bounding_boxes
            );
        }

        std::cout << "Finished processing video" << std::endl;

    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        std::cerr << "Please ensure yolov5s.onnx is in the current directory" << std::endl;
        return -1;
    }

    return 0;
}


