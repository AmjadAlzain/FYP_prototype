Real-Time Container Anomaly Detection on ESP32-S3-EYE
Optimization of Anomaly Detection Models Towards Real-Time Logistics Monitoring on Espressif ESP32 Edge Devices

Author: Amjad Alzain MuhammadSaeed Ali

University: University of Malaya, Faculty of Computer Science & Information Technology

GitHub Repository: https://github.com/AmjadAlzain/FYP_prototype

📖 Project Overview
This project presents a lightweight, edge-native container anomaly detection system designed to run entirely on a resource-constrained ESP32-S3-EYE microcontroller. The system addresses the critical need for fast, accurate, and cost-effective inspection of shipping containers in the logistics industry. By leveraging cutting-edge TinyML techniques, this solution automates the detection of structural damage like dents, cracks, and perforations, overcoming the limitations of manual inspections and expensive, cloud-reliant AI models.

The core of the system is a novel two-stage AI pipeline:

A TinyNAS/MCUNetV3-based CNN acts as an efficient feature extractor to first detect a container and potential regions of interest.

A Hyperdimensional Computing (HDC) classifier then analyzes the features to robustly and accurately classify the specific type of damage.

This approach allows the entire process—from image capture to final classification—to run in real-time on the edge device, ensuring privacy, eliminating latency, and removing the need for constant internet connectivity.

✨ Key Features
High-Accuracy Detection: A hybrid CNN and HDC model provides robust classification of container damage.

Real-Time Performance: Optimized to run at ~8-15 FPS on the ESP32-S3-EYE.

On-Device Intelligence: All processing happens locally. No cloud servers or internet connection required for inference.

Extremely Low-Cost: Built on affordable, off-the-shelf microcontroller hardware.

On-Device Learning: The HDC model supports on-device updates, allowing the system to learn from new examples in the field without a full retraining cycle.

Robust Dataset: Trained on a combination of real-world data from industry partners and the synthetic SeaFront dataset, enhanced with extensive data augmentation.

Project Structure
The repository is organized into two main parts: the model development scripts and the ESP32 firmware.

1_model_development/: Contains all Python scripts for dataset preprocessing, model training (TinyNAS detector and HDC classifier), and model conversion for deployment.

2_firmware_esp32/: Contains the complete C/C++ source code for the ESP32-S3-EYE, including camera drivers, TensorFlow Lite Micro integration, and the main application logic.

Getting Started
This section would typically contain instructions on how to build the firmware and run the models.

(Placeholder for future build and usage instructions)

Setup ESP-IDF: Clone and set up the Espressif IoT Development Framework.

Configure Project: Set the target to esp32s3.

Build and Flash: Use idf.py build and idf.py flash to deploy the firmware to the device.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
I would like to express my sincere appreciation to my supervisor, Prof. Dr. Loo Chu Kiong, for his unwavering guidance and mentorship. I am also thankful to Dr. Saw Shier Nee, Dr. Unizah Binti Obeidallah, and our industry partner, Infinity Logistics and Transport Ventures Limited, for their crucial feedback and support.
