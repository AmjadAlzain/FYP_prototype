# Canva Presentation Outline: FYP2 VIVA

**Theme:** Modern, clean, and professional. Use a consistent color palette (e.g., blues, greys, and a highlight color like green or orange). Use icons to represent concepts.

---

### Slide 1: Title Slide
- **Title:** Optimization of Anomaly Detection Models Towards Real-Time Logistics Monitoring on Espressif ESP32 Edge Devices
- **Subtitle:** FYP2 VIVA Presentation
- **Author:** Amjad Alzain MuhammadSaeed Ali (S2151110/1)
- **Visuals:** A high-quality background image of a modern port or shipping containers. University and Faculty logos at the bottom.

---

### Slide 2: Agenda
- **Title:** Agenda
- **Content:** A numbered or bulleted list of the presentation sections.
  1. Introduction & Problem Statement
  2. Project Objectives & Scope
  3. System Architecture & Methodology
  4. Technical Implementation (A Modular Journey)
  5. Testing, Evaluation & Performance
  6. Live Demonstration
  7. Conclusion & Future Work
- **Visuals:** Use icons for each agenda item (e.g., a lightbulb for Introduction, a gear for Implementation).

---

### Slide 3: Introduction: The Challenge in Global Logistics
- **Title:** The Core Challenge in Global Logistics
- **Key Point 1:** Containers are the lifeblood of trade, but they get damaged.
- **Key Point 2:** Manual inspections are slow, error-prone, and can't keep up with demand.
- **Key Point 3 (The Gap):** Powerful AI models exist but are too resource-heavy for on-site deployment in ports.
- **Visuals:** A split screen showing a manual inspector on one side and a graphic of a GPU/cloud on the other, with a "not feasible" icon.

---

### Slide 4: Our Goal: Smart, Efficient, Edge-First
- **Title:** Our Goal: A Smarter Solution
- **Main Statement (in large font):** To create a lightweight, real-time container anomaly detection system that runs entirely on a low-cost, low-power ESP32-S3-EYE device.
- **Visuals:** An image of the ESP32-S3-EYE board with lines connecting it to icons representing "Real-Time," "Low Power," and "On-Device AI."

---

### Slide 5: Problem Statement
- **Title:** Problem Statement
- **Problem 1: Speed & Accuracy:** Manual checks are a bottleneck.
- **Problem 2: Hardware Constraints:** Heavy AI models don't fit on microcontrollers.
- **Problem 3: Scalability:** Cloud solutions are expensive and require constant connectivity.
- **Visuals:** Use a three-column layout with a clear icon for each problem (e.g., a clock for speed, a microchip for hardware, a globe with a "no connection" symbol for scalability).

---

### Slide 6: Project Objectives
- **Title:** Our Three Core Objectives
- **Objective 1:** **Enhance Data Quality:** Use augmentation to prepare the model for real-world chaos.
- **Objective 2:** **Develop a Lightweight Model:** Train a hybrid MCUNetV3 + HDC system for the edge.
- **Objective 3:** **Deploy & Evaluate:** Prove it works on the ESP32-S3-EYE with on-device learning.
- **Visuals:** A timeline or flow graphic showing how each objective builds on the last. Add the "Infinity Logistics" logo under a "Collaboration" heading.

---

### Slide 7: System Architecture Overview
- **Title:** System Architecture
- **Content:** A simplified, visually appealing version of the Mermaid diagram from the module reports.
  - `[Camera Icon]` -> `[Preprocessing Icon]` -> `[AI Brain Icon: MCUNetV3 + HDC]` -> `[ESP32 Board Icon]` -> `[Monitor/UI Icon]`
  - Add a looping arrow from the UI back to the AI Brain, labeled "On-Device Learning."
- **Tagline:** A self-contained, intelligent pipeline on a single chip.

---

### Slide 8: Methodology: A 6-Module Journey
- **Title:** Our Development Methodology
- **Content:** Display the 6 modules as a circular or linear flow.
  - M1: Data
  - M2: Detection
  - M3: Classification
  - M4: Optimization
  - M5: Integration
  - M6: GUI & Testing
- **Visuals:** Give each module a distinct color and icon.

---

### Slide 9: Module 1 & 2: Data & Detection
- **Title:** Building the Foundation: Data & Detection
- **Left Side (Module 1):**
  - **Header:** Smart Data Processing
  - **Stats:** 15,000+ images processed, 35,000+ patches extracted.
  - **Visual:** Show a grid of augmented images (rotated, brightness changed).
- **Right Side (Module 2):**
  - **Header:** Lightweight TinyNAS Model
  - **Stat:** 89.3% mAP for container detection.
  - **Visual:** A simple diagram showing a full image going in and a bounding box coming out.

---

### Slide 10: Module 3: The HDC Brain
- **Title:** The Secret Sauce: Hyperdimensional Computing (HDC)
- **Concept:** A brain-inspired classifier that is robust, efficient, and can learn on the fly.
- **Key Stat 1:** **11% more robust to noise** than a standard CNN.
- **Key Stat 2:** Enables **on-device learning** in just 15ms.
- **Visual:** A graphic showing a feature vector being transformed into a complex "hypervector," then compared against learned patterns.

---

### Slide 11: Module 4 & 5: Optimization & Deployment
- **Title:** From Lab to Reality: Optimization & Deployment
- **Left Side (Module 4):**
  - **Header:** Model Optimization
  - **Stat:** 4.02x model compression with only -0.3% accuracy loss.
  - **Visual:** A funnel graphic showing a large model going in and a small, efficient model coming out.
- **Right Side (Module 5):**
  - **Header:** On-Device Integration
  - **Stat:** Full system running at 8.2 FPS on the ESP32.
  - **Visual:** A photo of the ESP32-S3-EYE running the system, with detection results on its screen.

---

### Slide 12: Module 6: GUI & Testing
- **Title:** Control & Validation
- **Left Side (GUI):**
  - **Header:** PC-Based Control GUI
  - **Features:** Real-time monitoring, training management, analytics.
  - **Visual:** A screenshot of the GUI application in action.
- **Right Side (Testing):**
  - **Header:** Comprehensive Test Suite
  - **Stat:** **84% Success Rate** across 50 automated tests.
  - **Visual:** A simple table showing the Pass/Fail status for Unit, Integration, and A/B tests.

---

### Slide 13: Final System Performance
- **Title:** Final Performance: Meeting the Challenge
- **Content:** A clean, easy-to-read table with the final metrics.
  | Performance Metric | Target | **Achieved** |
  | :--- | :--- | :--- |
  | Real-time FPS (on ESP32) | > 5 FPS | **8.2 FPS** |
  | End-to-End Accuracy | > 95% | **97.5%** |
  | Training Time per Sample | < 50 ms | **15 ms** |
  | Memory Usage (SRAM) | < 400 KB | **156 KB** |
- **Visuals:** Use green checkmarks next to each "Achieved" metric.

---

### Slide 14: Challenges & Solutions
- **Title:** Overcoming Hurdles
- **Content:** A four-quadrant grid.
  - **Quadrant 1: Model Size:** Solution -> TinyNAS & INT8 Quantization.
  - **Quadrant 2: Class Imbalance:** Solution -> Class Weighting.
  - **Quadrant 3: Conversion Errors:** Solution -> Robust Post-Training Quantization.
  - **Quadrant 4: Real-Time Constraints:** Solution -> Time-Sliced Training & Memory Pools.
- **Visuals:** A simple icon for each challenge.

---

### Slide 15: Live Demonstration
- **Title:** Live System Demonstration
- **Content:**
  1. Show the ESP32-S3-EYE device running.
  2. Demonstrate real-time damage detection.
  3. Showcase the on-device training feature by correcting a misclassification.
  4. Briefly show the PC-based GUI monitoring the device.
- **Visuals:** A large, clean slide with just the title and maybe a "Live Demo" icon.

---

### Slide 16: Conclusion
- **Title:** Conclusion
- **Main Statement:** This project successfully proves that high-accuracy, real-time AI with on-device learning is achievable on low-cost microcontrollers.
- **Key Takeaways:**
  - All objectives were met.
  - The hybrid CNN-HDC architecture is a novel and effective solution for the edge.
  - The system is a practical, scalable, and cost-effective tool for the logistics industry.

---

### Slide 17: Future Work
- **Title:** Future Work
- **Point 1: Core Model Enhancement:** Retrain with more augmented data to perfect bounding box accuracy.
- **Point 2: Multi-Damage Detection:** Upgrade the model to identify and locate multiple damage types on a single container.
- **Point 3: Advanced On-Device Learning:** Explore few-shot and continual learning for even faster adaptation.
- **Visuals:** A roadmap graphic showing the current project and these future steps.

---

### Slide 18: Thank You & Q&A
- **Title:** Thank You
- **Subtitle:** Questions?
- **Contact Info:**
  - Amjad Alzain MuhammadSaeed Ali
  - S2151110/1
  - [Your Email]
- **Visuals:** A clean, professional closing slide with University/Faculty logos.
