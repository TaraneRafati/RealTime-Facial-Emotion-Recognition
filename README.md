# Real-Time Facial Emotion Recognition System

This project implements a two-phase pipeline for detecting faces and classifying facial emotions from images and live video. Phase 1 provides the initial training and prototype scripts, while Phase 2 contains the refined, deployment-ready modules with YOLO-based face detection and clean dataset preparation.

---

## Project Goals
- Achieve at least **60% emotion classification accuracy**
- Support **5–7 emotion classes** depending on the dataset
- Provide a **real-time inference system** with:
  - Face detection  
  - Emotion prediction  
  - Bounding boxes and optional emoji overlays  

---

## Project Structure

```
.
├── phase-1/           
│   ├── train.py
│   ├── eval.py
│   ├── test.py
│   ├── webcam.py
│   ├── face_detector.py
│   ├── main.py
│   ├── split.py
│   ├── utils/
│   ├── models/
│   ├── weights/
│   └── output/
│
└── phase-2/            
    ├── face_detector.py
    ├── main.py
    ├── split.py
    ├── requirements.txt
    ├── utils/
    ├── models/
    ├── weights/
    └── output/
```

---

# Phase 1 (Prototype System)

Phase 1 focuses on establishing the baseline pipeline:

### Key Features
- **CNN-based emotion classifier** trained with data augmentation  
- Initial evaluation (accuracy, F1, confusion matrix)  
- Basic **webcam demo** for real-time inference  
- Early face detection prototype  
- Initial Train/Validation/Test split

This phase provides the first working version of the system but relies on raw dataset images, without refined face cropping.

---

# Phase 2 (Refined & Deployment-Ready System)

Phase 2 introduces a production-quality pipeline with improved face detection and dataset preparation.

### Key Features
- **YOLO-based face detection** via the final `YoloDetector` module  
- **Automated cropping** of detected faces for clean dataset creation  
- **Final Train/Test splitting** on cropped faces  
- Utility functions for visualization, preprocessing, and exporting results  
- `requirements.txt` for consistent environment setup  

This phase produces clean training data, stable detection modules, and ready-to-deploy components for the final real-time emotion recognition system.

---

## Outputs

Both phases save results in their respective `output/` folders, including:
- Trained models (`.keras`)  
- Evaluation metrics  
- Detection/cropping results  
- Visualization images  

---

*This project was developed for the Graph Theory & Algorithms for Computer Science course at KNTU (Spring 2024).*
