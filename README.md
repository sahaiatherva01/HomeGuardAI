# HomeGuard AI

### *Real-Time Human Pose Tracking and Intelligent Activity Monitoring System*

---

## Overview

**HomeGuard AI** is a real-time Computer Vision application designed to monitor human movement using pose estimation and intelligent activity analysis.

The system detects multiple people through a live webcam feed, tracks body landmarks, visualizes robotic skeleton overlays, and continuously analyzes movement patterns to identify potentially suspicious behaviors such as prolonged standing, face concealment, excessive camera attention, and abnormal posture.

Beyond pose detection, HomeGuard AI provides an interactive analytics dashboard that displays real-time performance metrics, occupancy statistics, motion analysis, historical session comparisons, and suspicious activity reports.

The project demonstrates how Computer Vision can be applied to smart surveillance while maintaining an intuitive and modern user experience.

> *"Understanding movement is the first step toward building intelligent surveillance systems."*

---

# Features

### Computer Vision

- Real-Time Multi-Person Pose Detection
- Human Pose Tracking using MediaPipe
- Stable Person Tracking across Frames
- Robot-Style Skeleton Visualization
- Live Webcam Processing
- Up to Four Person Detection
- Real-Time Landmark Rendering

### Activity Monitoring

- Standing Still Detection
- Looking Towards Camera Detection
- Face Hidden Detection
- Frequent Head Movement Analysis
- Suspicious Activity Detection
- Region of Interest (ROI) Monitoring
- Motion Intensity Analysis

### Analytics Dashboard

- Live FPS Monitoring
- Detection Confidence Score
- Motion Tracking
- Active Person Counter
- Session Duration Tracking
- Historical Performance Charts
- Suspicious Event Timeline
- Occupancy Statistics
- Session Comparison Dashboard
- Heatmap Visualization
- Risk Level Summary

### User Experience

- Responsive Dashboard
- Dark & Light Theme
- Live Charts
- Interactive Controls
- Real-Time Status Indicators
- Exportable Session Reports

---

# Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend Logic |
| Flask | Web Server & API |
| OpenCV | Image Processing |
| MediaPipe | Human Pose Estimation |
| NumPy | Mathematical Operations |
| HTML5 | Frontend Structure |
| CSS3 | Responsive UI & Animations |
| JavaScript | Dashboard Interactivity |
| Chart.js | Analytics Visualization |

---

# How It Works

The system follows a real-time processing pipeline:

1. The webcam continuously captures live video.
2. MediaPipe detects one or multiple people.
3. Human body landmarks are extracted.
4. Stable tracking assigns identities across frames.
5. Pose landmarks are rendered as robotic skeletons.
6. Motion and posture are analyzed.
7. Suspicious behaviors are identified using rule-based analysis.
8. Live statistics are sent to the dashboard.
9. Charts and analytics update continuously in real time.

---

# System Architecture

```text
Webcam Input
      │
      ▼
Video Capture
      │
      ▼
MediaPipe Pose Detection
      │
      ▼
Multi-Person Tracking
      │
      ▼
Landmark Processing
      │
      ▼
Motion Analysis
      │
      ▼
Suspicious Activity Detection
      │
      ▼
Analytics Engine
      │
      ▼
Flask API
      │
      ▼
Interactive Dashboard
```

---

# Key Features Demonstrated

## Computer Vision

- Human Pose Estimation
- Landmark Tracking
- Multi-Person Detection
- Pose Visualization
- Motion Analysis

## Artificial Intelligence

- Human Activity Recognition
- Rule-Based Behaviour Analysis
- Movement Pattern Detection
- Risk Assessment

## Backend Development

- REST APIs using Flask
- Real-Time Video Streaming
- Backend State Management
- Session Analytics

## Frontend Development

- Responsive Dashboard
- Live Charts
- Dynamic UI Updates
- Theme Switching
- Interactive Analytics

---

# Project Structure

```text
HomeGuard-AI/
│
├── server.py              # Flask backend
├── index.html             # Dashboard interface
├── pose_landmarker.task   # MediaPipe model
├── assets/                # Images & icons
├── README.md
```

---

# Performance

The current implementation is optimized for:

- Real-Time Processing
- Low Latency Detection
- Smooth Multi-Person Tracking
- Stable Landmark Rendering
- Lightweight CPU Execution
- Live Analytics Dashboard

---

# Future Enhancements

- YOLO-Based Person Detection
- DeepSORT Person Re-Identification
- Deep Learning Activity Recognition
- Heatmap Generation
- Cloud Event Storage
- AI-Based Risk Scoring
- Mobile Notifications
- Multi-Camera Support
- Face Recognition Integration
- Person Re-Identification Across Cameras

---

# What I Learned

Building HomeGuard AI provided practical experience with:

- Computer Vision Pipelines
- Human Pose Estimation
- MediaPipe Framework
- OpenCV Image Processing
- Flask Backend Development
- Real-Time Dashboard Design
- Human Activity Analysis
- Frontend Visualization
- Performance Optimization

It also strengthened my understanding of how real-time AI systems integrate perception, analytics, and user interfaces into a complete application.

---

# Why This Project Matters

Modern surveillance systems increasingly rely on intelligent vision rather than traditional CCTV monitoring.

HomeGuard AI demonstrates how Computer Vision can automatically interpret human movement, identify potentially suspicious behavior, and provide meaningful real-time insights through an interactive analytics dashboard.

The project combines AI, Computer Vision, backend development, and frontend engineering into a complete end-to-end intelligent monitoring system.

---

# License

This project is intended for educational purposes and demonstrates the practical application of Computer Vision, Human Pose Estimation, and Intelligent Activity Monitoring using modern AI technologies.
