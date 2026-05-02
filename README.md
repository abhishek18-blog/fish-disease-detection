<div align="center">
  <img src="banner.svg" alt="Smart Fish Disease Detection Banner" width="100%" />
</div>

# Smart Fish Disease Detection & Identification System

## 🐟 Overview
The **Smart Fish Disease Detection & Identification System** is a Deep Learning-based diagnostic tool designed for the aquaculture industry. It bridges the gap between expert knowledge and daily fish care by leveraging advanced Convolutional Neural Network (CNN) architectures to automate the identification of fish diseases from imagery. The system classifies diseases into four major categories: Bacterial, Viral, Fungal, and Parasitic.

**Developed by:** Rochelle D’Souza & Abhishek Deshmukh  


---

## 🎯 Problem Statement
In the aquaculture industry, fish health is the primary determinant of economic success. However, professional aquatic veterinarians are scarce and prohibitively expensive for small-to-midscale farmers and aquariums. Non-expert caregivers often rely on subjective visual inspections, leading to frequent misdiagnosis due to overlapping physical symptoms of different pathogens. Traditional diagnostic methods (microscopy, biopsy) are time-consuming, invasive, and costly.

---

## 🚀 Key Objectives
- **Automated Feature Learning:** Develop a Deep Learning framework to learn high-level representations (edges, textures, lesions) directly from fish disease imagery.
- **Comparative Architecture Analysis:** Conduct a comparative study of multiple CNN architectures to determine the optimal balance between diagnostic accuracy and computational efficiency.
- **Multi-Category Classification:** Identify major disease categories: **Bacterial, Viral, Fungal, Parasitic**, alongside healthy classifications.
- **Performance Evaluation:** Analyze model reliability through convergence curves and F1-scores to ensure generalizability across different water conditions and imaging qualities.
- **Deployment via Web Framework:** Build and deploy a functional web application to serve the models, allowing users to obtain automated diagnostic reports from uploaded images.

---

## 🏗️ System Architecture

### Multi-Model Approach
The system leverages a robust fail-safe mechanism by comparing the predictions of four distinct CNN architectures:
1. **EfficientNet-B0**
2. **DenseNet-121**
3. **ResNet-50**
4. **MobileNetV2**

### Backend & Inference
- **Framework:** FastAPI
- **Environment:** PyTorch execution environment.
- **Operation:** All four models are loaded into memory during server initialization for rapid, asynchronous inference.
- **Pre-processing Pipeline:** Standardized resizing to `224 × 224 pixels` and normalization using ImageNet statistics for consistent feature extraction.

### Application Interface (Frontend)
A professional, minimalist frontend presents a grid layout displaying comparative predictions and confidence percentages of each model side-by-side. Discrepancies between models serve as a diagnostic indicator for potential secondary infections or complex cases, ensuring transparency in the AI's decision-making process.

---

## 🔮 Future Scope
- **IoT Integration & Real-Time Monitoring:** 
  - Live Stream Diagnosis using Raspberry Pi or ESP32-CAM.
  - Sensor Fusion combining visual data with water quality sensors (pH, temperature, dissolved oxygen, and ammonia levels) to predict outbreaks before physical symptoms appear.
- **Edge Deployment & Offline Capabilities:**
  - Mobile optimization using TensorFlow Lite or ONNX to compress models for local execution on mobile devices.
  - Development of a Progressive Web App (PWA) for field usage without internet connectivity.

---

## 📚 References
- Kheriji, L., Kouadri, A., & Mansouri, M. (2025). Deep learning-based fish health monitoring and diagnosis: A review. IEEE Access.

<br />

<div align="center">
  <p>
    Made with curosity by <b>Rochelle D’Souza</b> & <b>Abhishek Deshmukh</b>
  </p>
  <p>
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  </p>
  <p>
    <i>Empowering Aquaculture with AI-Driven Health Diagnostics</i>
  </p>
</div>
