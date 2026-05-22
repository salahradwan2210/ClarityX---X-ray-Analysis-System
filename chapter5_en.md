# Chapter 5: DIAGNOLINK - System Implementation and Web Application

## 5.1 Overview

While previous chapters focused on the methodology and performance evaluation of the core AI model, this chapter details the translation of that research into a tangible, usable clinical tool: the DiagnoLink Web Application. The development of a user-centric application is a critical step in bridging the gap between theoretical accuracy and real-world clinical utility.

This chapter provides a comprehensive technical overview of the full-stack system, detailing the architecture, technologies, and features that bring AI-powered diagnosis to the end-user, transforming theoretical research into a practical tool that can be used in everyday clinical settings.

## 5.2 Technical Architecture

The DiagnoLink system is architected as a modern, robust client-server application. This design ensures a clear separation of concerns between the user interface and the backend processing, leading to a more scalable and maintainable system.

### 5.2.1 Key Components

The three primary components of the architecture are:

- **Frontend (Client)**: A dynamic and responsive web interface built with Next.js 15.x (React), responsible for all user interactions.
- **Backend (Server)**: A high-performance analysis server built with Python and FastAPI, which hosts and runs the AI model.
- **Database**: A secure and scalable cloud database provided by Supabase, which uses a PostgreSQL instance for data persistence.

The core AI model integrated into this system is the high-performing ConvNeXt Large CNN identified in our comparative evaluation.

### 5.2.2 System Diagram

The following figure illustrates the general structural architecture of the DiagnoLink system:

![Technical Architecture Diagram of DiagnoLink System]

## 5.3 Frontend Implementation: User Interface Design and Interaction

The frontend was developed to be intuitive for clinicians, enabling them to access the power of the AI model with minimal friction.

### 5.3.1 Technology Stack

The user interface was built using a modern, production-grade technology stack chosen for its performance, scalability, and developer experience:

- **Framework**: Next.js 15.2.4 was selected for its powerful features, including server-side rendering (SSR) for fast initial page loads and a robust routing system.
- **Language**: TypeScript was used to enforce type safety, significantly reducing runtime errors and improving code maintainability.
- **UI Components**: A custom component library was built using Tailwind CSS, a utility-first framework that allows for rapid development of clean, consistent user interfaces. State management was handled efficiently using native React Hooks.
- **Additional Libraries**: Framer Motion was integrated to provide fluid animations and transitions, enhancing the user experience. For reporting, React-PDF was used to generate dynamic, client-side PDF reports.

### 5.3.2 Application Structure and Key Pages

The application is structured around a clear user journey, with several key pages:

#### Login Page (/login)

Provides a secure authentication form using JWT, with client-side validation for immediate feedback.

![Login Page]

#### Dashboard (/dashboard)

Serves as the central hub after login, displaying user statistics and providing quick navigation to the main sections of the application.

![Main Dashboard]

#### Patient Management (/patients)

This section contains a searchable and filterable list of all patients. It also includes the form for adding a new patient record.

![Patient Management Page]

#### Patient Details (/patients/[id])

A dedicated page for each patient, displaying their personal information, medical history, and a list of all past X-ray analyses.

![Patient Details Page]

#### Analysis & Results Pages (/analysis, /results/[id])

This is the core workflow, where a user can upload a new X-ray image, enter the required metadata, and view the results. The results page is organized into intuitive tabs:
- "Findings"
- "Localization" (with heatmap overlay)
- "Description"
- "3D Reconstruction"

with an option to generate a final PDF report.

![Analysis Results Page]

## 5.4 Backend Implementation: The Analysis Engine

The backend is a high-performance Python server responsible for all the heavy lifting, including image processing and AI inference.

### 5.4.1 API and Server

FastAPI was chosen as the web framework for the backend due to its extremely high performance, asynchronous capabilities, and automatic generation of interactive API documentation. The server uses OpenCV and PIL for image manipulation and NumPy for numerical operations. It exposes two primary API endpoints:

- **/predict**: Receives the image and metadata, runs the full analysis pipeline, and returns the results in a structured JSON format.
- **/healthcheck**: A simple endpoint to verify that the server and the AI model are loaded and running correctly.

### 5.4.2 AI Model Integration

The backend loads the trained ConvNeXt Large model. This model features a multi-head architecture, with a classification head predicting the 14 disease probabilities and a separate localization head predicting bounding boxes for 8 of those diseases. The model's design, which includes an attention mechanism and a dedicated branch for processing patient metadata (age, gender, view position), is fully utilized by the backend server to generate its comprehensive predictions.

![Image Processing Flow Diagram in Backend Server]

## 5.5 Database and Security

### 5.5.1 Database Architecture

We utilized Supabase, a Backend-as-a-Service platform, which provides a scalable PostgreSQL database along with authentication and storage services out of the box. The database schema is designed with clear relationships to ensure data integrity:

- **Tables**: The primary tables include users (for doctor/admin accounts), patients (demographics and history), analyses (records of each X-ray scan), and results (the corresponding AI model outputs).
- **Relationships**: The data is linked via relational constraints, such as a one-to-many relationship between a patient and their analyses.

![Database Schema Diagram]

### 5.5.2 Security and Authentication

Security is paramount. The system implements a robust security model based on JSON Web Tokens (JWT) for authentication. User passwords are never stored in plain text; they are hashed using the industry-standard bcrypt algorithm with a salt. All API endpoints are protected and require a valid JWT, and Cross-Origin Resource Sharing (CORS) is configured to only allow requests from the designated frontend application, preventing unauthorized access.

## 5.6 System Specifications and Requirements

The system is designed with clear performance targets and requirements to ensure a smooth user experience.

### 5.6.1 Image Processing

The application supports standard medical (DICOM) and common image formats (JPEG, PNG). All images are processed into a 512×512 tensor and normalized using the standard ImageNet mean and standard deviation before being fed to the model.

### 5.6.2 Performance Targets

The end-to-end analysis time is designed to be under 5 seconds per image on appropriate hardware. The system aims to deliver the high diagnostic accuracy achieved during research, with a target mean AUC of ~97%.

### 5.6.3 System Requirements

The server requires a multi-core CPU and at least 16GB of RAM, with an optional CUDA-compatible GPU for accelerated performance. The client-side application requires only a modern web browser.

### 5.6.4 Scalability

The system is built with a modular design and an open API, facilitating future integration with hospital systems and allowing for horizontal scaling by running multiple instances of the backend service.

## 5.7 Key Features and Capabilities

### 5.7.1 Image Upload and Processing

The system allows users to easily upload X-ray images, with support for multiple formats. Images are automatically validated and quality-checked before processing.

### 5.7.2 AI-Powered Image Analysis

The system uses the advanced ConvNeXt Large model to analyze X-ray images and identify 14 different pathological conditions, providing probabilities for each diagnosis.

### 5.7.3 Visual Representation of Results

The system provides several ways to visually represent results:
- **Heatmap**: Highlights suspicious areas on the image with a color gradient.
- **Localization**: Bounding boxes showing the location of each detected disease.
- **3D Reconstruction**: An interactive 3D view of the image allowing better visualization of pathological patterns.

### 5.7.4 Report Generation

Users can create comprehensive PDF reports that include:
- Patient information
- Original X-ray image
- Analysis results with probabilities
- Visual representation of suspicious areas
- Physician's notes
- Comparisons with previous examinations (if available)

### 5.7.5 Patient and Record Management

The system provides complete patient management functionality, including:
- Adding, editing, and viewing patient records
- Tracking medical history and examination records
- Advanced patient search and filtering
- Comparing examinations over time to monitor the evolution of conditions

## 5.8 User Experience and Design

The user interface was designed with a focus on clarity, efficiency, and ease of use, taking into account the needs of medical professionals:

### 5.8.1 Design Principles

- **Simplicity**: Clean, distraction-free interface to facilitate focus on essential tasks.
- **Consistency**: Consistent UI elements across the application to enhance learnability.
- **Efficiency**: Workflow design to minimize the number of clicks and actions required.
- **Responsiveness**: The application works smoothly on all screen sizes, from desktop to tablet devices.

### 5.8.2 Key Design Elements

- Comprehensive dashboard with an overview of the most important data
- Intuitive navigation with quick access to key functions
- Organized display of results with different layers of detail
- Use of colors and icons to enhance data understanding
- Interactive elements (such as zoom/rotate) for controlling image display

## 5.9 Conclusion

The DiagnoLink application represents a successful transformation of advanced AI research into a practical clinical tool. By combining a high-performance deep learning model with a carefully designed user interface and robust backend system, DiagnoLink provides valuable diagnostic support to clinical physicians.

The system achieves its core objectives:
- Improving the accuracy of chest disease diagnosis through X-ray
- Reducing diagnostic time
- Providing a visual and interactive tool that aids in clinical decision-making
- Seamlessly integrating X-ray analysis workflow into the existing medical ecosystem

In the next chapter, we will discuss the results of clinical trials of the system and feedback from specialized users, as well as future directions for system development.
