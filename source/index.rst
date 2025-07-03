=========================================================
SEGMA-VISION-INDUSTRY: Voice-Activated Object Detection
=========================================================

.. note::
   Welcome to the official documentation for **SEGMA-VISION-INDUSTRY**. This application leverages state-of-the-art AI to provide a seamless, voice-activated object detection and segmentation experience.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

..
   Since this is a single index.rst file, the toctree is here for structure.
   If you add more .rst files later (e.g., install.rst, usage.rst),
   you would list them here.

Overview
========

**SEGMA-VISION-INDUSTRY** is an advanced Streamlit web application that revolutionizes object detection by enabling users to command the system with their voice. It listens to your instructions, identifies the objects you want to find, and then performs real-time detection and segmentation on a live video stream using powerful vision models like YOLO-World and SAM.

In addition to its core detection capabilities, the application includes a powerful, integrated feature for **fine-tuning** the underlying vision model on custom datasets. This allows for the creation of specialized models for highly accurate, domain-specific object recognition, all through a user-friendly interface.

Key Features
------------

*   **Voice-Powered**: Simply speak the names of objects you want to detect.
*   **Real-Time Performance**: Live detection and segmentation on a webcam feed.
*   **Zero-Shot Detection**: Uses YOLO-World for open-vocabulary detection, finding objects even without pre-defined classes.
*   **High-Quality Segmentation**: Integrates the Segment Anything Model (SAM) for precise instance masks.
*   **Integrated Model Fine-Tuning**: A key feature allowing you to train a specialized model on your own data directly through the UI.
*   **Flexible Training**: Supports both local (CPU/GPU) training and the generation of cloud-ready scripts (for Kaggle/Colab).
*   **Interactive UI**: Built with Streamlit for an intuitive user experience.

Installation and Setup
======================

Follow these steps carefully to get the application running on your local machine.

Prerequisites
-------------

You must have the following software installed:

*   **Python 3.8+**
*   **Git** (for cloning the repository)
*   **Ollama**: This project uses a local LLM via Ollama for processing voice commands.
    1.  Download and install Ollama from `https://ollama.com/`.
    2.  Once installed, run the following command in your terminal to pull the required model:

        .. code-block:: bash

           ollama pull gemma3:1b

        .. warning::
           The application will not work without Ollama running and the ``gemma3:1b`` model being available. You can verify this by running ``ollama list`` in your terminal.

Step 1: Clone the Repository
----------------------------

Open your terminal and clone the project from GitHub.

.. code-block:: bash

   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd SEGMA-VISION-INDUSTRY # Or your project's folder name

Step 2: Create a Python Environment
-----------------------------------

It is highly recommended to use a virtual environment.

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Step 3: Install Dependencies
----------------------------

The project relies on several powerful AI/ML libraries. Install them using pip:

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install streamlit ultralytics "supervision[video]" transformers "segment-anything-py" opencv-python numpy requests pyyaml
   pip install sounddevice scipy # Optional, for live microphone recording

.. note::
   The first command installs PyTorch with CUDA 11.8 support. Adjust the ``cu118`` part based on your system's CUDA version, or remove it to install the CPU-only version if you don't have an NVIDIA GPU.

Step 4: Download AI Model Checkpoints
-------------------------------------

The application requires pre-trained model weights for YOLO-World and SAM.

1.  **YOLO-World L Model**: Download ``yolov8l-world.pt`` from the official `Ultralytics YOLOv8 repository <https://github.com/ultralytics/ultralytics>`_.
2.  **Segment Anything Model (ViT-H)**: Download ``sam_vit_h_4b8939.pth`` from the `SAM project page <https://github.com/facebookresearch/segment-anything#model-checkpoints>`_.

Step 5: Configure Model Paths
-----------------------------

.. important::
   This is a critical step. You must update the script with the correct paths to the models you just downloaded.

Open the main Python script (e.g., ``app.py``) and locate the **GLOBAL CONFIGURATIONS** section. Modify the following lines to point to the absolute paths of your downloaded files:

.. code-block:: python

   # Find and update these lines
   YOLO_WORLD_CHECKPOINT_PATH_INFERENCE = r"C:/path/to/your/yolov8l-world.pt"
   SAM_CHECKPOINT_PATH = r"C:/path/to/your/sam_vit_h_4b8939.pth"

How to Run the Application
==========================

Ensure your Ollama service is running in the background. Then, from your project directory (with the virtual environment activated), run the following command:

.. code-block:: bash

   streamlit run your_script_name.py

Your web browser should automatically open with the application interface.

User Guide
==========

The application interface is navigated using the sidebar. The primary function is "Live Detection," with "Model Fine-tuning" available as a powerful tool to enhance the system's capabilities.

Live Detection (Core Workflow)
------------------------------

This is the main screen for real-time, voice-activated object detection.

**1. Set Detection Classes via Voice:**

*   In the sidebar, choose your audio input method:
    *   **Live Recording**: Select a recording duration and click "🎤 Record & Extract Classes". Speak the names of the objects you want to detect (e.g., "person, car, fire extinguisher").
    *   **Upload Audio File**: Upload a pre-recorded WAV or MP3 file and click "🗣️ Process Uploaded Audio".
*   The app will transcribe your audio using Whisper, send the text to Ollama to extract object names, and display the final class list.
*   **Open-Vocabulary Mode**: If you don't provide any classes, the model will attempt to detect any object it recognizes.

**2. Start the Detection:**

*   Once your classes are set (or you choose to run in open-vocabulary mode), click the **🚀 Start Real-Time Detection** button.
*   The application will access your webcam and display the live feed.
*   Detected objects will be highlighted with bounding boxes, segmentation masks, and labels.

**3. Stop the Detection:**

*   Click the **🛑 Stop Detection** button to stop the webcam feed.

**4. Using a Fine-Tuned Model:**

*   If you have trained a custom model using the fine-tuning feature, you can activate it for detection. The sidebar will indicate which model is currently active.
*   You can reset to the default YOLO-World model at any time from the sidebar.

Feature: Custom Model Fine-Tuning
---------------------------------

This page provides a guided workflow for fine-tuning the YOLO-World model on a custom dataset.

**Step 1: Upload Your Dataset**

*   Navigate to the "Model Fine-tuning" page from the sidebar.
*   Your dataset must be in a **ZIP file** with a specific folder structure (the standard YOLOv5/v8 format).

    .. code-block:: text

       your_dataset.zip
       └── your_dataset_folder/
           ├── train/
           │   ├── images/
           │   │   ├── img1.jpg
           │   │   └── img2.jpg
           │   └── labels/
           │       ├── img1.txt
           │       └── img2.txt
           ├── valid/  (or test/)
           │   ├── images/
           │   │   ├── img3.jpg
           │   │   └── ...
           │   └── labels/
           │       ├── img3.txt
           │       └── ...
           └── data.yaml  (Optional, the app generates its own)

*   Click "Choose a ZIP file" and upload your dataset. The app will automatically extract and validate its structure.

**Step 2: Define Parameters**

*   **Class Names**: Enter the names of your object classes, one per line, in the correct order (class 0, class 1, etc.).
*   **Experiment Name**: Give your training run a unique name.
*   **Hyperparameters**: Adjust basic settings like epochs, batch size, and image size. Advanced users can expand the "Advanced Hyperparameters" section to fine-tune learning rates, momentum, and more.
*   Click **Confirm & Proceed**.

**Step 3: Choose Fine-Tuning Mode**

You have two options for training:

1.  **Tune Locally**:
    *   This will run the training process directly on your machine (using your CPU or GPU).
    *   The app will generate and display the ``.yaml`` configuration file it will use.
    *   Click **🚀 Launch Local Fine-tuning** to begin. Monitor your terminal/console for detailed training logs.
    *   The UI will show a "Training in Progress" status.

2.  **Generate Cloud Script**:
    *   This option creates a Python script tailored for cloud environments like Kaggle or Google Colab, which often provide free GPU access.
    *   Click **📄 Generate Python Script**.
    *   The script will appear on the screen. **You must edit one line** in the script to point to your dataset's path within the cloud environment.
    *   Download the script and upload it along with your dataset to your chosen cloud platform.

**Step 4: View and Use Your New Model**

*   After local training is complete, the results page will appear.
*   It will show the path to your best-trained model (``best.pt``).
*   You can then:
    *   **Download Model**: Save the ``best.pt`` file to your computer.
    *   **Use this Fine-Tuned Model for Live Detection**: Click this button to immediately switch the "Live Detection" page to use your newly trained model.

**Resetting the Fine-Tuning Process**

*   At any point, you can click the **Full Finetuning Reset** button in the sidebar to clear all data and start the process over.

Configuration Reference
=======================

The main configurable variables are located at the top of the script.

*   ``DEVICE_PYTORCH``: Automatically detects CUDA GPU or defaults to CPU.
*   ``SPEECH_MODEL_ID``: The Hugging Face model ID for Whisper (default: "openai/whisper-base").
*   ``OLLAMA_API_URL``: The endpoint for your local Ollama instance.
*   ``OLLAMA_MODEL_NAME``: The model tag Ollama should use.
*   ``YOLO_WORLD_CHECKPOINT_PATH_INFERENCE``: **MUST BE EDITED.** Path to the base YOLO-World model.
*   ``SAM_CHECKPOINT_PATH``: **MUST BE EDITED.** Path to the SAM model.

Troubleshooting
===============

*   **Error: "Could not open webcam."**: Ensure your webcam is not being used by another application and that Streamlit has the necessary permissions.
*   **Error: "Ollama API call error"**: Make sure the Ollama application is running on your machine and that the model (e.g., ``gemma3:1b``) has been pulled successfully.
*   **Error: "YOLO-World path invalid"**: Double-check that the paths in the configuration section of the script are correct and point to the actual model files.
*   **Audio Recording Fails**: If live recording doesn't work, ensure you have installed ``sounddevice`` and ``scipy`` and that your microphone is properly configured. Alternatively, use the file upload option.