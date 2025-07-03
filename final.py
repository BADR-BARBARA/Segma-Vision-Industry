# =================================================================================
# ==        FINAL VERSION: COMBINED LIVE DETECTION & FINE-TUNING STREAMLIT APP   ==
# =================================================================================

# --- Core Imports ---
import os
import time
import json
import requests
import traceback
from typing import List, Union
import zipfile
import yaml
import shutil
import threading

# --- AI & ML Framework Imports ---
import torch
import cv2
import numpy as np
import supervision as sv
import streamlit as st
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline
from ultralytics import YOLO, YOLOWorld
from segment_anything import sam_model_registry, SamPredictor

# --- Optional: For Live Audio Recording ---
AUDIO_RECORDING_AVAILABLE = False
try:
    import sounddevice as sd
    from scipy.io.wavfile import write as write_wav
    AUDIO_RECORDING_AVAILABLE = True
except ImportError:
    pass

# ========================================================
# ===========       GLOBAL CONFIGURATIONS      ===========
# ========================================================

# --- General ---
DEVICE_PYTORCH = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- App 1: Live Detection Configurations ---
TORCH_DTYPE_SPEECH = torch.float16 if DEVICE_PYTORCH.type == 'cuda' else torch.float32
SPEECH_MODEL_ID = "openai/whisper-base"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "gemma3:1b" # IMPORTANT: VERIFY this tag against `ollama list`
# This path is the DEFAULT model. The active model can be changed by the user.
YOLO_WORLD_CHECKPOINT_PATH_INFERENCE = r"C:/Users/HP/Desktop/projet_metier/yolov8l-world.pt" 
SAM_CHECKPOINT_PATH = r"C:/Users/HP/Desktop/projet_metier/sam_vit_h_4b8939.pth"
SAM_ENCODER_VERSION = "vit_h"

# --- App 2: Fine-Tuning Configurations ---
TEMP_DATA_DIR = "temp_finetune_data"
WORKING_YAML_NAME = "temp_local_custom_data.yaml"
PROJECT_NAME_FOR_ULTRALYTICS = "finetuned_yolo_world_models"


# ========================================================
# ===========      APP 1: DETECTION LOGIC      ===========
# ========================================================

# --- Model Loading Functions (Detection) ---
@st.cache_resource
def load_whisper_model():
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            SPEECH_MODEL_ID, torch_dtype=TORCH_DTYPE_SPEECH, low_cpu_mem_usage=True, use_safetensors=True
        )
        if DEVICE_PYTORCH.type == 'cuda': model = model.to(DEVICE_PYTORCH)
        processor = AutoProcessor.from_pretrained(SPEECH_MODEL_ID)
        pipe = hf_pipeline(
            "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor, torch_dtype=TORCH_DTYPE_SPEECH,
            device=0 if DEVICE_PYTORCH.type == 'cuda' else -1,
        )
        return pipe
    except Exception as e: st.sidebar.error(f"Whisper load error: {str(e)[:100]}..."); return None

@st.cache_resource
def load_yolo_world_model_for_inference(model_path: str):
    if not os.path.isfile(model_path): st.sidebar.error(f"YOLO-World path invalid: {os.path.basename(model_path)}"); return None
    try: 
        model = YOLO(model_path)
        return model
    except Exception as e: st.sidebar.error(f"YOLO-World load error: {str(e)[:100]}..."); return None

@st.cache_resource
def load_sam_model():
    if not os.path.isfile(SAM_CHECKPOINT_PATH): st.sidebar.error(f"SAM path invalid."); return None
    try:
        sam_model_obj = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE_PYTORCH)
        predictor = SamPredictor(sam_model_obj)
        return predictor
    except Exception as e: st.sidebar.error(f"SAM load error: {str(e)[:100]}..."); return None

# --- Voice Processing Functions (Detection) ---
def record_audio(duration=7, fs=16000, filename_prefix="streamlit_live_input"):
    if not AUDIO_RECORDING_AVAILABLE: st.error("Audio recording libraries not available."); return None
    filename = f"{filename_prefix}_{int(time.time())}.wav"
    progress_bar = st.progress(0); status_text = st.empty()
    status_text.info(f"Recording for {duration} seconds... Speak now!")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        for i in range(duration): time.sleep(1); progress_bar.progress((i + 1) / duration)
        sd.wait(); write_wav(filename, fs, recording)
        status_text.success(f"Recording complete. Audio saved to {filename}")
        progress_bar.empty(); return filename
    except Exception as e: status_text.error(f"Audio recording error: {e}"); progress_bar.empty(); return None

def call_ollama_llm(prompt_text: str, status_placeholder) -> Union[str, None]:
    payload = {"model": OLLAMA_MODEL_NAME, "prompt": prompt_text, "stream": False, "options": {"temperature": 0.1, "num_predict": 100}}
    status_placeholder.info(f"Sending prompt to Ollama ({OLLAMA_MODEL_NAME})...")
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        response_data = response.json()
        if "response" in response_data: status_placeholder.success("LLM response received."); return response_data["response"].strip()
        if "error" in response_data: status_placeholder.error(f"Ollama API Error: {response_data['error']}")
        else: status_placeholder.warning(f"'response' key not in Ollama output.")
        return None
    except requests.exceptions.RequestException as e: status_placeholder.error(f"Ollama API call error: {e}"); return None

def get_classes_from_voice_streamlit(speech_pipe_st, use_live_rec: bool, audio_buffer=None, rec_duration: int = 7, status_ph=None) -> List[str]:
    transcribed_text = None; object_classes = []; temp_audio_file = None
    if status_ph is None: status_ph = st.empty()
    if not speech_pipe_st: status_ph.error("Speech pipeline unavailable."); return []
    if use_live_rec:
        temp_audio_file = record_audio(duration=rec_duration)
        if not temp_audio_file: return []
    elif audio_buffer:
        temp_audio_file = f"uploaded_audio_{int(time.time())}.wav"
        with open(temp_audio_file, "wb") as f: f.write(audio_buffer.getbuffer())
    else: return []
    try:
        result = speech_pipe_st(temp_audio_file, generate_kwargs={"language":"english"})
        transcribed_text = result["text"].strip()
        st.write(f"**Transcription:** \"{transcribed_text}\"")
    except Exception as e: status_ph.error(f"Whisper transcription error: {e}"); return []
    if transcribed_text:
        prompt = f"""<start_of_turn>user
You are an assistant that extracts physical object names from text for object detection. List object names separated by commas. No descriptions, colors, quantities, or verbs. Output format: object1,object2,object3 (or empty if none).
Text: "{transcribed_text}"<end_of_turn><start_of_turn>model"""
        generated = call_ollama_llm(prompt, status_ph)
        if generated:
            object_classes = [item.strip() for item in generated.split(',') if item.strip()]
            status_ph.success(f"Extracted Classes: {object_classes if object_classes else 'None'}")
            st.session_state.extracted_classes = object_classes
    if temp_audio_file and os.path.exists(temp_audio_file): os.remove(temp_audio_file)
    return st.session_state.extracted_classes

# --- Real-time Detection Functions ---
def segment_image_with_sam_st(sam_pred_st: SamPredictor, img_rgb: np.ndarray, boxes: np.ndarray) -> Union[np.ndarray, List]:
    if boxes.shape[0] == 0: return np.array([])
    sam_pred_st.set_image(img_rgb); masks_res = []
    if boxes.ndim == 1: boxes = np.expand_dims(boxes, axis=0)
    for box_coords in boxes:
        try:
            m, s, _ = sam_pred_st.predict(box=box_coords, multimask_output=True)
            if m is not None and s is not None and len(s) > 0: masks_res.append(m[np.argmax(s)])
        except Exception: pass
    if not masks_res: return np.array([])
    return np.array(masks_res)

def run_realtime_detection_streamlit(img_ph, classes_det: List[str], yolo_st: YOLO, sam_st: SamPredictor, box_thresh: float, nms_thresh: float):
    if not yolo_st or not sam_st: st.error("Detection models not loaded."); return
    if classes_det: yolo_st.set_classes(classes_det)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): st.error("Could not open webcam."); return
    box_ann = sv.BoxAnnotator(thickness=2)
    mask_ann = sv.MaskAnnotator(opacity=0.4)
    label_ann = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.BLACK)
    while not st.session_state.get("stop_detection", False):
        ret, frame_cv = cap.read()
        if not ret: time.sleep(0.1); continue
        results = yolo_st.predict(frame_cv, conf=box_thresh, verbose=False, iou=nms_thresh)
        detections_sv = sv.Detections.from_ultralytics(results[0])
        ann_frame = frame_cv.copy()
        if len(detections_sv) > 0:
            if classes_det:
                labels_for_display = [f"{classes_det[cid]} {conf:.2f}" for cid, conf in zip(detections_sv.class_id, detections_sv.confidence)]
                frame_rgb_cv = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                masks_sam = segment_image_with_sam_st(sam_st, frame_rgb_cv, detections_sv.xyxy)
                if isinstance(masks_sam, np.ndarray) and masks_sam.ndim == 3 and masks_sam.shape[0] == len(detections_sv):
                    detections_sv.mask = masks_sam
                    ann_frame = mask_ann.annotate(scene=ann_frame, detections=detections_sv)
                ann_frame = box_ann.annotate(scene=ann_frame, detections=detections_sv)
                ann_frame = label_ann.annotate(scene=ann_frame, detections=detections_sv, labels=labels_for_display)
            else:
                ann_frame = box_ann.annotate(scene=ann_frame, detections=detections_sv)
        img_ph.image(cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB), channels="RGB")
    cap.release(); st.info("Webcam stream stopped.")
    st.session_state.stop_detection = False; st.session_state.detection_running = False

# ========================================================
# ===========     APP 2: FINE-TUNING LOGIC     ===========
# ========================================================

# --- Helper Functions (Fine-Tuning) ---
def cleanup_filesystem_and_data_states():
    if os.path.exists(TEMP_DATA_DIR): shutil.rmtree(TEMP_DATA_DIR)
    st.session_state.data_extracted_and_valid = False
    st.session_state.yaml_for_local_tuning_generated = False
    st.session_state.actual_dataset_paths_relative = None
    st.session_state.dataset_root_for_yaml_abs = None
    if 'current_yaml_path_local' in st.session_state: del st.session_state.current_yaml_path_local

def full_reset_finetuning_app():
    cleanup_filesystem_and_data_states()
    keys_to_reset = [
        'ui_step', 'uploaded_file_name_cache', 'user_class_names', 'current_experiment_name',
        'current_training_params', 'tuning_mode_selection', 'cloud_script_generated',
        'training_active_local', 'training_finished_signal_local', 'training_exception_local',
        'best_model_path_local', 'current_training_thread_obj'
    ]
    for key in keys_to_reset:
        if key in st.session_state: del st.session_state[key]
    initialize_finetuning_default_states()
    st.session_state.uploaded_file_key_tracker = st.session_state.get('uploaded_file_key_tracker', 0) + 1
    st.query_params.clear()

def extract_zip(zip_file_buffer, extract_to_path):
    try:
        with zipfile.ZipFile(zip_file_buffer, 'r') as zip_ref: zip_ref.extractall(extract_to_path)
        return True
    except Exception as e: st.error(f"Error extracting ZIP file: {e}"); return False

def validate_dataset_structure(base_path):
    paths_to_check = {
        "Train Images": os.path.join(base_path, "train", "images"), "Train Labels": os.path.join(base_path, "train", "labels"),
        "Val Test Images": os.path.join(base_path, "test", "images"), "Val Test Labels": os.path.join(base_path, "test", "labels"),
        "Val Valid Images": os.path.join(base_path, "valid", "images"), "Val Valid Labels": os.path.join(base_path, "valid", "labels"),
    }
    missing = []; actual_paths = {"train_images": None, "val_images": None}
    if os.path.exists(paths_to_check["Train Images"]): actual_paths["train_images"] = os.path.join("train", "images")
    else: missing.append("Train Images folder")
    if not os.path.exists(paths_to_check["Train Labels"]): missing.append("Train Labels folder")
    if os.path.exists(paths_to_check["Val Test Images"]):
        actual_paths["val_images"] = os.path.join("test", "images")
        if not os.path.exists(paths_to_check["Val Test Labels"]): missing.append("Test Labels folder")
    elif os.path.exists(paths_to_check["Val Valid Images"]):
        actual_paths["val_images"] = os.path.join("valid", "images")
        if not os.path.exists(paths_to_check["Val Valid Labels"]): missing.append("Valid Labels folder")
    else: missing.append("Validation Images folder (test/ or valid/)")
    if missing: st.error(f"Dataset structure validation failed. Missing: {', '.join(missing)}"); return False, None
    if not actual_paths["train_images"] or not actual_paths["val_images"]: st.error("Could not find both train and validation image directories."); return False, None
    return True, actual_paths

def generate_yaml_for_local_tuning(dataset_base_path_abs, rel_train_img, rel_val_img, classes, yaml_path_abs):
    content = {'path': dataset_base_path_abs, 'train': rel_train_img, 'val': rel_val_img,
               'nc': len(classes), 'names': {i: n.strip() for i, n in enumerate(classes)}}
    try:
        with open(yaml_path_abs, 'w') as f: yaml.dump(content, f, sort_keys=False)
        return True
    except Exception as e: st.error(f"Error generating YAML: {e}"); return False

# --- Training Logic (Fine-Tuning) ---
def run_local_training_thread(yaml_path, class_names, params_dict, exp_name):
    st.session_state.is_thread_actually_running_local = True
    try:
        model = YOLOWorld('yolov8l-world.pt')
        model.set_classes(class_names)
        exp_path = os.path.join(PROJECT_NAME_FOR_ULTRALYTICS, exp_name)
        if os.path.exists(exp_path): shutil.rmtree(exp_path)
        train_args = params_dict.copy()
        train_args.update({'data': yaml_path, 'project': PROJECT_NAME_FOR_ULTRALYTICS, 'name': exp_name, 'device': 'cpu'})
        cleaned_args = {k: v for k, v in train_args.items() if v is not None}
        if cleaned_args.get('optimizer', '').lower() == 'auto': del cleaned_args['optimizer']
        model.train(**cleaned_args)
        st.session_state.best_model_path_local = os.path.join(exp_path, "weights", "best.pt")
        st.session_state.training_exception_local = None
    except Exception as e: st.session_state.training_exception_local = e
    finally:
        st.session_state.training_finished_signal_local = True
        st.session_state.is_thread_actually_running_local = False

def generate_cloud_notebook_script_content(class_names_list, training_params_dict, experiment_name_str):
    epochs = training_params_dict.get('epochs', 25); batch_size = training_params_dict.get('batch', 8); imgsz = training_params_dict.get('imgsz', 640)
    optional_args_list = []
    if training_params_dict.get('lr0') is not None: optional_args_list.append(f"lr0={training_params_dict['lr0']}")
    if training_params_dict.get('patience') is not None: optional_args_list.append(f"patience={training_params_dict['patience']}")
    optional_args_str = ", " + ", ".join(optional_args_list) if optional_args_list else ""
    class_names_repr = repr(class_names_list); num_classes = len(class_names_list)
    return f"""
# Cell 1: Install Ultralytics
!pip install ultralytics pyyaml -q
print("Ultralytics and PyYAML installed.")

# Cell 2: Prepare Dataset (Copy Labels, Modify YAML for Writable Cache Paths)
import yaml, os, shutil
KAGGLE_DATASET_INPUT_ROOT = "/kaggle/input/your-dataset-slug/your-dataset-folder-name" # <--- !!! CHANGE THIS !!!
WORKING_DATA_ROOT = "/kaggle/working/dataset_for_training"; YAML_FILE_PATH_IN_WORKING = os.path.join(WORKING_DATA_ROOT, "data_configured.yaml")
NUM_CLASSES = {num_classes}; CLASS_NAMES = {class_names_repr}; EXPERIMENT_NAME = "{experiment_name_str}"; ULTRALYTICS_PROJECT_DIR = "{PROJECT_NAME_FOR_ULTRALYTICS}"
print(f"Source Kaggle Dataset Root: {{KAGGLE_DATASET_INPUT_ROOT}}")
def setup_dataset_for_writable_cache(source_dataset_root, target_working_root, target_yaml_path):
    if not os.path.isdir(source_dataset_root): print(f"ERROR: Source '{{source_dataset_root}}' not found."); return None
    if os.path.exists(target_working_root): shutil.rmtree(target_working_root)
    os.makedirs(target_working_root, exist_ok=True)
    train_img_rel_path = os.path.join("train", "images"); train_label_rel_path = os.path.join("train", "labels")
    val_img_rel_path = os.path.join("test", "images") if os.path.isdir(os.path.join(source_dataset_root, "test")) else os.path.join("valid", "images")
    val_label_rel_path = os.path.dirname(val_img_rel_path).replace("images", "labels")
    if not os.path.isdir(os.path.join(source_dataset_root, val_img_rel_path)): print(f"ERROR: Neither 'test' nor 'valid' dir found."); return None
    shutil.copytree(os.path.join(source_dataset_root, train_label_rel_path), os.path.join(target_working_root, train_label_rel_path))
    shutil.copytree(os.path.join(source_dataset_root, val_label_rel_path), os.path.join(target_working_root, val_label_rel_path))
    yaml_content = {{"path": os.path.abspath(source_dataset_root), "train": train_img_rel_path, "val": val_img_rel_path, "nc": NUM_CLASSES, "names": {{i: name for i, name in enumerate(CLASS_NAMES)}}}}
    with open(target_yaml_path, "w") as f: yaml.dump(yaml_content, f, sort_keys=False)
    return target_yaml_path
yaml_path_for_training = setup_dataset_for_writable_cache(KAGGLE_DATASET_INPUT_ROOT, WORKING_DATA_ROOT, YAML_FILE_PATH_IN_WORKING)
if yaml_path_for_training: print("Data preparation complete.")

# Cell 3: Fine-tune YOLO-World
from ultralytics import YOLOWorld
import torch
if yaml_path_for_training and os.path.exists(yaml_path_for_training):
    model = YOLOWorld('yolov8l-world.pt'); model.set_classes(CLASS_NAMES)
    device_to_use = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = model.train(data=yaml_path_for_training, epochs={epochs}, batch={batch_size}, imgsz={imgsz}, project=ULTRALYTICS_PROJECT_DIR, name=EXPERIMENT_NAME, device=device_to_use, cache=True{optional_args_str})
    print("--- Training Finished! ---")
"""

# --- State Initialization (Fine-Tuning) ---
def initialize_finetuning_default_states():
    default_states = {
        'ui_step': "upload_dataset", 'uploaded_file_key_tracker': 0, 'uploaded_file_name_cache': None,
        'data_extracted_and_valid': False, 'yaml_for_local_tuning_generated': False,
        'actual_dataset_paths_relative': None, 'dataset_root_for_yaml_abs': None, 'current_yaml_path_local': None,
        'user_class_names': ["smoke", "fire"], 'current_experiment_name': f"run_{int(time.time())}",
        'current_training_params': {
            'epochs': 3, 'batch': 4, 'imgsz': 320, 'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005,
            'optimizer': 'auto', 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
            'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'patience': 10
        },
        'training_active_local': False, 'training_finished_signal_local': False, 'training_exception_local': None,
        'best_model_path_local': None, 'current_training_thread_obj': None, 'is_thread_actually_running_local': False,
    }
    for key, value in default_states.items():
        if key not in st.session_state: st.session_state[key] = value

# ========================================================
# ===========      STREAMLIT PAGE RENDERERS    ===========
# ========================================================
def render_detection_page():
    st.title("🗣️🎙️ Voice-Activated Real-Time Object Detection & Segmentation BADR & YASSIR 📸")
    st.sidebar.header("Active Model Control")
    st.sidebar.info(f"**Current Model:** `{os.path.basename(st.session_state.yolo_model_path_for_inference)}`")
    if st.session_state.yolo_model_path_for_inference != YOLO_WORLD_CHECKPOINT_PATH_INFERENCE:
        if st.sidebar.button("↩️ Reset to Default YOLO-World Model"):
            st.session_state.yolo_model_path_for_inference = YOLO_WORLD_CHECKPOINT_PATH_INFERENCE
            st.rerun()
    st.sidebar.markdown("---")
    if "app_initialized" not in st.session_state:
        st.session_state.whisper_pipeline = load_whisper_model()
        st.session_state.sam_predictor = load_sam_model()
        st.session_state.extracted_classes = []; st.session_state.detection_running = False
        st.session_state.stop_detection = False; st.session_state.app_initialized = True
    with st.spinner("Loading AI models, please wait..."):
        st.session_state.yolo_model = load_yolo_world_model_for_inference(st.session_state.yolo_model_path_for_inference)
    if not st.session_state.yolo_model: st.error("Failed to load the active YOLO model. Cannot proceed."); return
    st.sidebar.header("Voice Input for Classes")
    input_method = st.sidebar.radio("Audio Input:", ("Live Recording", "Upload Audio File"), index=0 if AUDIO_RECORDING_AVAILABLE else 1)
    status_ph = st.sidebar.empty()
    if input_method == "Live Recording" and AUDIO_RECORDING_AVAILABLE:
        rec_duration = st.sidebar.slider("Recording Duration (s):", 3, 15, 7)
        if st.sidebar.button("🎤 Record & Extract Classes"): get_classes_from_voice_streamlit(st.session_state.whisper_pipeline, True, rec_duration=rec_duration, status_ph=status_ph); st.rerun()
    elif input_method == "Upload Audio File":
        audio_file = st.sidebar.file_uploader("Upload Audio (WAV, MP3):", type=["wav", "mp3"])
        if audio_file and st.sidebar.button("🗣️ Process Uploaded Audio"): get_classes_from_voice_streamlit(st.session_state.whisper_pipeline, False, audio_buffer=audio_file, status_ph=status_ph); st.rerun()
    st.sidebar.markdown("---"); st.sidebar.header("Detection Control")
    col_info, col_video = st.columns([2, 3])
    with col_info:
        st.subheader("📝 Detection Status")
        if st.session_state.get("extracted_classes"): st.success(f"**Classes:** `{', '.join(st.session_state.extracted_classes)}`")
        else: st.info("No classes set. Model will run in open-vocabulary mode.")
        if not st.session_state.get("detection_running", False):
            if st.button("🚀 Start Real-Time Detection"):
                st.session_state.detection_running = True; st.session_state.stop_detection = False; st.rerun()
        else:
            if st.button("🛑 Stop Detection"): st.session_state.stop_detection = True; st.rerun()
    with col_video:
        st.subheader("📹 Live Feed")
        video_ph = st.empty()
        if not st.session_state.get("detection_running", False): video_ph.info("Detection not started.")
    if st.session_state.get("detection_running", False) and not st.session_state.get("stop_detection", False):
        run_realtime_detection_streamlit(video_ph, st.session_state.get("extracted_classes", []), st.session_state.yolo_model, st.session_state.sam_predictor, 0.05, 0.5)

def render_finetuning_page():
    st.title("🤖 Automatic YOLO-World Fine-tuning")
    initialize_finetuning_default_states()
    if st.session_state.ui_step == "upload_dataset":
        st.header("Step 1: Upload Dataset ZIP File")
        st.code("ZIP structure: your_dataset/train/[images|labels], your_dataset/valid/[images|labels]")
        uploaded_file = st.file_uploader("Choose a ZIP file", type="zip", key=f"uploader_{st.session_state.uploaded_file_key_tracker}")
        if uploaded_file:
            st.session_state.uploaded_file_name_cache = uploaded_file.name
            if os.path.exists(TEMP_DATA_DIR): shutil.rmtree(TEMP_DATA_DIR)
            os.makedirs(TEMP_DATA_DIR)
            with st.spinner("Extracting and validating..."):
                if extract_zip(uploaded_file, TEMP_DATA_DIR):
                    extracted = os.listdir(TEMP_DATA_DIR)
                    root = os.path.join(TEMP_DATA_DIR, extracted[0]) if len(extracted) == 1 and os.path.isdir(os.path.join(TEMP_DATA_DIR, extracted[0])) else TEMP_DATA_DIR
                    st.session_state.dataset_root_for_yaml_abs = os.path.abspath(root)
                    is_valid, rel_paths = validate_dataset_structure(root)
                    if is_valid: st.session_state.data_extracted_and_valid = True; st.session_state.actual_dataset_paths_relative = rel_paths; st.session_state.ui_step = "define_parameters"; st.rerun()
    elif st.session_state.ui_step == "define_parameters":
        st.header("Step 2: Define Classes & Training Parameters")
        with st.form("params_form"):
            _p = st.session_state.current_training_params
            class_names = st.text_area("Class Names", "\n".join(st.session_state.user_class_names), height=100)
            exp_name = st.text_input("Experiment Name", st.session_state.current_experiment_name)
            st.subheader("Basic Hyperparameters")
            c1,c2,c3=st.columns(3)
            epochs = c1.number_input("Epochs", 1, 1000, _p['epochs'])
            batch = c2.select_slider("Batch Size", [2,4,8,16,32,64], _p['batch'])
            imgsz = c3.select_slider("Image Size", [320,416,640], _p['imgsz'])
            with st.expander("Advanced Hyperparameters (Optional)"):
                c1,c2=st.columns(2); lr0=c1.number_input("LR0", 0.0, 0.1, _p['lr0'], 0.0001, "%.4f"); lrf=c2.number_input("LRF", 0.0, 0.1, _p['lrf'], 0.0001, "%.4f")
                c1,c2,c3=st.columns(3); mom=c1.number_input("Momentum", 0.0, 1.0, _p['momentum'], 0.001, "%.3f"); wd=c2.number_input("Weight Decay", 0.0, 0.01, _p['weight_decay'], 0.0001, "%.4f"); opt=c3.selectbox("Optimizer", ['auto','SGD','Adam','AdamW'], index=['auto','SGD','Adam','AdamW'].index(_p['optimizer']))
                c1,c2,c3=st.columns(3); w_e=c1.number_input("Warmup Epochs",0.0,10.0,_p['warmup_epochs'],0.1,"%.1f"); w_m=c2.number_input("Warmup Momentum",0.0,1.0,_p['warmup_momentum'],0.01,"%.2f"); w_b=c3.number_input("Warmup Bias LR",0.0,1.0,_p['warmup_bias_lr'],0.01,"%.2f")
                c1,c2,c3=st.columns(3); box=c1.number_input("Box Gain",0.0,100.0,_p['box'],0.1,"%.2f"); cls=c2.number_input("Cls Gain",0.0,100.0,_p['cls'],0.01,"%.2f"); dfl=c3.number_input("Dfl Gain",0.0,100.0,_p['dfl'],0.1,"%.2f")
            if st.form_submit_button("Confirm & Proceed"):
                st.session_state.user_class_names = [c.strip() for c in class_names.split('\n') if c.strip()]
                st.session_state.current_experiment_name = exp_name.replace(" ", "_")
                st.session_state.current_training_params = {'epochs':epochs,'batch':batch,'imgsz':imgsz,'lr0':lr0 or None,'lrf':lrf or None,'momentum':mom or None,'weight_decay':wd or None,'optimizer':opt,'warmup_epochs':w_e or None,'warmup_momentum':w_m or None,'warmup_bias_lr':w_b or None,'box':box or None,'cls':cls or None,'dfl':dfl or None,'patience':_p['patience']}
                st.session_state.ui_step = "choose_mode"; st.rerun()
    elif st.session_state.ui_step == "choose_mode":
        st.header("Step 3: Choose Fine-tuning Mode")
        mode = st.radio("How to fine-tune?", ("Tune Locally", "Generate Cloud Script"))
        if mode == "Tune Locally":
            yaml_path = os.path.abspath(os.path.join(TEMP_DATA_DIR, WORKING_YAML_NAME))
            if generate_yaml_for_local_tuning(st.session_state.dataset_root_for_yaml_abs, st.session_state.actual_dataset_paths_relative['train_images'], st.session_state.actual_dataset_paths_relative['val_images'], st.session_state.user_class_names, yaml_path):
                st.session_state.current_yaml_path_local = yaml_path
                st.code(open(yaml_path).read(), language='yaml')
                if st.button("🚀 Launch Local Fine-tuning"):
                    thread = threading.Thread(target=run_local_training_thread, args=(yaml_path, st.session_state.user_class_names, st.session_state.current_training_params, st.session_state.current_experiment_name))
                    thread.start(); st.session_state.current_training_thread_obj = thread
                    st.session_state.ui_step = "local_training_inprogress"; st.rerun()
        elif mode == "Generate Cloud Script":
            if st.button("📄 Generate Python Script"):
                script = generate_cloud_notebook_script_content(st.session_state.user_class_names, st.session_state.current_training_params, st.session_state.current_experiment_name)
                st.code(script, language='python'); st.download_button("Download Script", script, f"finetune.py")
    elif st.session_state.ui_step == "local_training_inprogress":
        st.header("🚀 Local Training In Progress..."); st.info("Monitor console for logs.")
        with st.spinner("Model fine-tuning is underway..."):
            if st.session_state.get('training_finished_signal_local'): st.session_state.ui_step = "show_results"; st.rerun()
            else: time.sleep(2); st.rerun()
    elif st.session_state.ui_step == "show_results":
        st.header("🏁 Local Training Results")
        ex = st.session_state.get('training_exception_local'); model = st.session_state.get('best_model_path_local')
        if ex: st.error(f"Training failed: {ex}")
        elif model and os.path.exists(model):
            st.success("Fine-tuning complete!"); st.markdown(f"**Model Path:** `{os.path.abspath(model)}`")
            st.markdown("---")
            if st.button("✅ Use this Fine-Tuned Model for Live Detection"):
                st.session_state.yolo_model_path_for_inference = os.path.abspath(model)
                st.success(f"Model '{os.path.basename(model)}' is now active. Switch to the 'Live Detection' page to use it.")
            st.markdown("---")
            with open(model, "rb") as fp: st.download_button("Download Model (best.pt)", fp, "finetuned_best.pt")
        else: st.warning("Training finished, but model path not found.")
        if st.button("Start Over"): full_reset_finetuning_app(); st.rerun()
    st.sidebar.markdown("---")
    if st.sidebar.button("Full Finetuning Reset"): full_reset_finetuning_app(); st.rerun()

# ========================================================
# ===========           MAIN APP ROUTER          ===========
# ========================================================
def main():
    st.set_page_config(page_title="Segma Vision Pro", layout="wide")
    if 'yolo_model_path_for_inference' not in st.session_state:
        st.session_state.yolo_model_path_for_inference = YOLO_WORLD_CHECKPOINT_PATH_INFERENCE
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose an application", ["Live Detection", "Model Fine-tuning"])
    st.sidebar.markdown("---")
    if page == "Live Detection":
        render_detection_page()
    elif page == "Model Fine-tuning":
        render_finetuning_page()

if __name__ == "__main__":
    main()