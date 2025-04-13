from termcolor import colored
import numpy as np
import win32gui, win32ui, win32con
import torch
import serial
import time
import keyboard
import pathlib
import math
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
import kmNet
import threading
import json
import os
import winsound
import aimbot # Make sure aimbot is imported
from collections import deque
import queue
from threading import Event, Lock
from nicegui import ui
import webbrowser

# Define a single unified configuration with ALL necessary keys
DEFAULT_CONFIG = {
    # --- Meta Settings ---
    "exit_key": "end",
    "config_save_path": "config.txt",

    # --- Detection Settings ---
    "confidence": 0.5, # Adjusted default based on Valorant-A.I
    "fov": 320, # Capture FOV size (square)
    "target_priority": "center", # Options: "center", "distance", "size"
    "enemy_color_enabled": False, # Toggle for color check
    "lower_color": [230, 40, 40], # Lower bound for RED in RGB (Note: Valorant-A.I used BGR, check if conversion needed)
    "upper_color": [255, 90, 104], # Upper bound for RED in RGB

    # --- Aimbot General Settings ---
    "aimbot_enabled": True,
    "activation_range": 100.0, # Max distance in pixels from center to activate aimbot/trigger
    "valorant_sensitivity": 0.3,
    "aim_height_adjustment": 0, # Pixels up/down from center of bbox

    # --- Aimbot Modes & Keys (Using hardcoded kmNet keys for now) ---
    # "aim_key": "right", # Hardcoded to kmNet.isdown_right()
    # "trigger_key": "side2", # Hardcoded to kmNet.isdown_side2()
    "aimbot_mode": "normal", # Options: "normal", "rage", "flick", "silent"

    # --- Normal/Follow Mode Settings (Current System) ---
    "follow_mode_threshold_px": 30.0, # Distance to switch to follow steps/smoothing
    "tracking_speed_multiplier": 1.0, # Multiplier specifically for follow mode steps
    "aim_deadzone_px": 1.0,
    "aim_speed_min": 0.5,
    "aim_speed_max": 100.0,
    "prediction_factor": 1.0,
    "estimated_latency_ms": 25,
    "smoothing_alpha_far": 0.95,
    "smoothing_alpha_mid": 0.7,
    "smoothing_alpha_near": 0.3,
    "smoothing_alpha_precision": 0.15,
    "smoothing_transition_far": 80.0,
    "smoothing_transition_mid": 30.0,
    "smoothing_transition_near": 15.0,
    "aim_mode_settings": {
        "normal": {
            "steps": [
                [15, 1.5], [30, 3.0], [80, 8.0], [float('inf'), 10.0]
            ]
        },
        "follow": {
             "steps": [
                [10, 0.8], [25, 1.5], [50, 3.0], [float('inf'), 5.0]
             ]
        },
        # Add placeholders for other modes if needed, though they might bypass steps
        "rage": {"steps": [[float('inf'), 15.0]]}, # Example: High multiplier for rage
        "flick": {"steps": []}, # Not used, handled differently
        "silent": {"steps": []} # Not used, handled differently
    },

    # --- Flick/Silent Specific Settings ---
    "flick_smoothing": 10, # Number of steps for legit flick (lower = faster)
    "flick_cooldown_ms": 50,
    "silent_cooldown_ms": 50,

    # --- Triggerbot Settings ---
    "triggerbot_enabled": True,
    "trigger_delay_ms": 10, # Delay between trigger activation and click

    # --- Anti-Recoil Settings ---
    "anti_recoil_enabled": False,
    "anti_recoil_strength": 0.5, # Vertical pixels to pull down per shot (needs tuning)
    "anti_recoil_delay_ms": 100 # Delay after shot before applying recoil control
}


# Global variables
CONFIG = DEFAULT_CONFIG.copy() # Now uses the comprehensive DEFAULT_CONFIG
fov = CONFIG["fov"] # This should now work
mid = fov / 2
model = None
running = False
detection_times = deque(maxlen=120)
detection_fps = 0
last_fps_update = time.time()
fps_update_interval = 1.0
loop_times = deque(maxlen=120)
loop_fps = 0
last_loop_fps_update = time.time()
detection_event = Event()
detection_lock = Lock()
latest_detection = None
config_lock = threading.Lock()

class ConfigManager:
    def __init__(self, config_path="config.txt"):
        self.config_path = config_path
        # Load config uses the comprehensive DEFAULT_CONFIG defined above for comparison
        self.config = self.load_config()
        # Ensure DEFAULT_CONFIG path matches loaded path
        DEFAULT_CONFIG["config_save_path"] = config_path
        self.ui_elements = {}

    def load_config(self):
        # Load existing config or use defaults, ensuring all keys from the comprehensive DEFAULT_CONFIG are present
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    loaded_config = json.load(f)
                    # Ensure all default keys exist, add if missing
                    # Use the comprehensive DEFAULT_CONFIG for the check
                    for key, default_value in DEFAULT_CONFIG.items():
                        if key not in loaded_config:
                            loaded_config[key] = default_value
                        # Special handling for nested dicts like aim_mode_settings
                        elif key == "aim_mode_settings":
                             # Ensure it's a dict
                             if not isinstance(loaded_config[key], dict):
                                 loaded_config[key] = default_value # Reset if not a dict
                             else:
                                 # Ensure sub-keys exist within the dict
                                 for mode_key, mode_default in default_value.items():
                                     if mode_key not in loaded_config[key]:
                                         loaded_config[key][mode_key] = mode_default
                                     # Ensure steps exist and are lists (basic check)
                                     elif "steps" not in loaded_config[key][mode_key] or not isinstance(loaded_config[key][mode_key].get("steps"), list):
                                         loaded_config[key][mode_key]["steps"] = mode_default.get("steps", [])

                    # Remove keys present in loaded_config but not in DEFAULT_CONFIG (optional cleanup)
                    keys_to_remove = [k for k in loaded_config if k not in DEFAULT_CONFIG]
                    for k in keys_to_remove:
                        if k in loaded_config: # Check existence before deleting
                            del loaded_config[k]

                    return loaded_config
            except Exception as e:
                print(f"Error loading config, using defaults: {e}")
        # Return a copy of the comprehensive DEFAULT_CONFIG if file doesn't exist or loading fails
        return DEFAULT_CONFIG.copy()

    def save_config(self):
        try:
            with config_lock: # Ensure thread safety when saving
                with open(self.config_path, "w") as f:
                    json.dump(self.config, f, indent=4)
            print("Config saved successfully.")
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def build_ui(self):
        ui.label("ValoAI Configuration").classes("text-2xl bold")
        ui.label("Changes are applied in real-time. Settings are auto-saved.").classes("text-lg")

        with ui.column().classes("w-full"):

            # --- General Toggles & Mode ---
            with ui.card().classes("w-full mb-4"):
                ui.label("Master Controls").classes("text-xl font-bold")
                with ui.grid(columns=3).classes("w-full gap-4"):
                    self._create_checkbox("aimbot_enabled", "Aimbot Enabled")
                    self._create_checkbox("triggerbot_enabled", "Triggerbot Enabled")
                    self._create_checkbox("anti_recoil_enabled", "Anti-Recoil Enabled")
                self._create_select("aimbot_mode", "Aimbot Mode", ["normal", "rage", "flick", "silent"])

            # --- Detection Settings ---
            with ui.card().classes("w-full mb-4"):
                ui.label("Detection Settings").classes("text-xl font-bold")
                with ui.grid(columns=3).classes("w-full gap-4"):
                    self._create_input("confidence", "Confidence (0-1)", is_number=True, number_format="%.2f")
                    self._create_input("fov", "Capture FOV (px)", is_number=True, number_format="%.0f")
                    self._create_input("activation_range", "Activation Range (px)", is_number=True, number_format="%.1f")
                self._create_select("target_priority", "Target Priority", ["center", "distance", "size"])
                with ui.row().classes("items-center"):
                    self._create_checkbox("enemy_color_enabled", "Enable Color Check")
                    # TODO: Add color pickers for lower/upper color if needed, complex in NiceGUI
                    ui.label("(Color check uses Red by default)").classes("text-xs italic ml-2")

            # --- Sensitivity & Aim Tuning ---
            with ui.card().classes("w-full mb-4"):
                ui.label("Sensitivity & Aim Tuning").classes("text-xl font-bold")
                with ui.grid(columns=2).classes("w-full gap-4"):
                    self._create_input("valorant_sensitivity", "Valorant Sensitivity", is_number=True, number_format="%.3f")
                    self._create_input("aim_height_adjustment", "Height Adjust (px)", is_number=True, number_format="%.0f")
                    self._create_input("aim_deadzone_px", "Aim Deadzone (px)", is_number=True, number_format="%.1f")
                    self._create_input("aim_speed_min", "Min Aim Speed", is_number=True, number_format="%.1f")
                    self._create_input("aim_speed_max", "Max Aim Speed", is_number=True, number_format="%.1f")
                    self._create_input("follow_mode_threshold_px", "Follow Threshold (px)", is_number=True, number_format="%.1f")
                    self._create_input("tracking_speed_multiplier", "Follow Speed Multiplier", is_number=True, number_format="%.2f")

            # --- Prediction Settings ---
            with ui.card().classes("w-full mb-4"):
                ui.label("Prediction").classes("text-xl font-bold")
                with ui.grid(columns=2).classes("w-full gap-4"):
                    self._create_input("prediction_factor", "Prediction Factor (0-1)", is_number=True, number_format="%.2f")
                    self._create_input("estimated_latency_ms", "Est. Latency (ms)", is_number=True, number_format="%.0f")

            # --- Adaptive Smoothing Settings ---
            with ui.card().classes("w-full mb-4"):
                ui.label("Adaptive Smoothing (Normal/Follow Mode)").classes("text-xl font-bold")
                with ui.grid(columns=4).classes("w-full gap-4 items-center"):
                    ui.label("Alpha (Lower=Smoother)")
                    self._create_input("smoothing_alpha_precision", "Precision", is_number=True, number_format="%.2f")
                    self._create_input("smoothing_alpha_near", "Near", is_number=True, number_format="%.2f")
                    self._create_input("smoothing_alpha_mid", "Mid", is_number=True, number_format="%.2f")
                    self._create_input("smoothing_alpha_far", "Far", is_number=True, number_format="%.2f")
                    ui.label("Transition Dist (px)")
                    self._create_input("smoothing_transition_near", "Near->Prec", is_number=True, number_format="%.1f")
                    self._create_input("smoothing_transition_mid", "Mid->Near", is_number=True, number_format="%.1f")
                    self._create_input("smoothing_transition_far", "Far->Mid", is_number=True, number_format="%.1f")

            # --- Aim Steps (JSON Text Areas) ---
            with ui.card().classes("w-full mb-4"):
                ui.label("Aim Step Multipliers (Normal/Follow Mode)").classes("text-xl font-bold")
                ui.label("Format: [[dist_threshold1, speed_multiplier1], [dist_threshold2, speed_multiplier2], ...]").classes("text-xs italic")
                with ui.grid(columns=1).classes("w-full gap-4"):
                    self._create_textarea("aim_mode_settings.normal.steps", "Normal Mode Steps")
                    self._create_textarea("aim_mode_settings.follow.steps", "Follow Mode Steps")
                    # Optionally show rage steps if needed
                    # self._create_textarea("aim_mode_settings.rage.steps", "Rage Mode Steps")

            # --- Flick/Silent/Trigger/Recoil Settings ---
            with ui.card().classes("w-full mb-4"):
                ui.label("Other Modes & Helpers").classes("text-xl font-bold")
                with ui.grid(columns=2).classes("w-full gap-4"):
                    self._create_input("flick_smoothing", "Flick Smoothing Steps", is_number=True, number_format="%.0f")
                    self._create_input("flick_cooldown_ms", "Flick Cooldown (ms)", is_number=True, number_format="%.0f")
                    self._create_input("silent_cooldown_ms", "Silent Cooldown (ms)", is_number=True, number_format="%.0f")
                    self._create_input("trigger_delay_ms", "Trigger Delay (ms)", is_number=True, number_format="%.0f")
                    self._create_input("anti_recoil_strength", "Anti-Recoil Strength", is_number=True, number_format="%.2f")
                    self._create_input("anti_recoil_delay_ms", "Anti-Recoil Delay (ms)", is_number=True, number_format="%.0f")

        # --- Config Management & Status ---
        with ui.row().classes("mt-4"):
            ui.button("Reset to Defaults", on_click=self.on_reset).classes("bg-red-500")
            # Add Load/Save specific file later if needed

        ui.label("Status").classes("text-xl mt-4")
        self.status_label = ui.label("Idle").classes("text-lg")

        ui.label("Performance").classes("text-xl mt-4")
        self.fps_label = ui.label("Waiting for data...").classes("text-lg")
        ui.timer(1.0, lambda: self.update_fps_display())

    # --- New UI Element Creators ---
    def _create_checkbox(self, key, label):
        """Create a checkbox."""
        current_val = self._get_nested_config(key)
        element = ui.checkbox(label, value=current_val)
        element.on('change', lambda e, k=key: self.update_value(k, e.sender.value))
        self.ui_elements[key] = element
        return element

    def _create_select(self, key, label, options):
        """Create a dropdown selection."""
        current_val = self._get_nested_config(key)
        element = ui.select(options, label=label, value=current_val)
        element.on('change', lambda e, k=key: self.update_value(k, e.sender.value))
        self.ui_elements[key] = element
        return element

    # --- Modified/Existing UI Element Creators ---
    def _create_input(self, key, label, is_number=False, number_format="%.3f"):
        """Create a standard input/number field."""
        default_val = self._get_nested_default(key)
        current_val = self._get_nested_config(key)

        if is_number:
            # Use provided format string
            element = ui.number(label=label, value=current_val, format=number_format)
            element.on('change', lambda e, k=key: self.update_value(k, e.sender.value))
        else:
            element = ui.input(label=label, value=str(current_val))
            element.on('change', lambda e, k=key: self.update_value(k, e.sender.value))

        self.ui_elements[key] = element
        return element

    def _create_textarea(self, key, label):
        """Create a text area for JSON input."""
        default_val_list = self._get_nested_default(key)
        current_val_list = self._get_nested_config(key)

        # Convert list to JSON string for the text area
        current_json_str = json.dumps(current_val_list, indent=2)

        element = ui.textarea(label=label, value=current_json_str).props('rows=5')
        element.on('change', lambda e, k=key: self.update_value(k, e.sender.value, is_json=True))

        self.ui_elements[key] = element
        return element

    def _get_nested_config(self, key_path):
        """Helper to get value from nested config dict using dot notation."""
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else: # Handle case where path doesn't fully exist
                    return self._get_nested_default(key_path) # Return default if path invalid
        except KeyError:
             return self._get_nested_default(key_path) # Return default if key missing
        return value

    def _get_nested_default(self, key_path):
        """Helper to get value from nested default config dict using dot notation."""
        keys = key_path.split('.')
        value = DEFAULT_CONFIG
        try:
            for key in keys:
                 value = value[key]
        except KeyError:
            print(f"Warning: Default key not found: {key_path}")
            return None # Or raise error, or return specific default
        return value

    def _set_nested_config(self, key_path, value):
         """Helper to set value in nested config dict using dot notation."""
         keys = key_path.split('.')
         d = self.config
         for key in keys[:-1]:
             d = d.setdefault(key, {})
         d[keys[-1]] = value

    def update_value(self, key, value, is_json=False):
        """Update configuration values in real-time with appropriate type conversion and JSON handling"""
        global CONFIG, fov, mid, model

        # Check if the key is still valid (exists in DEFAULT_CONFIG) before proceeding
        # Need a more robust check for nested keys
        def check_key_exists(config_dict, key_path):
            keys = key_path.split('.')
            temp_dict = config_dict
            for k in keys:
                if isinstance(temp_dict, dict) and k in temp_dict:
                    temp_dict = temp_dict[k]
                else:
                    return False
            return True

        if not check_key_exists(DEFAULT_CONFIG, key):
             print(f"Attempted to update removed or invalid key: {key}")
             # Try to update UI element back to original value if possible
             if key in self.ui_elements:
                 try:
                     last_valid_value = self._get_nested_config(key)
                     if isinstance(self.ui_elements[key], ui.textarea):
                         self.ui_elements[key].value = json.dumps(last_valid_value, indent=2)
                     elif isinstance(self.ui_elements[key], ui.checkbox):
                         self.ui_elements[key].value = bool(last_valid_value)
                     else:
                         self.ui_elements[key].value = last_valid_value
                 except Exception as revert_err:
                     print(f"Could not revert UI for invalid key {key}: {revert_err}")
             return False

        try:
            default_val = self._get_nested_default(key)
            parsed_value = value # Start with the raw value

            if is_json:
                try:
                    parsed_value = json.loads(value)
                    # Basic validation: Check if it's a list of lists/tuples
                    if not isinstance(parsed_value, list) or not all(isinstance(item, (list, tuple)) for item in parsed_value):
                        raise ValueError("Invalid JSON structure for steps.")
                except json.JSONDecodeError as e:
                    self.status_label.set_text(f"Invalid JSON for {key}: {e}")
                    # Revert UI element to last valid config value
                    if key in self.ui_elements:
                         last_valid_value = self._get_nested_config(key)
                         self.ui_elements[key].value = json.dumps(last_valid_value, indent=2)
                    return False
                except ValueError as e:
                     self.status_label.set_text(f"Invalid data for {key}: {e}")
                     # Revert UI element
                     if key in self.ui_elements:
                          last_valid_value = self._get_nested_config(key)
                          self.ui_elements[key].value = json.dumps(last_valid_value, indent=2)
                     return False
            elif isinstance(default_val, bool):
                 parsed_value = bool(value)
            elif isinstance(default_val, int):
                try:
                    parsed_value = int(float(value))
                except (ValueError, TypeError):
                    parsed_value = default_val
                    if key in self.ui_elements: self.ui_elements[key].value = parsed_value
            elif isinstance(default_val, float):
                try:
                    parsed_value = float(value)
                except (ValueError, TypeError):
                    parsed_value = default_val
                    if key in self.ui_elements: self.ui_elements[key].value = parsed_value
            # else: string type, parsed_value remains as value

            # Update the nested config dictionaries
            with config_lock:
                self._set_nested_config(key, parsed_value) # Use helper to set nested value
                CONFIG = self.config.copy() # Update global CONFIG

                # Update aimbot's config directly
                aimbot.CONFIG = self.config.copy()

                # Special handling for FOV
                if key == "fov":
                    fov = int(parsed_value) # Ensure fov is int
                    mid = fov / 2
                    # Need to recalculate capture region if using bettercam
                    # update_capture_region(fov)

                # Update model confidence
                if key == "confidence" and 'model' in globals() and model is not None:
                    model.conf = parsed_value

            self.status_label.set_text(f"Updated: {key} = {value}")
            self.save_config() # Auto-save on successful update
            return True

        except Exception as e:
            self.status_label.set_text(f"Error updating {key}: {str(e)}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            # Attempt to revert UI element on error
            if key in self.ui_elements:
                 try:
                     last_valid_value = self._get_nested_config(key) # Get last known good value
                     # Revert based on type
                     if isinstance(self.ui_elements[key], ui.textarea):
                         self.ui_elements[key].value = json.dumps(last_valid_value, indent=2)
                     elif isinstance(self.ui_elements[key], ui.checkbox):
                         self.ui_elements[key].value = bool(last_valid_value)
                     else:
                         self.ui_elements[key].value = last_valid_value
                 except Exception as revert_err:
                     print(f"Could not revert UI for key {key} after error: {revert_err}")
            return False

    def update_fps_display(self):
        global detection_fps, loop_fps
        if 'detection_fps' in globals() and 'loop_fps' in globals():
            self.fps_label.set_text(f"AI Detections/s: {detection_fps:.1f} | Loop FPS: {loop_fps:.1f}")


    def on_reset(self):
        """Reset configuration to defaults"""
        global CONFIG, fov, mid

        with config_lock:
            self.config = json.loads(json.dumps(DEFAULT_CONFIG)) # Deep copy via JSON
            CONFIG = self.config.copy()
            aimbot.CONFIG = self.config.copy() # Reset aimbot's config too

            fov = CONFIG["fov"]
            mid = fov / 2
            # update_capture_region(fov) # If using bettercam

        # Update UI elements, handling nested keys and types
        for key, element in self.ui_elements.items():
            try:
                default_val = self._get_nested_default(key)
                if default_val is not None:
                    if isinstance(element, ui.textarea):
                         element.value = json.dumps(default_val, indent=2)
                    elif isinstance(element, ui.checkbox):
                         element.value = bool(default_val)
                    elif isinstance(element, ui.select):
                         element.value = default_val # Assumes default is valid option
                    else: # Handles input, number
                         element.value = default_val
            except Exception as e:
                print(f"Error resetting UI element {key}: {e}")

        self.save_config() # Auto-save after reset
        ui.notify("Configuration reset to defaults", color="blue")

def start_program():
    global detection_times, detection_fps, last_fps_update, fps_update_interval
    global loop_times, loop_fps, last_loop_fps_update
    global detection_queue, detection_event, running, detection_lock, latest_detection
    global CONFIG, fov, mid, model, running

    running = True

    print(colored('''
       ___     _ _    __  __
      / __|___| | |___\ \/ /__ _ _ __
     | (__/ -_) | / _ \>  </ _` | '  |
      \___\___|_|_\___/_/\_\__,_|_|_|_|

    ''', "magenta", attrs=['bold']))

    # Load model using config settings
    with config_lock:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='v/scripts/best.pt')
        model.conf = CONFIG["confidence"]
        model.max_det = 5
        model.nms = 0.75 # Consider making this configurable too?

    if torch.cuda.is_available():
        print(colored("CUDA ACCELERATION [ENABLED]", "green"))
        model.to('cuda')
    else:
        print(colored("CUDA ACCELERATION [DISABLED] - Using CPU", "yellow"))


    kmNet.init('192.168.2.188','12353', 'F065E04E') # Consider making IP/Port configurable?
    kmNet.monitor(10000)

    print('Running!')
    # Updated print statement for new config
    with config_lock:
        print("--- Key Settings ---")
        print(f"Exit Key: {CONFIG['exit_key']}")
        print(f"Aim Key: Right Mouse (Hardcoded)")
        print(f"Trigger Key: Side Mouse 2 (Hardcoded)")
        print("--- Core Settings ---")
        print(f"Aimbot Enabled: {CONFIG['aimbot_enabled']}")
        print(f"Aimbot Mode: {CONFIG['aimbot_mode']}")
        print(f"Triggerbot Enabled: {CONFIG['triggerbot_enabled']}")
        print(f"Anti-Recoil Enabled: {CONFIG['anti_recoil_enabled']}")
        print("--- Detection ---")
        print(f"FOV: {CONFIG['fov']}x{CONFIG['fov']}")
        print(f"Confidence: {CONFIG['confidence']}")
        print(f"Activation Range: {CONFIG['activation_range']}px")
        print(f"Target Priority: {CONFIG['target_priority']}")
        print(f"Color Check: {CONFIG['enemy_color_enabled']}")
        print("--- Tuning ---")
        print(f"Sensitivity: {CONFIG['valorant_sensitivity']}")
        print(f"Prediction Latency: {CONFIG['estimated_latency_ms']}ms")
        # Add more prints if desired

    play_beep()
    print("Starting threads...")

    detection_thread_obj = threading.Thread(target=detection_thread)
    main_thread_obj = threading.Thread(target=main_thread)

    detection_thread_obj.daemon = True
    main_thread_obj.daemon = True

    detection_thread_obj.start()
    main_thread_obj.start()

    try:
        while running:
            with config_lock: # Read exit key safely
                 exit_key_local = CONFIG["exit_key"]
            if keyboard.is_pressed(exit_key_local):
                running = False
                break
            time.sleep(0.1) # Main loop sleep
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        running = False
        print("\nShutting down threads...")
        # Signal threads to stop if they check 'running'
        detection_event.set() # Wake up detection thread if waiting
        # Join threads with timeout
        detection_thread_obj.join(timeout=1.0)
        main_thread_obj.join(timeout=1.0)
        print("Exited cleanly")

def windowcapture():
    hwnd = None
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, fov, fov)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (260, 200), dcObj, (800, 380), win32con.SRCCOPY)
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8').reshape((fov, fov, 4))
    
    img = img[:, :, :3]
    
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    return img

def get_target_priority(detections, df):
    if CONFIG["target_priority"] == "distance":
        return detections
    
    elif CONFIG["target_priority"] == "center":
        screen_center_x, screen_center_y = fov/2, fov/2
        for i, (_, detection) in enumerate(detections):
            xmin, ymin, xmax, ymax = detection[:4]
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            
            distance = math.sqrt((center_x - screen_center_x)**2 + (center_y - screen_center_y)**2)
            detections[i] = (distance, detection)
        
        return sorted(detections, key=lambda x: x[0])
    
    elif CONFIG["target_priority"] == "size":
        for i, (_, detection) in enumerate(detections):
            xmin, ymin, xmax, ymax = detection[:4]
            size = (xmax - xmin) * (ymax - ymin)
            detections[i] = (size, detection)
        
        return sorted(detections, key=lambda x: x[0], reverse=True)
    
    return detections

def calculate_aim_point(xmin, ymin, xmax, ymax):
    center_x = (xmin + xmax) / 2
    
    height_factor = 0.5 + (CONFIG["aim_height_adjustment"] / 20)
    center_y = ymin + (ymax - ymin) * height_factor
    
    return center_x, center_y

def update_detection_fps():
    global detection_fps, last_fps_update
    
    current_time = time.time()
    detection_times.append(current_time)
    
    if current_time - last_fps_update >= fps_update_interval:
        one_second_ago = current_time - 1.0
        detection_fps = sum(1 for t in detection_times if t > one_second_ago)
        last_fps_update = current_time
    
    return detection_fps

def update_loop_fps():
    global loop_fps, last_loop_fps_update
    
    current_time = time.time()
    loop_times.append(current_time)
    
    if current_time - last_loop_fps_update >= fps_update_interval:
        one_second_ago = current_time - 1.0
        loop_fps = sum(1 for t in loop_times if t > one_second_ago)
        detection_fps = sum(1 for t in detection_times if t > one_second_ago)
        last_loop_fps_update = current_time
        
        print(f"\rAI Detections/s: {detection_fps} | Loop FPS: {loop_fps}        ", end="")
    
    return loop_fps

def detection_thread():
    global running, detection_fps, latest_detection, model, detection_times, detection_event, detection_lock

    while running:
        try:
            # Check if model confidence needs update (thread-safe read)
            with config_lock:
                current_conf = CONFIG["confidence"]
            if model and hasattr(model, 'conf') and model.conf != current_conf:
                 model.conf = current_conf # Update if necessary

            sct_img = windowcapture() # Capture screen

            results = model(sct_img) # Perform detection

            detections = [] # Process results
            if len(results.pred) > 0 and len(results.pred[0]) > 0:
                pred = results.pred[0]
                for i in range(len(pred)):
                    try:
                        # Extract bounding box coordinates
                        xmin, ymin, xmax, ymax = pred[i, :4].tolist()
                        # Calculate distance from center (mid) for potential sorting
                        center_x = (xmin + xmax) / 2
                        center_y = (ymin + ymax) / 2
                        distance = math.sqrt((center_x - mid)**2 + (center_y - mid)**2)
                        # Store distance and bbox
                        detections.append((distance, [xmin, ymin, xmax, ymax]))
                    except Exception as det_err:
                        # print(f"Error processing detection {i}: {det_err}") # Optional debug
                        continue

            # Update shared detection data
            with detection_lock:
                if detections:
                    # Apply target prioritization logic
                    prioritized_detections = get_target_priority(detections, None) # Pass None for df
                    latest_detection = prioritized_detections
                    detection_times.append(time.time()) # Record time only if detection found
                    detection_event.set() # Signal main thread
                else:
                    latest_detection = None # Clear if no detections
                    # Do not signal if no detections? Or signal with None? Current logic implies signal only on detection.

        except Exception as e:
            print(f"\nDetection thread error: {e}")
            # Consider adding a small sleep here to prevent spamming errors in case of persistent issues
            time.sleep(0.1)


def main_thread():
    global running, loop_fps, last_click_time, loop_times, detection_fps, last_loop_fps_update

    last_click_time = time.time()
    fps_check_time = time.time() + fps_update_interval
    aim_key_was_pressed = False

    while running:
        loop_start = time.time()

        # --- Read Config Safely ---
        with config_lock:
            exit_key = CONFIG["exit_key"]
            follow_threshold = float(CONFIG.get("follow_mode_threshold_px", 30.0))
            aimbot_enabled = CONFIG.get("aimbot_enabled", True)
            triggerbot_enabled = CONFIG.get("triggerbot_enabled", True)
            aimbot_mode = CONFIG.get("aimbot_mode", "normal")
            activation_range = float(CONFIG.get("activation_range", 100.0))
            enemy_color_enabled = CONFIG.get("enemy_color_enabled", False)
            lower_color = CONFIG.get("lower_color", [230, 40, 40])
            upper_color = CONFIG.get("upper_color", [255, 90, 104])

        # --- Check Exit Key ---
        if keyboard.is_pressed(exit_key):
            print("\nExiting program...")
            running = False
            break

        # --- Check Aim Key State (Hardcoded) ---
        aim_key_pressed = kmNet.isdown_right() == 1

        # --- Reset Aim State on Key Press/Release ---
        if aim_key_pressed and not aim_key_was_pressed:
            aimbot.reset_aim_state() # Reset when key is first pressed
        elif not aim_key_pressed and aim_key_was_pressed:
            aimbot.reset_aim_state() # Reset when key is released
        aim_key_was_pressed = aim_key_pressed # Update state for next loop

        # --- Check Trigger Key State ---
        trigger_key_pressed = kmNet.isdown_side2() == 1

        # --- Wait for Detection Signal ---
        detection_available = detection_event.wait(timeout=0.001) # Small timeout to prevent blocking

        if detection_available:
            detection_event.clear() # Clear the event immediately

            with detection_lock:
                prioritized_detections = latest_detection # Copy the latest detection list

            if prioritized_detections:
                # Find best target with closest distance or by color
                best_target = None
                closest_distance = float('inf')
                
                for dist_val, detection in prioritized_detections:
                    xmin, ymin, xmax, ymax = detection

                    # Calculate center of target
                    target_center_x, target_center_y = aimbot.calculate_aim_point(xmin, ymin, xmax, ymax)
                    x_offset, y_offset = aimbot.get_aim_offset(target_center_x, target_center_y, mid)
                    distance = math.sqrt(x_offset**2 + y_offset**2)

                    # Check if target is within activation range
                    if distance > activation_range:
                        continue
                        
                    # If enemy color check is enabled, verify target has the right color
                    if enemy_color_enabled:
                        # Get the portion of the image for color checking
                        try:
                            # Capture a new screenshot or use the current one from detection thread
                            sct_img = windowcapture()
                            
                            # Extract the target area (add padding for color check)
                            xmin_i, ymin_i = max(0, int(xmin - 10)), max(0, int(ymin - 10))
                            xmax_i, ymax_i = min(fov, int(xmax + 10)), min(fov, int(ymax + 10))
                            
                            target_img = sct_img[ymin_i:ymax_i, xmin_i:xmax_i]
                            
                            # Convert to RGB for color check
                            target_rgb = target_img  # Already RGB for our capture
                            
                            # Create mask for color range
                            import cv2
                            mask = cv2.inRange(target_rgb, np.array(lower_color), np.array(upper_color))
                            
                            # Check percentage of pixels matching color
                            color_percentage = np.count_nonzero(mask) / (target_img.shape[0] * target_img.shape[1])
                            
                            if color_percentage <= 0:  # No color match
                                continue
                                
                        except Exception as e:
                            # If color check fails, just continue with this target
                            print(f"Color check error: {e}")
                    
                    # Simple closest target selection
                    if distance < closest_distance:
                        closest_distance = distance
                        best_target = detection

                # If we found a valid target
                if best_target is not None:
                    xmin, ymin, xmax, ymax = best_target

                    # Calculate aim point and offset
                    center_x, center_y = aimbot.calculate_aim_point(xmin, ymin, xmax, ymax)
                    x, y = aimbot.get_aim_offset(center_x, center_y, mid)
                    
                    # Check distance for follow-mode and current aim mode
                    current_distance = math.sqrt(x**2 + y**2)
                    
                    # Determine aim mode for normal/follow dynamic switching
                    current_aim_mode = aimbot_mode  # Use configured mode by default
                    
                    # Only override for normal mode with follow threshold
                    if aimbot_mode == 'normal' and current_distance <= follow_threshold:
                        current_aim_mode = 'follow'

                    # Check if crosshair is within the bounding box (for triggerbot)
                    crosshair_padding = 5  # Small tolerance
                    cross_in_box = (xmin - crosshair_padding <= mid <= xmax + crosshair_padding) and \
                                  (ymin - crosshair_padding <= mid <= ymax + crosshair_padding)

                    # --- Aiming Logic ---
                    if aim_key_pressed and aimbot_enabled:
                        aimbot.aim_at_target(x, y, mode=current_aim_mode)

                    # --- Triggerbot Logic ---
                    if trigger_key_pressed and triggerbot_enabled and cross_in_box:
                        if aimbot.can_click():
                            kmNet.left(1)  # Press left mouse button
                            time.sleep(0.01)  # Short delay
                            kmNet.left(0)  # Release left mouse button

        # --- Update FPS Counters ---
        loop_times.append(time.time())  # Record loop end time
        current_time = time.time()
        if current_time >= fps_check_time:
            with detection_lock:  # Access detection_times safely
                one_second_ago = current_time - 1.0
                current_loop_fps = sum(1 for t in loop_times if t > one_second_ago)
                current_detection_fps = sum(1 for t in detection_times if t > one_second_ago)

            loop_fps = current_loop_fps
            detection_fps = current_detection_fps
            fps_check_time = current_time + fps_update_interval  # Schedule next check

        # Optional small sleep to prevent 100% CPU usage if loop is very fast
        # time.sleep(0.001)


def play_beep():
    try:
        winsound.Beep(1000, 200)
    except Exception as e:
        print("Beep error:", e)

if __name__ == '__main__':
    config_manager = ConfigManager() # Creates/Loads config using the comprehensive DEFAULT_CONFIG

    # Update global CONFIG and aimbot.CONFIG from loaded config
    with config_lock:
        CONFIG = config_manager.config.copy()
        # Update aimbot.CONFIG with the potentially loaded/merged config
        # aimbot.py will only use the keys it knows about
        aimbot.CONFIG = config_manager.config.copy()

    # Set global fov/mid based on loaded config
    fov = CONFIG["fov"] # This should now be safe
    mid = fov / 2

    # Build the UI based on the loaded config (will not include aim/trigger keys)
    config_manager.build_ui()

    # Start the main program logic in a separate thread
    main_program_thread = threading.Thread(target=start_program)
    main_program_thread.daemon = True # Allow program to exit even if this thread is running
    main_program_thread.start()

    # Start the NiceGUI server
    # show=False prevents NiceGUI from opening a new tab each time if reload=True (which it isn't here)
    # Set reload=False for production/stable use
    ui.run(title="ValoAI Configuration", port=8080, reload=False, show=False)





