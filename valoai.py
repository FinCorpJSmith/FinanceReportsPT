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
import aimbot
from collections import deque
import queue
from threading import Event, Lock
from nicegui import ui
import webbrowser

# Define a single unified configuration
DEFAULT_CONFIG = {
    "aim_key": "right",
    "trigger_key": "side2",
    "exit_key": "end",
    "confidence": 0.4,
    "smoothing": 1.5,
    "fov": 320,
    "aim_height_adjustment": 0,
    "trigger_delay": 0.35,
    "target_priority": "center",
    "bezier_steps": 10,
    "accel_factor": 1.1,
    "bezier_start_delay": 0.01,
    "bezier_end_delay": 0.0008,
    "flick_cooldown": 0.5,
    "valorant_sensitivity": 0.4,
    "aim_mode": "legit"
}

# Global variables
CONFIG = DEFAULT_CONFIG.copy()
fov = CONFIG["fov"]
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
        self.config = self.load_config()
        self.ui_elements = {}
        
    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    loaded_config = json.load(f)
                    for key, default_value in DEFAULT_CONFIG.items():
                        if key not in loaded_config:
                            loaded_config[key] = default_value
                    return loaded_config
            except Exception as e:
                print(f"Error loading config, using defaults: {e}")
        return DEFAULT_CONFIG.copy()

    def save_config(self):
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)
            print("Config saved successfully.")
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def build_ui(self):
        ui.label("ValoAI Configuration").classes("text-2xl bold")
        ui.label("Changes are applied in real-time. Click 'Save Config' to persist settings.").classes("text-lg")
        
        with ui.column().classes("w-full"):
            with ui.card().classes("w-full mb-4"):
                ui.label("Aiming Settings").classes("text-xl font-bold")
                with ui.grid(columns=2).classes("w-full gap-4"):
                    self._create_input("aim_key", "Aim Key")
                    self._create_input("aim_mode", "Aim Mode: bezier/legit/flick/rage/flick_legit/silent")
                    self._create_input("smoothing", "Smoothing", is_number=True)
                    self._create_input("aim_height_adjustment", "Height Adjustment", is_number=True)
                    self._create_input("valorant_sensitivity", "Valorant Sensitivity", is_number=True)
                
            with ui.card().classes("w-full mb-4"):
                ui.label("Detection Settings").classes("text-xl font-bold")
                with ui.grid(columns=2).classes("w-full gap-4"):
                    self._create_input("confidence", "Detection Confidence", is_number=True)
                    self._create_input("fov", "Field of View", is_number=True)
                    self._create_input("target_priority", "Target Priority")
            
            with ui.card().classes("w-full mb-4"):
                ui.label("Trigger Settings").classes("text-xl font-bold")
                with ui.grid(columns=2).classes("w-full gap-4"):
                    self._create_input("trigger_key", "Trigger Key")
                    self._create_input("trigger_delay", "Trigger Delay", is_number=True)
                    
            with ui.card().classes("w-full mb-4"):
                ui.label("Advanced Settings").classes("text-xl font-bold")
                with ui.grid(columns=2).classes("w-full gap-4"):
                    self._create_input("bezier_steps", "Bezier Steps", is_number=True)
                    self._create_input("accel_factor", "Accel Factor", is_number=True)
                    self._create_input("bezier_start_delay", "Start Delay", is_number=True)
                    self._create_input("bezier_end_delay", "End Delay", is_number=True)
                    self._create_input("flick_cooldown", "Flick Cooldown", is_number=True)
                    self._create_input("exit_key", "Exit Key")
        
        with ui.row().classes("mt-4"):
            ui.button("Save Config", on_click=self.on_save).classes("bg-green-500")
            ui.button("Reset to Defaults", on_click=self.on_reset).classes("bg-red-500")
            
        ui.label("Status").classes("text-xl mt-4")
        self.status_label = ui.label("Idle").classes("text-lg")
        
        ui.label("Performance").classes("text-xl mt-4")
        self.fps_label = ui.label("Waiting for data...").classes("text-lg")
        ui.timer(1.0, lambda: self.update_fps_display())

    def _create_input(self, key, label, is_number=False):
        """Create an input field with real-time update capability"""
        default_val = DEFAULT_CONFIG[key]
        current_val = self.config[key]
        
        # Create the appropriate input type
        if is_number:
            # Force display format based on the default value type
            if isinstance(default_val, int):
                format_str = "%.0f"  # No decimal places for integers
            else:
                format_str = "%.2f"  # Two decimal places for floats
                
            element = ui.number(
                label=label,
                value=current_val,
                format=format_str
            )
            
            # Define the update function correctly accessing the value
            def on_number_change(e):
                # The value is in the actual element, not in the event
                new_value = element.value
                self.update_value(key, new_value)
                
            element.on('change', on_number_change)
        else:
            element = ui.input(label=label, value=str(current_val))
            
            # Define the update function correctly accessing the value
            def on_text_change(e):
                # The value is in the actual element, not in the event
                new_value = element.value
                self.update_value(key, new_value)
                
            element.on('change', on_text_change)
            
        # Store reference to the element
        self.ui_elements[key] = element
        
        return element
        
    def update_value(self, key, value):
        """Update configuration values in real-time with appropriate type conversion"""
        global CONFIG, fov, mid, model
        
        try:
            default_val = DEFAULT_CONFIG[key]
            
            # Convert value to the appropriate type
            if isinstance(default_val, int):
                try:
                    # Force integer conversion
                    value = int(float(value))
                except (ValueError, TypeError):
                    value = default_val
                    # Update UI element to show correct value
                    if key in self.ui_elements:
                        self.ui_elements[key].value = value
            elif isinstance(default_val, float):
                try:
                    # Force float conversion
                    value = float(value)
                except (ValueError, TypeError):
                    value = default_val
                    # Update UI element to show correct value
                    if key in self.ui_elements:
                        self.ui_elements[key].value = value
            
            # Ensure the value is set even if it's a string
            with config_lock:
                self.config[key] = value
                CONFIG[key] = value
                
                # Special handling for FOV which needs to update mid
                if key == "fov":
                    fov = value
                    mid = fov / 2
                
                # Update model confidence if model is initialized
                if key == "confidence" and 'model' in globals() and model is not None:
                    model.conf = value
                
                # Update aimbot's config too
                aimbot.CONFIG[key] = value
            
            # Update the status
            self.status_label.set_text(f"Updated: {key} = {value}")
            
            # Auto-save the configuration to persist settings
            self.save_config()
            
            return True
        
        except Exception as e:
            self.status_label.set_text(f"Error updating {key}: {str(e)}")
            return False

    def update_fps_display(self):
        global detection_fps, loop_fps
        
        if 'detection_fps' in globals() and 'loop_fps' in globals():
            self.fps_label.set_text(f"AI Detections/s: {detection_fps} | Loop FPS: {loop_fps}")

    def on_save(self):
        """Save the current configuration to disk"""
        if self.save_config():
            ui.notify("Configuration saved successfully!", color="green")
        else:
            ui.notify("Failed to save configuration!", color="red")

    def on_reset(self):
        """Reset configuration to defaults"""
        global CONFIG, fov, mid
        
        with config_lock:
            self.config = DEFAULT_CONFIG.copy()
            CONFIG = DEFAULT_CONFIG.copy()
            fov = CONFIG["fov"]
            mid = fov / 2
            
            # Update aimbot's config too
            for key, value in DEFAULT_CONFIG.items():
                aimbot.CONFIG[key] = value
        
        # Update UI elements
        for key, element in self.ui_elements.items():
            try:
                default_val = DEFAULT_CONFIG[key]
                # Update the UI element with the default value
                element.value = default_val
            except Exception as e:
                print(f"Error resetting UI element {key}: {e}")
        
        # Auto-save after reset
        self.save_config()
        ui.notify("Configuration reset to defaults", color="blue")

def start_program():
    global detection_times, detection_fps, last_fps_update, fps_update_interval
    global loop_times, loop_fps, last_loop_fps_update
    global detection_queue, detection_event, running, detection_lock, latest_detection
    global CONFIG, fov, mid, model
    
    running = True

    print(colored('''
       ___     _ _    __  __           
      / __|___| | |___\ \/ /__ _ _ __  
     | (__/ -_) | / _ \>  </ _` | '  \ 
      \___\___|_|_\___/_/\_\__,_|_|_|_|
                                       
    ''', "magenta", attrs=['bold']))

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='v/scripts/best.pt')
    model.conf = CONFIG["confidence"]
    model.max_det = 5
    model.nms = 0.75

    if torch.cuda.is_available():
        print(colored("CUDA ACCELERATION [ENABLED]", "green"))
        model.to('cuda')

    kmNet.init('192.168.2.188','12353', 'F065E04E')
    kmNet.monitor(10000)

    print('Running!')
    print(f"Settings: FOV={CONFIG['fov']}, Smoothing={CONFIG['smoothing']}, Confidence={CONFIG['confidence']}")
    print(f"Aim key: {CONFIG['aim_key']}, Trigger key: {CONFIG['trigger_key']}, Exit key: {CONFIG['exit_key']}")
    play_beep()
    print("Starting multithreaded FPS counters...")

    detection_thread_obj = threading.Thread(target=detection_thread)
    main_thread_obj = threading.Thread(target=main_thread)

    detection_thread_obj.daemon = True
    main_thread_obj.daemon = True

    detection_thread_obj.start()
    main_thread_obj.start()

    try:
        while running:
            time.sleep(0.1)
            if keyboard.is_pressed(CONFIG["exit_key"]):
                running = False
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        running = False
        print("\nShutting down threads...")
        detection_thread_obj.join(timeout=1.0)
        main_thread_obj.join(timeout=1.0)
        print("Exited cleanly")

def calculatedistance(x, y):
    snap_fov = 20
    snap_fov_half = snap_fov / 2

    aimbot.aim_at_target(int(x), int(y), mode=CONFIG.get("aim_mode", "legit"), snap_fov=snap_fov)

def tb():
    def action():
        kmNet.left(1)
        kmNet.left(0)
        time.sleep(CONFIG["trigger_delay"])
    
    thread = threading.Thread(target=action)
    thread.daemon = True
    thread.start()
   
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
    global running, detection_fps, latest_detection, model
    
    while running:
        try:
            with config_lock:
                if model and hasattr(model, 'conf') and model.conf != CONFIG["confidence"]:
                    model.conf = CONFIG["confidence"]
            
            sct_img = windowcapture()
            
            results = model(sct_img)
            
            if len(results.pred) > 0 and len(results.pred[0]) > 0:
                with detection_lock:
                    detection_times.append(time.time())
                
                pred = results.pred[0]
                detections = []
                
                for i in range(len(pred)):
                    try:
                        xmin, ymin, xmax, ymax = pred[i, :4].tolist()
                        center_x = (xmin + xmax) / 2
                        center_y = (ymin + ymax) / 2
                        distance = math.sqrt((center_x - mid)**2 + (center_y - mid)**2)
                        detections.append((distance, [xmin, ymin, xmax, ymax]))
                    except Exception:
                        continue
                
                if detections:
                    prioritized_detections = get_target_priority(detections, None)
                    
                    with detection_lock:
                        latest_detection = prioritized_detections
                        detection_event.set()
            
            else:
                with detection_lock:
                    latest_detection = None
        
        except Exception as e:
            print(f"Detection error: {e}")

def main_thread():
    global running, loop_fps, last_click_time
    
    last_click_time = time.time()
    fps_check_time = time.time() + fps_update_interval
    
    while running:
        loop_start = time.time()
        loop_times.append(loop_start)
        
        with config_lock:
            click_cooldown = CONFIG["trigger_delay"]
            exit_key = CONFIG["exit_key"]
            aim_key = CONFIG["aim_key"]
            trigger_key = CONFIG["trigger_key"]
        
        if keyboard.is_pressed(exit_key):
            print("\nExiting program...")
            running = False
            break
        
        detection_available = detection_event.wait(timeout=0.001)
        
        if detection_available:
            detection_event.clear()
            
            with detection_lock:
                prioritized_detections = latest_detection
            
            if prioritized_detections:
                _, top_detection = prioritized_detections[0]
                xmin, ymin, xmax, ymax = top_detection
                
                center_x, center_y = calculate_aim_point(xmin, ymin, xmax, ymax)
                
                x = center_x - mid
                y = center_y - mid
                
                cross_in_box = (xmin-6 <= mid <= xmax+6) and (ymin-4 <= mid <= ymax+4)
                
                if kmNet.isdown_right():
                    calculatedistance(x, y)
                
                if kmNet.isdown_side2() and cross_in_box:
                    current_time = time.time()
                    if current_time - last_click_time > click_cooldown:
                        tb()
                        last_click_time = current_time
        
        current_time = time.time()
        if current_time >= fps_check_time:
            with detection_lock:
                one_second_ago = current_time - 1.0
                loop_fps = sum(1 for t in loop_times if t > one_second_ago)
                detection_fps = sum(1 for t in detection_times if t > one_second_ago)
            
            print(f"\rAI Detections/s: {detection_fps} | Loop FPS: {loop_fps}        ", end="")
            fps_check_time = current_time + fps_update_interval

def play_beep():
    try:
        winsound.Beep(1000, 200)
    except Exception as e:
        print("Beep error:", e)

if __name__ == '__main__':
    config_manager = ConfigManager()
    
    CONFIG.update(config_manager.config)
    
    for key, value in CONFIG.items():
        aimbot.CONFIG[key] = value
    
    fov = CONFIG["fov"]
    mid = fov / 2
    
    config_manager.build_ui()
    
    # Start the main program logic in a separate thread
    main_program_thread = threading.Thread(target=start_program)
    main_program_thread.daemon = True
    main_program_thread.start()
    
    # Start the NiceGUI server with show=True to use NiceGUI's native browser opening (once)
    ui.run(title="ValoAI Configuration", port=8080, reload=False, show=True)





