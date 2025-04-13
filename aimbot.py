import math
import time
import threading
import kmNet

# --- Configuration ---
CONFIG = {
    # Basic settings
    "exit_key": "end",
    "aim_height_adjustment": 0,
    "trigger_delay_ms": 10,
    "valorant_sensitivity": 0.4,  # Matches Valorant-AI-Aimbot default

    # --- Aimbot General Settings ---
    "aimbot_enabled": True,
    "aimbot_mode": "normal",  # Options: "normal", "rage", "flick", "silent"
    "activation_range": 100.0,
    
    # --- Direct Aiming Config (Valorant-AI-Aimbot style) ---
    "aimbot_smoothing": 5.0,  # Higher = smoother (like Valorant-AI-Aimbot)
    "flick_smoothing": 50,    # Steps for flick (like Valorant-AI-Aimbot)
    "aim_speed": 1.0,         # Base value - calculated based on sensitivity
    
    # Anti-Recoil Settings
    "anti_recoil_enabled": True,
    "anti_recoil_strength": 0.5,  # Base strength
    "anti_recoil_multiplier": 25,  # Multiplier (from VAI)
    
    # Cooldown Settings
    "flick_cooldown_ms": 50,
    "silent_cooldown_ms": 50,
    
    # Color Detection (from VAI)
    "enemy_color_enabled": True,
    "lower_color": [230, 40, 40],  # Default Red (BGR)
    "upper_color": [255, 90, 104]  # Default Red (BGR)
}

# --- Global State Variables ---
last_click_time = 0
can_fire_flick = [True]  # Thread-safe flag for flick/silent cooldown
anti_recoil_value = 0    # Current anti-recoil offset (increased while firing)
is_anti_recoil_active = False  # Flag to start/stop anti-recoil thread

# --- Helper Functions ---
def reset_aim_state():
    """
    Reset any stateful aiming variables.
    Called when aim key is pressed or released to ensure clean state.
    """
    global can_fire_flick
    # Reset flick cooldown
    can_fire_flick = [True]


def cooldown(delay_seconds):
    """Cooldown helper function for flick and silent aim modes."""
    time.sleep(delay_seconds)
    can_fire_flick[0] = True


def get_key_state(v_key):
    """
    Check if a key is pressed (compatibility function).
    Using kmNet functions for mouse buttons.
    """
    # Map common mouse buttons
    if v_key == 0x01:  # Left mouse
        return kmNet.isdown_left() == 1
    elif v_key == 0x02:  # Right mouse
        return kmNet.isdown_right() == 1
    elif v_key == 0x05:  # Mouse button 4 / side 1
        return kmNet.isdown_side1() == 1
    elif v_key == 0x06:  # Mouse button 5 / side 2
        return kmNet.isdown_side2() == 1
    else:
        # For non-mouse keys, this won't work
        # but we're only using it for mouse buttons
        return False


def calculate_bezier_point(t, p0, p1, p2):
    """Calculate point on a quadratic bezier curve."""
    u = (1 - t)
    tt = (t * t)
    uu = (u * u)

    x = int(uu * p0[0] + 2 * u * t * p1[0] + tt * p2[0])
    y = int(uu * p0[1] + 2 * u * t * p1[1] + tt * p2[1])

    return (x, y)


def move_mouse_smoothly_bezier(starting_point, control_point, target, steps):
    """Move mouse along bezier curve for smoother, more human-like movement."""
    for i in range(steps):
        t = i / steps
        bezierPointX, bezierPointY = calculate_bezier_point(t, starting_point, control_point, target)
        
        # Handle X movement with 127 limit
        if abs(bezierPointX - starting_point[0]) > 127:
            while abs(bezierPointX - starting_point[0]) > 127:
                step_x = 127 if bezierPointX > starting_point[0] else -127
                kmNet.move(int(step_x), 0)
                bezierPointX -= step_x
        
        # Handle Y movement with 127 limit
        if abs(bezierPointY - starting_point[1]) > 127:
            while abs(bezierPointY - starting_point[1]) > 127:
                step_y = 127 if bezierPointY > starting_point[1] else -127
                kmNet.move(0, int(step_y))
                bezierPointY -= step_y
        
        # Move remaining distance
        move_x = int(bezierPointX - starting_point[0])
        move_y = int(bezierPointY - starting_point[1])
        kmNet.move(move_x, move_y)
        
        # Update starting point for next iteration
        starting_point = (bezierPointX, bezierPointY)


# --- Anti-Recoil Thread (From Valorant-AI-Aimbot) ---
def anti_recoil_thread():
    """Thread that manages anti-recoil compensation while firing."""
    global anti_recoil_value, is_anti_recoil_active
    
    while True:
        if not is_anti_recoil_active:
            time.sleep(0.1)  # Sleep when not active to save resources
            continue
        
        # Check if left mouse button is pressed (firing)
        if kmNet.isdown_left() == 1:
            # Calculate anti-recoil increment based on settings (directly from VAI)
            anti_recoil_strength = float(CONFIG.get("anti_recoil_strength", 0.5))
            anti_recoil_multi = float(CONFIG.get("anti_recoil_multiplier", 25))
            
            # FIXED: Generate a NEGATIVE value for anti-recoil to force downward movement
            anti_recoil_value -= 0.025 * anti_recoil_strength * anti_recoil_multi
            time.sleep(0.0025)  # Same timing as VAI
        else:
            # Reset when not firing
            anti_recoil_value = 0
            time.sleep(0.01)  # Small sleep to prevent CPU usage


# --- Start Anti-Recoil Thread ---
def start_anti_recoil():
    """Starts the anti-recoil thread if enabled."""
    global is_anti_recoil_active
    
    if CONFIG.get("anti_recoil_enabled", True):  # Changed default to True
        is_anti_recoil_active = True
        recoil_thread = threading.Thread(target=anti_recoil_thread, daemon=True)
        recoil_thread.start()
        print("Anti-recoil thread started")


# --- Normal Aimbot Mode ---
def aimbot_tracking_normal(head_center_x, head_center_y, fov_center_x, fov_center_y):
    """Normal aimbot tracking with configurable smoothing - based on VAI."""
    global anti_recoil_value
      # Calculate aim speed based on sensitivity with additional scaling
    sensitivity = float(CONFIG.get("valorant_sensitivity", 0.4))
    
    # Modified formula with dampening factor to prevent overshooting
    # Lower values = slower, more controlled movement
    dampening_factor = 0.2  # Reduce speed by 80% from original formula
    
    # Scale the aim speed more conservatively for better control
    aim_speed = dampening_factor * (1/sensitivity)
    
    # Apply smoothing divisor (VAI uses division for smoothing)
    smoothing = float(CONFIG.get("aimbot_smoothing", 5.0))
    # Convert to VAI's 0.1-1.0 scale for compatibility
    smoothing_adjusted = smoothing / 10.0
    aim_speed_adjusted = aim_speed / smoothing_adjusted
      # Calculate vector to target
    x_offset = head_center_x - fov_center_x
    y_offset = head_center_y - fov_center_y
    
    # Horizontal movement calculation
    xdif = x_offset * aim_speed_adjusted
    
    # Vertical movement with anti-recoil compensation
    # INVERTED: Adding negative anti-recoil value to force downward movement
    ydif = (y_offset * aim_speed_adjusted) + (-anti_recoil_value)
    
    # Handle X movement (kmNet max value is 127)
    if abs(xdif) > 127:
        while abs(xdif) > 127:
            step_x = 127 if xdif > 0 else -127
            kmNet.move(int(step_x), 0)
            xdif -= step_x
    else:
        move_x = int(xdif)
    
    # Handle Y movement (kmNet max value is 127)
    if abs(ydif) > 127:
        while abs(ydif) > 127:
            step_y = 127 if ydif > 0 else -127
            kmNet.move(0, int(step_y))
            ydif -= step_y
    else:
        move_y = int(ydif)
    
    # Execute final movement
    kmNet.move(int(xdif), int(ydif))


# --- Rage Aimbot Mode (Fast, minimal smoothing) ---
def aimbot_tracking_rage(head_center_x, head_center_y, fov_center_x, fov_center_y):
    """Rage aimbot mode with minimal smoothing - based on VAI AimbotTrackingRage."""
    global anti_recoil_value
      # Calculate aim speed based on sensitivity with additional scaling
    sensitivity = float(CONFIG.get("valorant_sensitivity", 0.4))
    
    # Modified formula with dampening factor to prevent overshooting
    # Use slightly faster movement for rage mode, but still dampened
    dampening_factor = 0.25  # Reduce speed by 75% from original formula
    
    # Scale the aim speed more conservatively for better control
    aim_speed = dampening_factor * (1/sensitivity)
    
    # No smoothing division in rage mode (directly from VAI)
    x_offset = head_center_x - fov_center_x
    y_offset = head_center_y - fov_center_y
    
    # Direct multiplication without smoothing
    xdif = x_offset * aim_speed
    ydif = y_offset * aim_speed
    
    # Handle X movement (kmNet max value is 127)
    if abs(xdif) > 127:
        while abs(xdif) > 127:
            step_x = 127 if xdif > 0 else -127
            kmNet.move(int(step_x), 0)
            xdif -= step_x
    
    # Handle Y movement (kmNet max value is 127)
    if abs(ydif) > 127:
        while abs(ydif) > 127:
            step_y = 127 if ydif > 0 else -127
            kmNet.move(0, int(step_y))
            ydif -= step_y    # Execute final movement with anti-recoil
    # INVERTED: Using negative anti-recoil value to force downward movement
    kmNet.move(int(xdif), int(ydif + (-anti_recoil_value)))
    
    # Small delay after rage movement (from VAI)
    time.sleep(0.02)


# --- Flick Mode (with auto-fire) ---
def flick_bot_rage(head_center_x, head_center_y, fov_center_x, fov_center_y):
    """Flick bot in rage mode - fast flick + click (from VAI FlickBot)."""
    # Only proceed if not on cooldown
    if not can_fire_flick[0]:
        return
      # Calculate aim speed based on sensitivity with additional scaling
    sensitivity = float(CONFIG.get("valorant_sensitivity", 0.4))
    
    # Modified formula with dampening factor to prevent overshooting
    # Flick mode needs precise targeting
    dampening_factor = 0.2  # Reduce speed by 80% from original formula
    
    # Scale the aim speed more conservatively for better control
    aim_speed = dampening_factor * (1/sensitivity)
    
    # Right-click detection (from VAI)
    is_right_click = kmNet.isdown_right() == 1
    
    # Apply different multiplier based on right-click state (directly from VAI)
    if is_right_click:
        xdif = (head_center_x - fov_center_x) * aim_speed * 0.9
        ydif = (head_center_y - fov_center_y) * aim_speed * 0.9
    else:
        xdif = (head_center_x - fov_center_x) * aim_speed * 1.05
        ydif = (head_center_y - fov_center_y) * aim_speed * 1.05
    
    # Handle X movement
    if abs(xdif) > 127:
        while abs(xdif) > 127:
            step_x = 127 if xdif > 0 else -127
            kmNet.move(int(step_x), 0)
            xdif -= step_x
    
    # Handle Y movement
    if abs(ydif) > 127:
        while abs(ydif) > 127:
            step_y = 127 if ydif > 0 else -127
            kmNet.move(0, int(step_y))
            ydif -= step_y
    
    # Execute final movement
    kmNet.move(int(xdif), int(ydif))
    
    # Auto-click (directly from VAI FlickBot)
    kmNet.left(1)  # Press
    time.sleep(0.01)  # Short delay
    kmNet.left(0)  # Release
    
    # Start cooldown (from VAI)
    can_fire_flick[0] = False
    cooldown_ms = float(CONFIG.get("flick_cooldown_ms", 50)) / 1000.0
    threading.Thread(target=cooldown, args=(cooldown_ms,)).start()


def flick_bot_smooth(head_center_x, head_center_y, fov_center_x, fov_center_y):
    """Flick bot with smoothing (from VAI FlickBotLegit)."""
    # Only proceed if not on cooldown
    if not can_fire_flick[0]:
        return
    
    # Calculate aim speed based on sensitivity
    sensitivity = float(CONFIG.get("valorant_sensitivity", 0.4))
    aim_speed = 1.0 * (1/sensitivity)
    
    # Right-click detection
    is_right_click = kmNet.isdown_right() == 1
    
    # Apply different multiplier based on right-click state (from VAI)
    if is_right_click:
        xdif = (head_center_x - fov_center_x) * aim_speed * 0.9
        ydif = (head_center_y - fov_center_y) * aim_speed * 0.9
    else:
        xdif = (head_center_x - fov_center_x) * aim_speed * 1.05
        ydif = (head_center_y - fov_center_y) * aim_speed * 1.05
    
    # Get smoothing steps (from VAI)
    steps = int(CONFIG.get("flick_smoothing", 50))
    
    # Calculate step sizes
    x_step = xdif / steps
    y_step = ydif / steps
    
    # Track accumulated fractional movement
    x_accum = 0.0
    y_accum = 0.0
    
    # Move in steps for smoother appearance (from VAI FlickBotLegit)
    for _ in range(steps):
        x_accum += x_step
        y_accum += y_step
        
        # Execute movement with rounding
        kmNet.move(int(round(x_accum)), int(round(y_accum)))
        
        # Subtract the integer part we've already moved
        x_accum -= int(round(x_accum))
        y_accum -= int(round(y_accum))
    
    # Auto-click
    kmNet.left(1)  # Press
    time.sleep(0.01)  # Short delay
    kmNet.left(0)  # Release
    
    # Cooldown
    cooldown_ms = float(CONFIG.get("flick_cooldown_ms", 50)) / 1000.0
    time.sleep(cooldown_ms)
    can_fire_flick[0] = True


# --- Silent Aim ---
def silent_aim(head_center_x, head_center_y, fov_center_x, fov_center_y):
    """Silent aim - from VAI SilentAim."""
    # Only proceed if not on cooldown
    if not can_fire_flick[0]:
        return
    
    # Calculate aim speed based on sensitivity
    sensitivity = float(CONFIG.get("valorant_sensitivity", 0.4))
    aim_speed = 1.0 * (1/sensitivity)
    
    # Right-click detection
    is_right_click = kmNet.isdown_right() == 1
    
    # Apply different multiplier based on right-click state (from VAI)
    if is_right_click:
        xdif = (head_center_x - fov_center_x) * aim_speed * 0.9
        ydif = (head_center_y - fov_center_y) * aim_speed * 0.9
    else:
        xdif = (head_center_x - fov_center_x) * aim_speed * 1.05
        ydif = (head_center_y - fov_center_y) * aim_speed * 1.05
    
    # Calculate reverse movement (From VAI SilentAim)
    reverse_x = -xdif
    reverse_y = -ydif
    
    # Handle X movement (exact implementation from VAI)
    if abs(xdif) > 127:
        while abs(xdif) > 127:
            step_x = 127 if xdif > 0 else -127
            kmNet.move(int(step_x), 0)
            xdif -= step_x
    else:
        kmNet.move(int(xdif), 0)
    
    # Handle Y movement
    if abs(ydif) > 127:
        while abs(ydif) > 127:
            step_y = 127 if ydif > 0 else -127
            kmNet.move(0, int(step_y))
            ydif -= step_y
    else:
        kmNet.move(0, int(ydif))
    
    # Micro delay and click (from VAI)
    time.sleep(1e-44)  # Ultra short delay
    kmNet.left(1)  # Press
    time.sleep(1e-44)  # Ultra short delay
    kmNet.left(0)  # Release
    
    # Move back to original position - X (from VAI)
    if abs(reverse_x) > 127:
        while abs(reverse_x) > 127:
            step_reverse_x = 127 if reverse_x > 0 else -127
            kmNet.move(int(step_reverse_x), 0)
            reverse_x -= step_reverse_x
    else:
        kmNet.move(int(reverse_x), 0)
    
    # Move back to original position - Y
    if abs(reverse_y) > 127:
        while abs(reverse_y) > 127:
            step_reverse_y = 127 if reverse_y > 0 else -127
            kmNet.move(0, int(step_reverse_y))
            reverse_y -= step_reverse_y
    else:
        kmNet.move(0, int(reverse_y))
    
    # Start cooldown
    can_fire_flick[0] = False
    cooldown_ms = float(CONFIG.get("silent_cooldown_ms", 50)) / 1000.0
    threading.Thread(target=cooldown, args=(cooldown_ms,)).start()


# --- Main Interface Functions ---

def can_click():
    """Check if enough time has passed since last click for triggerbot."""
    global last_click_time
    current_time = time.time()
    trigger_delay = float(CONFIG.get("trigger_delay_ms", 10)) / 1000.0
    
    if current_time - last_click_time >= trigger_delay:
        last_click_time = current_time
        return True
    return False


def calculate_aim_point(xmin, ymin, xmax, ymax):
    """Calculate the aim point with height adjustment."""
    center_x = (xmin + xmax) / 2
    
    # Apply any height adjustment (0 = center, positive = higher, negative = lower)
    adj = float(CONFIG.get("aim_height_adjustment", 0))
    adj = max(min(adj, 10), -10)  # Limit adjustment range
    height_factor = 0.5 - (adj / 20.0)
    
    # Calculate Y position with adjustment
    center_y = ymin + (ymax - ymin) * height_factor
    
    return center_x, center_y


def get_aim_offset(cX, cY, mid):
    """Calculate x, y offsets from center."""
    mid_float = float(mid)
    x = cX - mid_float
    y = cY - mid_float
    return x, y


def aim_at_target(x, y, mode=None):
    """
    Main aiming function that dispatches to the appropriate aiming mode.
    Args:
        x, y: Target offset from center
        mode: Optional override for aiming mode
    """
    # Check if aimbot is enabled
    if not CONFIG.get("aimbot_enabled", True):
        return False
    
    # Get center coordinates from fov
    fov = CONFIG.get("fov", 320)
    fov_center = fov / 2
    
    # Calculate target coordinates
    target_x = x + fov_center
    target_y = y + fov_center
    
    # Use provided mode or get from config
    if mode is None:
        mode = CONFIG.get("aimbot_mode", "normal")
    
    # Make sure anti-recoil is running
    global is_anti_recoil_active
    if not is_anti_recoil_active and CONFIG.get("anti_recoil_enabled", True):
        start_anti_recoil()
    
    # Dispatch to appropriate aiming mode
    if mode == "rage":
        aimbot_tracking_rage(target_x, target_y, fov_center, fov_center)
    elif mode == "flick":
        if int(CONFIG.get("flick_smoothing", 50)) <= 5:
            flick_bot_rage(target_x, target_y, fov_center, fov_center)
        else:
            flick_bot_smooth(target_x, target_y, fov_center, fov_center)
    elif mode == "silent":
        silent_aim(target_x, target_y, fov_center, fov_center)
    else:  # normal mode (default)
        aimbot_tracking_normal(target_x, target_y, fov_center, fov_center)
    
    return True

# Start anti-recoil thread if enabled
if CONFIG.get("anti_recoil_enabled", True):
    start_anti_recoil()