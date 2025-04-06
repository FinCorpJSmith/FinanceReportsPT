import math
import time
import threading
import kmNet

# This configuration will be updated from valoai.py
CONFIG = {
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
last_click_time = 0

def CalculateBezierPoint(t, p0, p1, p2):
    """Calculate a point along a quadratic Bezier curve."""
    u = 1 - t
    tt = t * t
    uu = u * u
    
    x = int(uu * p0[0] + 2 * u * t * p1[0] + tt * p2[0])
    y = int(uu * p0[1] + 2 * u * t * p1[1] + tt * p2[1])
    
    return (x, y)

def move_mouse_bezier(start_x, start_y, target_x, target_y, steps=None):
    """Move mouse along a Bezier curve from current position to target with acceleration."""
    # Use default values from CONFIG
    steps = steps or CONFIG["bezier_steps"]
    accel_factor = CONFIG.get("accel_factor", 1.3)
    start_delay = CONFIG.get("bezier_start_delay", 0.006)
    end_delay = CONFIG.get("bezier_end_delay", 0.0007)
    
    # Apply valorant sensitivity adjustment
    sensitivity_factor = CONFIG["valorant_sensitivity"] / 0.4
    
    starting_point = (0, 0)  # Starting with no movement
    
    # Create a control point for the Bezier curve
    offset_x = (target_x - start_x) * 0.3
    offset_y = (target_y - start_y) * 0.2
    control_point = (offset_x, offset_y)
    
    target_point = (target_x, target_y)
    
    current_point = starting_point
    
    for i in range(steps):
        # Apply acceleration curve to the t parameter (ease-in)
        progress = i / steps
        t = pow(progress, accel_factor)
        
        # Calculate point on bezier curve with accelerated parameter
        bezier_x, bezier_y = CalculateBezierPoint(t, starting_point, control_point, target_point)
        
        # Calculate movement from current position
        move_x = bezier_x - current_point[0]
        move_y = bezier_y - current_point[1]
        
        # Adjust for sensitivity - higher sensitivity means smaller movements needed
        move_x = int(move_x / sensitivity_factor)
        move_y = int(move_y / sensitivity_factor)
        
        # Handle movement limits (kmNet can't handle moves larger than 127)
        if abs(move_x) > 127:
            while abs(move_x) > 127:
                step_x = 127 if move_x > 0 else -127
                kmNet.move(step_x, 0)
                move_x -= step_x
        
        if abs(move_y) > 127:
            while abs(move_y) > 127:
                step_y = 127 if move_y > 0 else -127
                kmNet.move(0, step_y)
                move_y -= step_y
        
        # Move the remaining distance
        if move_x != 0 or move_y != 0:
            kmNet.move(int(move_x), int(move_y))
        
        current_point = (bezier_x, bezier_y)
        
        # Calculate delay that decreases over time (starts slow, gets faster)
        delay = start_delay - (start_delay - end_delay) * progress
        time.sleep(delay)

def aimbot_tracking_legit(x, y):
    """Smooth, human-like tracking for aim assistance with acceleration."""
    # Base smoothing value
    smoothing = CONFIG["smoothing"]
    
    # Apply valorant sensitivity adjustment
    sensitivity_factor = CONFIG["valorant_sensitivity"] / 0.4
    
    # Calculate distance to target
    distance = math.sqrt(x*x + y*y)
    
    # Apply acceleration curve based on distance
    # - For targets far away, use higher smoothing (slower, more precise initial movement)
    # - For targets nearby, use lower smoothing (faster tracking when already on target)
    distance_factor = min(1.0, distance / 80)  # Normalize distance, max effect at 80 pixels
    
    # Calculate acceleration smoothing - larger distance = more smoothing (slower)
    # As we get closer to target, smoothing decreases (faster)
    # The acceleration_factor ranges from 0.6 (on target) to 1.0 (far from target)
    acceleration_factor = 0.6 + (1.7 * distance_factor)
    
    # Apply acceleration to smoothing
    dynamic_smoothing = smoothing * acceleration_factor
    
    # Calculate speed with acceleration applied
    adjusted_speed = 1.0 / (dynamic_smoothing * sensitivity_factor)
    
    # Calculate movement with dynamic smoothing applied
    xdif = x * adjusted_speed
    ydif = y * adjusted_speed
    
    # Handle movement limits
    if abs(xdif) > 127:
        while abs(xdif) > 127:
            step_xdif = 127 if xdif > 0 else -127
            kmNet.move(int(step_xdif), 0)
            xdif -= step_xdif
    
    if abs(ydif) > 127:
        while abs(ydif) > 127:
            step_ydif = 127 if ydif > 0 else -127
            kmNet.move(0, int(step_ydif))
            ydif -= step_ydif
    
    kmNet.move(int(xdif), int(ydif))

def aimbot_tracking_rage(x, y):
    """Aggressive tracking with minimal smoothing."""
    # Use a reduced smoothing value for rage mode
    smoothing = CONFIG["smoothing"] * 0.5  # Lower smoothing = faster movement
    
    # Apply valorant sensitivity adjustment
    sensitivity_factor = CONFIG["valorant_sensitivity"] / 0.4
    adjusted_speed = 1.0 / (smoothing * sensitivity_factor)
    
    xdif = x * adjusted_speed
    ydif = y * adjusted_speed
    
    # Handle movement limits
    if abs(xdif) > 127:
        while abs(xdif) > 127:
            step_xdif = 127 if xdif > 0 else -127
            kmNet.move(int(step_xdif), 0)
            xdif -= step_xdif
    
    if abs(ydif) > 127:
        while abs(ydif) > 127:
            step_ydif = 127 if ydif > 0 else -127
            kmNet.move(0, int(step_ydif))
            ydif -= step_ydif
    
    kmNet.move(int(xdif), int(ydif))
    time.sleep(0.02)

def can_click():
    """Check if clicking is allowed based on cooldown."""
    global last_click_time
    current_time = time.time()
    if current_time - last_click_time >= CONFIG["flick_cooldown"]:
        last_click_time = current_time
        return True
    return False

def flick_bot(x, y):
    """Implement flick aiming functionality with improved accuracy."""
    # Apply standard smoothing
    smoothing = CONFIG["smoothing"] * 0.7  # Less smoothing for flicks
    
    # Apply valorant sensitivity adjustment
    sensitivity_factor = CONFIG["valorant_sensitivity"] / 0.4
    adjusted_speed = 1.0 / (smoothing * sensitivity_factor)
    
    # Calculate exact target coordinates
    xdif = x * adjusted_speed
    ydif = y * adjusted_speed
    
    # Handle movement limits while preserving exact targeting
    if abs(xdif) > 127:
        while abs(xdif) > 127:
            step_xdif = 127 if xdif > 0 else -127
            kmNet.move(int(step_xdif), 0)
            xdif -= step_xdif
    
    if abs(ydif) > 127:
        while abs(ydif) > 127:
            step_ydif = 127 if ydif > 0 else -127
            kmNet.move(0, int(step_ydif))
            ydif -= step_ydif
    
    # Move the final precise distance
    kmNet.move(int(xdif), int(ydif))
    
    # Immediate click for better timing
    kmNet.left(1)
    # Short delay to ensure the click registers
    time.sleep(0.03)
    kmNet.left(0)

def flick_bot_legit(x, y):
    """Implement smoother flick aiming with steps."""
    smoothing = CONFIG["smoothing"] * 0.8  # Slightly less smoothing for flicks
    
    # Apply valorant sensitivity adjustment
    sensitivity_factor = CONFIG["valorant_sensitivity"] / 0.4
    adjusted_speed = 1.0 / (smoothing * sensitivity_factor)
    
    # Calculate exact movement needed
    xdif = x * adjusted_speed
    ydif = y * adjusted_speed
    
    # More precise stepping for smoother movement
    steps = max(5, CONFIG["bezier_steps"])
    
    # Calculate step size (the smaller the steps, the more precise the movement)
    x_step = xdif / steps
    y_step = ydif / steps
    
    # Use accumulators to keep track of floating-point movement
    x_accum = 0.0
    y_accum = 0.0
    
    # Move in small steps to ensure accuracy
    for i in range(steps):
        # Add the step to the accumulator
        x_accum += x_step
        y_accum += y_step
        
        # Get the integer movement for this step
        move_x = int(round(x_accum))
        move_y = int(round(y_accum))
        
        if abs(move_x) > 127:
            while abs(move_x) > 127:
                step_move_x = 127 if move_x > 0 else -127
                kmNet.move(step_move_x, 0)
                move_x -= step_move_x
        
        if abs(move_y) > 127:
            while abs(move_y) > 127:
                step_move_y = 127 if move_y > 0 else -127
                kmNet.move(0, step_move_y)
                move_y -= step_move_y
        
        if move_x != 0 or move_y != 0:
            kmNet.move(move_x, move_y)
        
        # Reset accumulator by subtracting what we've already moved
        x_accum -= move_x
        y_accum -= move_y
        
        # Progressive delay - faster at the end of the movement
        delay = 0.006 * (1 - i/steps)
        time.sleep(max(0.001, delay))
    
    # Ensure we've arrived at the exact target position
    # Calculate any remaining movement due to rounding errors
    remaining_x = x * adjusted_speed - sum([int(round(x_step * i)) for i in range(steps)])
    remaining_y = y * adjusted_speed - sum([int(round(y_step * i)) for i in range(steps)])
    
    # Apply the remaining movement to ensure perfect accuracy
    if abs(remaining_x) > 0 or abs(remaining_y) > 0:
        kmNet.move(int(round(remaining_x)), int(round(remaining_y)))
    
    # Click immediately after arriving at target
    kmNet.left(1)
    time.sleep(0.03)
    kmNet.left(0)

def silent_aim(x, y):
    """Implement silent aim (move to target, shoot, move back)."""
    smoothing = CONFIG["smoothing"] * 0.5  # Less smoothing for silent aim
    
    # Apply valorant sensitivity adjustment
    sensitivity_factor = CONFIG["valorant_sensitivity"] / 0.4
    adjusted_speed = 1.0 / (smoothing * sensitivity_factor)
    
    xdif = x * adjusted_speed
    ydif = y * adjusted_speed
    
    # Store the reverse movements needed to go back
    reverse_x = -xdif
    reverse_y = -ydif
    
    # Move to target
    if abs(xdif) > 127:
        while abs(xdif) > 127:
            step_xdif = 127 if xdif > 0 else -127
            kmNet.move(int(step_xdif), 0)
            xdif -= step_xdif
    
    if abs(ydif) > 127:
        while abs(ydif) > 127:
            step_ydif = 127 if ydif > 0 else -127
            kmNet.move(0, int(step_ydif))
            ydif -= step_ydif
    
    kmNet.move(int(xdif), int(ydif))
    
    # Minimal delay
    time.sleep(0.001)
    
    # Click
    kmNet.left(1)
    kmNet.left(0)
    
    # Minimal delay
    time.sleep(0.001)
    
    # Move back
    if abs(reverse_x) > 127:
        while abs(reverse_x) > 127:
            step_reverse_x = 127 if reverse_x > 0 else -127
            kmNet.move(int(step_reverse_x), 0)
            reverse_x -= step_reverse_x
    
    if abs(reverse_y) > 127:
        while abs(reverse_y) > 127:
            step_reverse_y = 127 if reverse_y > 0 else -127
            kmNet.move(0, step_reverse_y)
            reverse_y -= step_reverse_y
    
    kmNet.move(int(reverse_x), int(reverse_y))

def aim_at_target(x, y, mode="legit", snap_fov=20):
    """Main function to aim at a target using the specified mode."""
    # If the target is within the snap FOV and it's just a small adjustment, use direct movement
    if abs(x) <= snap_fov/2 and abs(y) <= snap_fov/2:
        if mode != "silent":  # Silent aim handles its own movement
            # Apply valorant sensitivity adjustment
            sensitivity_factor = CONFIG["valorant_sensitivity"] / 0.4
            adjusted_x = int(x / sensitivity_factor)
            adjusted_y = int(y / sensitivity_factor)
            
            kmNet.mask_x(1)
            kmNet.mask_y(1)
            kmNet.move(adjusted_x, adjusted_y)
            kmNet.unmask_all()
        
        return True

    # Choose aiming method based on mode
    if mode == "legit":
        aimbot_tracking_legit(x, y)
    elif mode == "rage":
        aimbot_tracking_rage(x, y)
    elif mode == "bezier":
        move_mouse_bezier(0, 0, x, y, CONFIG["bezier_steps"])
    elif mode == "flick" and can_click():
        flick_bot(x, y)
    elif mode == "flick_legit" and can_click():
        flick_bot_legit(x, y)
    elif mode == "silent" and can_click():
        silent_aim(x, y)
    else:
        # If no specific mode matched, default to legit
        aimbot_tracking_legit(x, y)
    
    return True

def calculate_aim_point(xmin, ymin, xmax, ymax):
    """Calculate the aim point with height adjustment."""
    center_x = (xmin + xmax) / 2
    
    # Apply height adjustment (-10 to 10 range, negative is higher)
    height_factor = 0.5 + (CONFIG["aim_height_adjustment"] / 20)
    center_y = ymin + (ymax - ymin) * height_factor
    
    return center_x, center_y

def get_aim_offset(cX, cY, mid):
    """Calculate x, y offsets from center."""
    x = cX - mid if cX > mid else -(mid - cX)
    y = cY - mid if cY > mid else -(mid - cY)
    return x, y