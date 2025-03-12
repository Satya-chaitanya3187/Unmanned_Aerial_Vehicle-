from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import RPi.GPIO as GPIO
import math
import logging

# Configure logging
logging.basicConfig(
    filename='/home/a201-/Log1.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Define GPIO pin for obstacle detection
DETECTION_PIN = 18
GPIO.setup(DETECTION_PIN, GPIO.IN)

# Connect to the vehicle
vehicle = connect('/dev/serial0', baud=921600, wait_ready=True)

def arm_and_takeoff(target_altitude):
    logging.info("Arming motors")
    while not vehicle.is_armable:
        logging.info("Waiting for vehicle to become armable...")
        time.sleep(1)

    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        logging.info("Waiting for arming...")
        time.sleep(1)

    logging.info("Taking off...")
    vehicle.simple_takeoff(target_altitude)

    while True:
        current_altitude = vehicle.location.global_relative_frame.alt
        logging.info(f"Current altitude: {current_altitude:.2f} m")
        if current_altitude >= target_altitude * 0.95:
            logging.info("Target altitude reached.")
            break
        time.sleep(1)

def move_east(distance):
    current_location = vehicle.location.global_relative_frame
    earth_radius = 6378137.0
    dLon = distance / (earth_radius * math.cos(math.pi * current_location.lat / 180))

    newlon = current_location.lon + (dLon * 180 / math.pi)
    vehicle.simple_goto(LocationGlobalRelative(current_location.lat, newlon, current_location.alt))

def land():
    logging.info("Landing...")
    vehicle.mode = VehicleMode("LAND")
    while vehicle.armed:
        logging.info("Waiting for drone to land...")
        time.sleep(1)

def obstacle_detected():
    return GPIO.input(DETECTION_PIN) == GPIO.HIGH

def main():
    try:
        logging.info("Waiting for Arduino data...")
        time.sleep(5)  # Delay for Arduino to initialize and send data

        arm_and_takeoff(1)
        logging.info("Hovering for observation...")
        time.sleep(5)

        if obstacle_detected():
            logging.info("Obstacle detected! Moving east 5 meters.")
            move_east(5)
            time.sleep(10)  # Wait to ensure position is reached
        else:
            logging.info("No obstacle detected. Hovering for 15 seconds.")
            time.sleep(15)

        land()

    except KeyboardInterrupt:
        logging.info("Mission interrupted by user.")

    finally:
        vehicle.close()
        logging.info("Connection closed.")

if name == "main":
    main()
