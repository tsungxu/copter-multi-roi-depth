import time
import cv2
import numpy as np
import depthai as dai
from pymavlink import mavutil
import threading
from sys import stdout

debug = False # run debug commands when true

newConfig = False

########################## 
# Setup DepthAI pipeline #
##########################

pipeline = dai.Pipeline() # Create DepthAI pipeline

# Define DepthAI sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
camRgb = pipeline.create(dai.node.ColorCamera)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# Set properties for depth AI pipeline
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

frame_size = (960, 540)
camRgb.setPreviewSize(frame_size)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Config for multiple ROIs to cover field of view
spatialLocationCalculator.inputConfig.setWaitForMessage(False)

for i in range(3):  # adjust this to change the number of ROIs in width
    for j in range(3):  # adjust this to change the number of ROIs in height
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 400
        config.depthThresholds.upperThreshold = 9000
        config.roi = dai.Rect(
            dai.Point2f(i * 0.33, j * 0.33),
            dai.Point2f((i + 1) * 0.33, (j + 1) * 0.33),
        )

        spatialLocationCalculator.initialConfig.addROI(config)

# Linking the pipeline sources
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# Create XLinkOut for RGB frames
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

########################## 
# Setup MAVLINK connection #
##########################

mavlink_connection = mavutil.mavlink_connection('/dev/ttyTHS1', baud=115200) # Create a MAVLink connection

heartbeat_frequency = 1  # 1 Hz
distance_sensor_frequency = 25

last_heartbeat_time = time.time()
last_distance_sensor_time = time.time()

lidar_distance = 0.0  # Global variable to store lidar distance

# Define a function to receive lidar rangefinder messages
def receive_distance_sensor():
    global lidar_distance

    while True:
        msg = mavlink_connection.recv_match(type='RANGEFINDER', blocking=True)
        lidar_distance = msg.distance
        # print(f'Current distance: {lidar_distance}') # already in meters

# Start a separate thread to receive distance sensor messages
distance_sensor_thread = threading.Thread(target=receive_distance_sensor, daemon=True)
distance_sensor_thread.start()

########################## 
# Setup video writer #
##########################

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_file_path = '/home/tsungxu/video_test.mp4'
frame_rate = 30.0
video_out = cv2.VideoWriter(video_file_path, fourcc, frame_rate, frame_size)

##########################
# Connect to device and start pipeline
##########################

with dai.Device(pipeline) as device:
    print("camera connected and running script")

    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)

    # rgb output
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    # initiate ui variables
    if debug:
        fontType = cv2.FONT_HERSHEY_TRIPLEX
        color = (255, 255, 255)

    try: # handle termination with thread running for lidar
        while True:
            inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived

            depthFrame = inDepth.getFrame() # depthFrame values are in millimeters
            # make zero values large (10m) for obstacle detection outdoors
            depthFrame = np.where(depthFrame == 0, 10000, depthFrame)

            depth_downscaled = depthFrame[::4]
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
            max_depth = np.percentile(depth_downscaled, 99)
            depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            spatialData = spatialCalcQueue.get().getSpatialLocations()

            minDepth = float('inf') # initialize minimum depth to send to mavlink as 10 meters
            minRoi = None  # initialize minRoi for viewing on saved video
            validDepthData = [] # for non-zero minimum depth

            # get the rgb frame for writing to file and, when debugging, preview 
            inRgb = rgbQueue.get()
            frame = inRgb.getCvFrame()

            # filter the spatial data that have a non-zero depth
            for depthData in spatialData:
                roi = depthData.config.roi
                roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])  # adjust this to frame's dimensions
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)

                # Check if the minimum depth in the ROI is greater than zero and mask and ignore zero values
                roiDepth = np.ma.masked_equal(depthFrame[ymin:ymax, xmin:xmax], 0)
                min_roiDepth = np.ma.min(roiDepth)
                if min_roiDepth is not np.ma.masked:  # Check if the minimum value is not masked
                    validDepthData.append((depthData, xmin, ymin, xmax, ymax))

            # use valid spatial data for calculating depth and drawing ui
            for depthData, xmin, ymin, xmax, ymax in validDepthData:  # unpack tuples here
                roiDepth = np.ma.masked_equal(depthFrame[ymin:ymax, xmin:xmax], 0)
                average_roiDepth = np.ma.mean(roiDepth)

                if average_roiDepth is not np.ma.masked:  # Check if the average value is not masked
                    # Update minDepth if this ROI's distance is smaller
                    if average_roiDepth < minDepth:
                        minDepth = average_roiDepth
                        minRoi = (xmin, ymin, xmax, ymax)  # store this ROI as the one with smallest distance

                    # Draw UI for depth frame video preview
                    if debug:
                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, thickness=2)
                        cv2.putText(depthFrameColor, "{:.1f}m".format(average_roiDepth / 1000), (xmin + 10, ymin + 20), fontType, 0.6, color)
            
            ##################
            # send heartbeat to flight controller #
            ##################

            # Send a heartbeat message every second. https://mavlink.io/en/mavgen_python/#heartbeat
            current_time = time.time()
            
            if current_time - last_heartbeat_time > heartbeat_frequency:
                # The arguments for the heartbeat message are type, autopilot, base_mode, custom_mode, system_status, and mavlink_version.
                mavlink_connection.mav.heartbeat_send(
                    18,  # Type of the system - MAV_TYPE_ONBOARD_CONTROLLER: https://mavlink.io/en/messages/minimal.html#MAV_TYPE
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,  # Autopilot type
                    0,  # System mode https://mavlink.io/en/messages/common.html#MAV_MODE
                    0,  # Custom mode, this is system specific
                    3,  # System status https://mavlink.io/en/messages/common.html#MAV_STATE
                )
                last_heartbeat_time = current_time

            ##################
            # send distance sensor data to fcu. For multi_roi, this needs to be out of for loop, which iterates once for each roi #
            ##################

            # see https://mavlink.io/en/messages/common.html for fields
            if current_time - last_distance_sensor_time > 1/distance_sensor_frequency:
                time_difference = current_time - last_distance_sensor_time  # calculate time difference before updating last_distance_sensor_time
                
                mavlink_connection.mav.distance_sensor_send(
                    time_boot_ms=0,  # timestamp in ms since system boot
                    min_distance=40,  # minimum distance the sensor can measure (cm)
                    max_distance=900,  # maximum distance the sensor can measure (cm)
                    current_distance=int((minDepth) / 10),  # current distance measured (cm)
                    type=4,  # type of distance sensor: 0 = laser, 4 = unknown. See MAV_DISTANCE_SENSOR
                    id=1,  # onboard ID of the sensor
                    orientation=0,  # forward facing, see MAV_SENSOR_ORIENTATION
                    covariance=0,  # measurement covariance in centimeters, 0 for unknown / invalid readings
                )
                last_distance_sensor_time = current_time # reset last distance sensor time

                # in the rgb video output, draw a rectangle around the closest object ROI
                # if minRoi is not None:
                # Get the aspect ratios
                aspect_ratio_width = frame.shape[1] / depthFrameColor.shape[1]
                aspect_ratio_height = frame.shape[0] / depthFrameColor.shape[0]

                # Scale the ROI coordinates
                scaled_roi = [
                    int(minRoi[0] * aspect_ratio_width),
                    int(minRoi[1] * aspect_ratio_height),
                    int(minRoi[2] * aspect_ratio_width),
                    int(minRoi[3] * aspect_ratio_height),
                ]

                # Now use scaled_roi to draw the rectangle on the RGB frame
                distance_text = f"Closest object: {round(minDepth/1000, 2)} meters"
                elevation_text = f"Elevation: {round(lidar_distance, 2)} meters"
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Get text sizes
                (distance_text_w, distance_text_h), _ = cv2.getTextSize(distance_text, font, 0.6, 1)
                (elevation_text_w, elevation_text_h), _ = cv2.getTextSize(elevation_text, font, 0.6, 1)

                # Create a semi-transparent rectangle with size enough to fit both lines of text
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (max(distance_text_w, elevation_text_w) + 5, distance_text_h + elevation_text_h + 30), (196, 196, 196), -1) 

                # Combine the original frame with the overlay
                alpha = 0.7  # Transparency factor
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Write both lines of text onto the frame
                cv2.putText(frame, distance_text, (10, 30), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, elevation_text, (10, 30 + distance_text_h + 10), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # Now use scaled_roi to draw the roi bounding box on the RGB frame
                # Only draw the rectangle on the RGB frame when the distance is less than 10 meters
                if minDepth < 9500:
                    cv2.rectangle(frame, (scaled_roi[0], scaled_roi[1]), (scaled_roi[2], scaled_roi[3]), (255, 255, 255), 3)


            # reset minDepth for next iteration
            minDepth = float('inf')
            validDepthData = []
                
            # video_out.write(frame)
            try:
                video_out.write(frame) # Write frame to video file
            except cv2.error as e:
                print(f"Error writing frame to video file: {e}")
                break

            # Show the preview video frames
            if debug: 
                cv2.imshow("depth", depthFrameColor)
                cv2.imshow("rgb", frame)

                key = cv2.waitKey(1) & 0xFF  # Get ASCII value of key
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        print(" Stopping.")

    # release video writer
    video_out.release()
