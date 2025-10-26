import sys
import pyzed.sl as sl
import cv2
from ur5py.ur5 import UR5Robot, UR2RT
from autolab_core import RigidTransform
import threading, time


WRIST_CAM_ID = 16347230
FIXED_CAM_ID = 22008760



def camera_setup():
    zed = sl.Camera()

    #intial parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # HD1080, HD720, etc.
    init_params.camera_fps = 30
    init_params.camera_image_flip = sl.FLIP_MODE.AUTO
    init_params.set_from_serial_number(FIXED_CAM_ID)

    # Open the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera with serial {FIXED_CAM_ID}")
        sys.exit(1)

    # Prepare image container
    image = sl.Mat()

    print("Press 'q' to exit.")
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()

            # Show image in OpenCV window
            cv2.imshow("ZED Live Feed", frame)

             # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to grab frame.")

    # Cleanup
    zed.close()
    cv2.destroyAllWindows()

def robot_setup(ur):
    home_pose = ur.get_pose()
    print("Home pose: ", home_pose)

    ur.move_pose(RigidTransform(rotation = home_pose.rotation, translation = home_pose.translation + [-0.1, -0.1, -0.05]))

    ur.move_pose(home_pose)

    print("Robot moved to home")

    


def main():
    
    ur = UR5Robot()
    print("UR5 ready")

    print("Robot home pose: ", ur.get_pose())
    print("Robot home joints: ", ur.get_joints())

    print("Calibrating Robot:")

    # Start camera in a separate thread
    camera_thread = threading.Thread(target=camera_setup)
    camera_thread.daemon = True  # Dies when main thread dies
    camera_thread.start()
    
    # Give camera a moment to initialize
    time.sleep(5)
    
    print("Calibrating Robot:")
    robot_setup(ur)
    print("Robot calibrated")
    
    # Keep main thread alive while camera runs
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()