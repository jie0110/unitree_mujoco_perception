ROBOT = "g1" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1" 
ROBOT_SCENE = "/home/jiewang/instinct_project/unitree_mujoco/unitree_robots/g1/scene_29dof_terrain.xml"
DOMAIN_ID = 6 # Domain id
INTERFACE = "lo" # Interface 

USE_JOYSTICK = 1 # Simulate Unitree WirelessController using a gamepad
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 1 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = True # Virtual spring band, used for lifting h1

SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer

ENABLE_DEPTH_RENDER = True   # Show depth image in OpenCV window and publish point clouds
CAMERA_NAME = "depth_cam"    # MuJoCo camera name for depth rendering
CAMERA_WIDTH = 480            # Camera resolution width (pixels)
CAMERA_HEIGHT = 270           # Camera resolution height (pixels)
CAMERA_MAX_DEPTH = 2.5       # Depth clamp range (meters)
CAMERA_DT = 0.0166            # 60 fps for depth rendering thread
CAMERA_FRAME_ID = "depth_cam"  # frame_id written into published PointCloud2 messages
