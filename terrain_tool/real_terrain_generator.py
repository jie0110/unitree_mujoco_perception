import xml.etree.ElementTree as xml_et
import numpy as np
import cv2
import noise

ROBOT = "g1"
INPUT_SCENE_PATH = "./scene.xml"
OUTPUT_SCENE_PATH = "../unitree_robots/" + ROBOT + "/scene_29dof_terrain_with_camera.xml"


# zyx euler angle to quaternion
def euler_to_quat(roll, pitch, yaw):
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


# zyx euler angle to rotation matrix
def euler_to_rot(roll, pitch, yaw):
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )

    rot_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


# 2d rotate
def rot2d(x, y, yaw):
    nx = x * np.cos(yaw) - y * np.sin(yaw)
    ny = x * np.sin(yaw) + y * np.cos(yaw)
    return nx, ny


# 3d rotate
def rot3d(pos, euler):
    R = euler_to_rot(euler[0], euler[1], euler[2])
    return R @ pos


def list_to_str(vec):
    return " ".join(str(s) for s in vec)


class TerrainGenerator:

    def __init__(self) -> None:
        self.scene = xml_et.parse(INPUT_SCENE_PATH)
        self.root = self.scene.getroot()
        self.worldbody = self.root.find("worldbody")
        self.asset = self.root.find("asset")

    # Add Box to scene
    def AddBox(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0],
               size=[0.1, 0.1, 0.1]):
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = "box"
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddGeometry(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0],
               size=[0.1, 0.1],geo_type="box"):

        # geo_type supports "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = geo_type
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddStairs(self,
                  init_pos=[1.0, 0.0, 0.0],
                  yaw=0.0,
                  width=0.2,
                  height=0.15,
                  length=1.5,
                  stair_nums=10):

        local_pos = [0.0, 0.0, -0.5 * height + init_pos[2]]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw], [width, length, height])

    def AddSuspendStairs(self,
                         init_pos=[1.0, 0.0, 0.0],
                         yaw=1.0,
                         width=0.2,
                         height=0.15,
                         length=1.5,
                         gap=0.1,
                         stair_nums=10):

        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw],
                        [width, length, abs(height - gap)])

    def AddRoughGround(self,
                       init_pos=[1.0, 0.0, 0.0],
                       euler=[0.0, -0.0, 0.0],
                       nums=[10, 10],
                       box_size=[0.5, 0.5, 0.5],
                       box_euler=[0.0, 0.0, 0.0],
                       separation=[0.2, 0.2],
                       box_size_rand=[0.05, 0.05, 0.05],
                       box_euler_rand=[0.2, 0.2, 0.2],
                       separation_rand=[0.05, 0.05]):

        local_pos = [0.0, 0.0, -0.5 * box_size[2]]
        new_separation = np.array(separation) + np.array(
            separation_rand) * np.random.uniform(-1.0, 1.0, 2)
        for i in range(nums[0]):
            local_pos[0] += new_separation[0]
            local_pos[1] = 0.0
            for j in range(nums[1]):
                new_box_size = np.array(box_size) + np.array(
                    box_size_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_box_euler = np.array(box_euler) + np.array(
                    box_euler_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_separation = np.array(separation) + np.array(
                    separation_rand) * np.random.uniform(-1.0, 1.0, 2)

                local_pos[1] += new_separation[1]
                pos = rot3d(local_pos, euler) + np.array(init_pos)
                self.AddBox(pos, new_box_euler, new_box_size)

    def AddPerlinHeighField(
            self,
            position=[1.0, 0.0, 0.0],  # position
            euler=[0.0, -0.0, 0.0],  # attitude
            size=[1.0, 1.0],  # width and length
            height_scale=0.2,  # max height
            negative_height=0.2,  # height in the negative direction of z axis
            image_width=128,  # height field image size
            img_height=128,
            smooth=100.0,  # smooth scale
            perlin_octaves=6,  # perlin noise parameter
            perlin_persistence=0.5,
            perlin_lacunarity=2.0,
            output_hfield_image="height_field.png"):

        # Generating height field based on perlin noise
        terrain_image = np.zeros((img_height, image_width), dtype=np.uint8)
        for y in range(image_width):
            for x in range(image_width):
                # Perlin noise
                noise_value = noise.pnoise2(x / smooth,
                                            y / smooth,
                                            octaves=perlin_octaves,
                                            persistence=perlin_persistence,
                                            lacunarity=perlin_lacunarity)
                terrain_image[y, x] = int((noise_value + 1) / 2 * 255)

        cv2.imwrite("../unitree_robots/" + ROBOT + "/" + output_hfield_image,
                    terrain_image)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "perlin_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = "../" + output_hfield_image

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "perlin_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddHeighFieldFromImage(
            self,
            position=[1.0, 0.0, 0.0],  # position
            euler=[0.0, -0.0, 0.0],  # attitude
            size=[2.0, 1.6],  # width and length
            height_scale=0.02,  # max height
            negative_height=0.1,  # height in the negative direction of z axis
            input_img=None,
            output_hfield_image="height_field.png",
            image_scale=[1.0, 1.0],  # reduce image resolution
            invert_gray=False):

        input_image = cv2.imread(input_img)  # 替换为你的图像文件路径

        width = int(input_image.shape[1] * image_scale[0])
        height = int(input_image.shape[0] * image_scale[1])
        resized_image = cv2.resize(input_image, (width, height),
                                   interpolation=cv2.INTER_AREA)
        terrain_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        if invert_gray:
            terrain_image = 255 - position
        cv2.imwrite("../unitree_robots/" + ROBOT + "/" + output_hfield_image,
                    terrain_image)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "image_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = "../" + output_hfield_image

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "image_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def Save(self):
        self.scene.write(OUTPUT_SCENE_PATH)


def create_multi_stair_with_walls(tg, pos, stair_width, stair_height, stair_nums, stair_length, plane_width, stairwell_length):
    plane_length = stair_length * 2 + stairwell_length
    plane_height = 0.1

    stairs_total_length = stair_nums * stair_width
    stairs_total_height = stair_nums * stair_height

    stair_init_pos = [pos[0] + plane_width / 2 - stair_width / 2, pos[1] + stairwell_length / 2 + stair_length / 2, pos[2]]
    tg.AddStairs(init_pos=stair_init_pos, yaw=0.0, width=stair_width, height=stair_height, length=stair_length, stair_nums=stair_nums)

    center_plane_pos = [stair_init_pos[0] + stairs_total_length + plane_width / 2 + stair_width / 2, pos[1], stair_init_pos[2] + stairs_total_height - plane_height / 2]

    tg.AddBox(position=center_plane_pos, euler=[0, 0, 0.0], size=[plane_width, plane_length, plane_height])

    second_stair_init_pos = [center_plane_pos[0] - plane_width / 2 + stair_width / 2, pos[1] - (stairwell_length / 2 + stair_length / 2), center_plane_pos[2] + plane_height / 2]

    tg.AddStairs(init_pos=second_stair_init_pos, yaw=np.pi, width=stair_width, height=stair_height, length=stair_length, stair_nums=stair_nums)

    last_plane_pos = [pos[0], pos[1], pos[2] + stairs_total_height * 2 - plane_height / 2]
    tg.AddBox(position=last_plane_pos, euler=[0, 0, 0.0], size=[plane_width, plane_length, plane_height])

    # wall
    wall_thickness = 0.05
    
    side_wall_length = stairs_total_length + plane_width * 2
    side_wall_height = stairs_total_height * 3
    side_wall_x = pos[0] + plane_width / 2 + stairs_total_length / 2
    side_wall_y_offset = plane_length / 2 + wall_thickness / 2
    
    stright_wall_height = stairs_total_height * 3
    stright_wall_length = plane_length

    left_side_wall_pos = [side_wall_x, pos[1] + side_wall_y_offset, pos[2] + side_wall_height / 2]
    tg.AddBox(position=left_side_wall_pos, euler=[0, 0, 0.0], size=[side_wall_length, wall_thickness, side_wall_height])
    right_side_wall_pos = [side_wall_x, pos[1] - side_wall_y_offset, pos[2] + side_wall_height / 2]
    tg.AddBox(position=right_side_wall_pos, euler=[0, 0, 0.0], size=[side_wall_length, wall_thickness, side_wall_height])

    forward_wall_pos = [center_plane_pos[0] + plane_width / 2 + wall_thickness / 2,  pos[1], pos[2] + stright_wall_height / 2]
    
    tg.AddBox(position=forward_wall_pos, euler=[0, 0, 0.0], size=[wall_thickness, stright_wall_length, stright_wall_height])
    
    # back_wall_pos = [pos[0] - plane_width / 2 - wall_thickness / 2, pos[1], pos[2] + stairs_total_height * 2 + stairs_total_height / 2]
    # tg.AddBox(position=back_wall_pos, euler=[0, 0, 0.0], size=[wall_thickness, stright_wall_length, stairs_total_height])
    

def create_bridge_with_walls(tg, pos, stair_width, stair_height, stair_nums, stair_length, flat_plane_length):

    # Stairs
    stair_init_pos = pos
    
    tg.AddStairs(init_pos=stair_init_pos, yaw=0.0,
                 width=stair_width, height=stair_height,
                 length=stair_length, stair_nums=stair_nums)
    
    # Calculate position after stairs
    stairs_total_length = stair_nums * stair_width
    plane_thickness = 0.1
    
    flat_plane_pos = [stair_init_pos[0] + stairs_total_length + stair_width / 2, stair_init_pos[1], stair_height * stair_nums - plane_thickness / 2]
    
    # Flat plane (thin box)
    flat_plane_pos[0] = flat_plane_pos[0] + flat_plane_length / 2
    tg.AddBox(position=flat_plane_pos,
              euler=[0, 0, 0.0],
              size=[flat_plane_length, stair_length, plane_thickness])
    
    # Calculate position after flat plane
    after_stair_pos = [flat_plane_pos[0] + flat_plane_length / 2 + stairs_total_length + stair_width / 2, stair_init_pos[1], 0]
    
    yaw_deg = 180
    yaw_rad = np.deg2rad(yaw_deg)
    
    tg.AddStairs(init_pos=after_stair_pos, yaw=yaw_rad,
                 width=stair_width, height=stair_height,
                 length=stair_length, stair_nums=stair_nums)
    
    # Add walls on both sides covering the entire structure (stairs + flat plane + stairs)
    wall_thickness = 0.05
    wall_height = 4
    total_length = stairs_total_length * 2 + flat_plane_length
    
    # Calculate center position of the entire structure
    center_x = flat_plane_pos[0]
    center_y = flat_plane_pos[1]
    center_z = wall_height / 2
    
    # Left wall (negative Y side)
    left_wall_pos = [center_x,
                     center_y - stair_length / 2 - wall_thickness / 2,
                     center_z]
    tg.AddBox(position=left_wall_pos,
              euler=[0, 0, 0.0],
              size=[total_length, wall_thickness, wall_height])
    
    # Right wall (positive Y side)
    right_wall_pos = [center_x,
                      center_y + stair_length / 2 + wall_thickness / 2,
                      center_z]
    tg.AddBox(position=right_wall_pos,
              euler=[0, 0, 0.0],
              size=[total_length, wall_thickness, wall_height])
    # ------------------
    
if __name__ == "__main__":
    tg = TerrainGenerator()

    create_bridge_with_walls(tg, pos=[1.0, 2.0, 0.0], stair_width=0.30, stair_height=0.16, stair_nums=13, stair_length=1.5, flat_plane_length=2.0)

    create_multi_stair_with_walls(tg, pos=[-2.0, 5.0, 0.0], stair_width=0.30, stair_height=0.16, stair_nums=13, stair_length=1.5, plane_width=2.0, stairwell_length=0.2)

    tg.Save()