import os
import xml.etree.ElementTree as ET

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('map_vla_description')
    urdf = xacro.process_file(os.path.join(share, 'urdf', 'map_vla.urdf.xacro')).toxml()

    # joint_state_publisher is not installed: publish zeros for every non-fixed joint
    # (mimic joints are included too; robot_state_publisher applies them directly)
    joints = [j.get('name') for j in ET.fromstring(urdf).iter('joint')
              if j.get('type') != 'fixed']

    return LaunchDescription([
        # placeholders until SLAM (map->odom) and wheel odometry (odom->base_footprint) publish these
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='map_to_odom', arguments=['--frame-id', 'map', '--child-frame-id', 'odom']),
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='odom_to_base_footprint',
             arguments=['--frame-id', 'odom', '--child-frame-id', 'base_footprint']),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': urdf}]),
        Node(package='map_vla_description', executable='zero_joint_states.py',
             parameters=[{'joints': joints}]),
        Node(package='rviz2', executable='rviz2',
             arguments=['-d', os.path.join(share, 'rviz', 'map_vla.rviz')]),
    ])
