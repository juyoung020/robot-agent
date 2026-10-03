#!/usr/bin/python3
"""Publish zero joint states (stamped) for the given joint names until a real source exists."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ZeroJointStates(Node):
    def __init__(self):
        super().__init__('zero_joint_states')
        self.names = self.declare_parameter('joints', [''] ).value
        self.pub = self.create_publisher(JointState, 'joint_states', 10)
        self.create_timer(0.05, self.tick)

    def tick(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.names)
        msg.position = [0.0] * len(msg.name)
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(ZeroJointStates())


if __name__ == '__main__':
    main()
