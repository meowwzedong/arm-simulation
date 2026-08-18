#!/usr/bin/env python3
"""
STEP keyboard control for the full arm + gripper via arm_controller
(JointTrajectoryController).

Each TAP of a key moves one joint by a fixed step. ALL controlled joints are
published together in every message, because JointTrajectoryController rejects
partial goals by default (allow_partial_joints_goal: false).

BOTH gripper jaws are commanded from the SAME keypair (o/p): the o/p keys set
gripper_right_joint and gripper_left_joint together, so no URDF <mimic> tag or
Gazebo mimic plugin is needed. Both jaws must therefore be ordinary commanded
joints (position command interface in the URDF, listed in controllers.yaml).

Note the jaws are PRISMATIC (metres, 0 -> 0.015), so they use a much smaller
per-tap step than the revolute joints.

Run in its OWN terminal (it needs the terminal for keyboard input),
while the simulation launch runs in another terminal.
"""

import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# ---- Settings you can tune ----
TOPIC = '/arm_controller/joint_trajectory'
MOVE_TIME = 0.3    # s to reach each new target

# Relationship between the two jaws: left_target = GRIP_SIGN * right_target
# If the jaws move APART when you expect them to close, flip this to -1.0.
GRIP_SIGN = 1.0

# Order here MUST match the 'joints:' list in controllers.yaml.
# tuple: name, lower, upper, step
JOINTS = [
    ('joint1',             -1.57,  1.57,  0.10),
    ('joint2',             -1.57,  1.57,  0.10),
    ('joint3',             -1.57,  1.57,  0.10),
    ('joint4',             -1.57,  1.57,  0.10),
    ('joint5',             -1.57,  1.57,  0.10),
    ('right_tooth_joint',  0.0, 0.015, 0.003),
    ('left_tooth_joint',   0.0, 0.015, 0.003),
]

JOINT_NAMES = [j[0] for j in JOINTS]
IDX = {name: i for i, (name, *_) in enumerate(JOINTS)}
GRIP_R = IDX['right_tooth_joint']
GRIP_L = IDX['left_tooth_joint']

# Single-joint keys: key -> (joint_index, direction)
KEYMAP = {
    'q': (IDX['joint1'], -1), 'w': (IDX['joint1'], +1),
    'e': (IDX['joint2'], -1), 'r': (IDX['joint2'], +1),
    't': (IDX['joint3'], -1), 'y': (IDX['joint3'], +1),
    'u': (IDX['joint4'], -1), 'i': (IDX['joint4'], +1),
    'a': (IDX['joint5'], -1), 's': (IDX['joint5'], +1),
}

# Gripper keys: one key drives BOTH jaws.  key -> direction
GRIP_KEYS = {'o': -1, 'p': +1}

HELP = """
============== arm + gripper STEP control ==============
  q / w : joint1  negative / positive
  e / r : joint2  negative / positive
  t / y : joint3  negative / positive
  u / i : joint4  negative / positive   (wrist / gripper body)
  o / p : BOTH gripper jaws  close / open  (0 -> 0.015 m)
  x     : send all joints to zero
--------------------------------------------------------
  Revolute joints move 0.10 rad per tap; jaws move 0.003 m per tap.
  (If o/p feel reversed, swap the two keys or flip GRIP_SIGN above.)
========================================================
"""


class KeyboardStep(Node):
    def __init__(self):
        super().__init__('keyboard_step')
        self.pub = self.create_publisher(JointTrajectory, TOPIC, 10)
        self.targets = [0.0] * len(JOINTS)
        while self.pub.get_subscription_count() == 0 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('controller connected')

    def clamp(self, index, value):
        _, lower, upper, _ = JOINTS[index]
        return max(lower, min(upper, value))

    def move_joint(self, index, direction):
        step = JOINTS[index][3]
        self.targets[index] = self.clamp(
            index, self.targets[index] + direction * step)

    def move_gripper(self, direction):
        """Step both jaws from a single key."""
        step = JOINTS[GRIP_R][3]
        new_right = self.clamp(GRIP_R, self.targets[GRIP_R] + direction * step)
        self.targets[GRIP_R] = new_right
        self.targets[GRIP_L] = self.clamp(GRIP_L, GRIP_SIGN * new_right)

    def send(self, duration_s):
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(t) for t in self.targets]
        sec = int(duration_s)
        pt.time_from_start = Duration(sec=sec,
                                      nanosec=int((duration_s - sec) * 1e9))
        msg.points = [pt]
        self.pub.publish(msg)

    def status(self):
        return '  '.join('%s=%+.3f' % (JOINTS[i][0], self.targets[i])
                         for i in range(len(JOINTS)))


def get_key(timeout):
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.read(1)
    return ''


def main():
    rclpy.init()
    node = KeyboardStep()

    print(HELP)
    settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())

    try:
        while rclpy.ok():
            key = get_key(0.1)

            if key == '\x03':                      # Ctrl-C
                break
            elif key == 'x':
                node.targets = [0.0] * len(JOINTS)
                node.send(0.5)
                sys.stdout.write('\r\n[targets] %s\r\n' % node.status())
            elif key in GRIP_KEYS:
                node.move_gripper(GRIP_KEYS[key])
                node.send(MOVE_TIME)
                sys.stdout.write('\r\n[targets] %s\r\n' % node.status())
            elif key in KEYMAP:
                index, direction = KEYMAP[key]
                node.move_joint(index, direction)
                node.send(MOVE_TIME)
                sys.stdout.write('\r\n[targets] %s\r\n' % node.status())

            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()
        print('\nStopped.')


if __name__ == '__main__':
    main()