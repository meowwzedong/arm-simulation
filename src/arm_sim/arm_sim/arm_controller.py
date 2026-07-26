#!/usr/bin/env python3
"""
STEP keyboard control for joint1 + joint2 via arm_controller
(JointTrajectoryController).

Each TAP of a key moves one joint by a fixed angle. Both joint targets are
always published together -- see note below about allow_partial_joints_goal.

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
STEP = 0.10        # rad per tap (~5.7 deg)
MOVE_TIME = 0.3    # s to reach each new target

# Order here MUST match the 'joints:' list in controllers.yaml
JOINTS = [
    # name,      lower,  upper,  key_neg, key_pos
    ('joint1',   -1.57,   1.57,   'q',     'w'),
    ('joint2',   -1.57,   1.57,   'e',     'r'),
    ('joint3',   -1.57,   1.57,   't',     'y'),
    ('gripper_joint',   -1.57,   1.57,   'u',     'i'),
]

JOINT_NAMES = [j[0] for j in JOINTS]
KEYMAP = {}                      # key -> (joint_index, direction)
for i, (_, _, _, kn, kp) in enumerate(JOINTS):
    KEYMAP[kn] = (i, -1)
    KEYMAP[kp] = (i, +1)

HELP = """
============== joint1 + joint2 STEP control ==============
  q / w : joint1  negative / positive
  e / r : joint2  negative / positive
  t / y : joint3  negative / positive
  u / i : gripper_joint  negative / positive
  x     : all the joints to zero
----------------------------------------------------------
  Each tap moves that joint by {step} rad.
==========================================================
""".format(step=STEP)


class KeyboardStep(Node):
    def __init__(self):
        super().__init__('keyboard_step')
        self.pub = self.create_publisher(JointTrajectory, TOPIC, 10)
        self.targets = [0.0] * len(JOINTS)

    def clamp(self, index, value):
        _, lower, upper, _, _ = JOINTS[index]
        return max(lower, min(upper, value))

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
        return '  '.join('%s=%+.2f' % (JOINTS[i][0], self.targets[i])
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
            elif key in KEYMAP:
                index, direction = KEYMAP[key]
                node.targets[index] = node.clamp(
                    index, node.targets[index] + direction * STEP)
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