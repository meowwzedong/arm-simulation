#!/usr/bin/env python3
"""
keyboard_ik.py

Jacobian-based inverse kinematics controller for arm_sim.

You type an X Y Z target (in the base_link frame) and this node drives all five
revolute joints (joint1..joint5) so the tip of the LEFT gripper tooth reaches it.
It solves the redundant 5-DOF chain numerically with a damped-least-squares
Jacobian, seeding from a random valid joint vector each solve.

The redundancy is resolved by an ORIENTATION constraint: the gripper's approach
axis (the tooth-frame +x axis, which runs out through the teeth) is driven to point
straight DOWN (-Z in base_link). This makes the wrist hook over the top of the
target so the upper links trail behind instead of crossing the gripper's path --
i.e. the arm approaches from above, teeth leading, ready to pick.

The gripper prismatic joints are held open (0.0). Every trajectory goal includes
ALL seven declared joints, because JointTrajectoryController rejects partial goals.

Run:
    ros2 run arm_sim keyboard_ik
    # then type e.g.:   0.05 -0.02 0.22
"""

import sys
import random
import numpy as np

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration


# ---------------------------------------------------------------------------
# Configuration -- change these two strings if your setup differs
# ---------------------------------------------------------------------------
CONTROLLER_TOPIC = "/arm_controller/joint_trajectory"

# Order the controller expects. joint1..joint5 are solved by IK; the two tooth
# joints are appended (held open) so every goal is a full 7-joint message.
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5"]
GRIPPER_JOINTS = ["left_tooth_joint", "right_tooth_joint"]
ALL_JOINTS = ARM_JOINTS + GRIPPER_JOINTS
GRIPPER_OPEN = [0.0, 0.0]

# Revolute joint limits (rad). These MUST match your URDF -- the solver will
# happily produce angles your URDF forbids, and Gazebo will then clamp them and
# miss the target.
#
# NOTE: with the original +/-1.57 on every joint, only ~3 in 10 targets can be
# reached with the teeth pointing down -- the pitch joints cannot bend far enough
# to hook the wrist over the target, so the upper links end up in the way.
# Widening joints 2-5 to +/-3.14 raises that to ~9 in 10. Widen them in
# arm.urdf.xacro (and the ros2_control command_interface min/max) to match.
JOINT_LOWER = np.array([-3.14, -3.14, -3.14, -3.14, -3.14])
JOINT_UPPER = np.array([ 3.14,  3.14,  3.14,  3.14,  3.14])

MOVE_TIME_SEC = 2.5   # how long the arm takes to reach the commanded pose

# Desired approach direction for the gripper teeth, in base_link frame.
# [0, 0, -1] = point straight down (approach the target from above).
# The tooth-frame +x axis is the approach axis (it runs out through the teeth).
APPROACH_DIR = np.array([0.0, 0.0, -1.0])
# How hard to enforce the approach direction vs. hitting the position exactly.
# This scales the orientation error into the same units as the position error
# (metres per radian), so ~0.08 means "1 rad of tilt costs as much as 8 cm of
# position error". Raise it if the arm still creeps into the gripper's way.
ORIENT_WEIGHT = 0.08

# How far off straight-down the teeth may point and still count as "good".
# 12 deg is tight enough to keep the upper links clear. This is a preference,
# not a hard requirement -- position is always satisfied first.
ORIENT_TOL_DEG = 12.0

# How many random seeds to try per target. More attempts -> better chance of a
# steeply-downward pose. Most targets solve on the first few.
SOLVE_ATTEMPTS = 40

# Give up early if this many consecutive fresh seeds fail to improve on the best
# result so far and the position has never been hit -- the target is unreachable.
STALL_LIMIT = 5


# ---------------------------------------------------------------------------
# Kinematics -- built directly from arm.urdf.xacro
# ---------------------------------------------------------------------------
def rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_T(xyz, rpy):
    M = np.eye(4)
    M[:3, :3] = rpy_to_R(*rpy)
    M[:3, 3] = xyz
    return M


def axis_rot(axis, theta):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = np.cos(theta), np.sin(theta)
    C = 1.0 - c
    R = np.array([
        [c + x * x * C,   x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,   y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    return M


# Fixed parent->child transforms and rotation axes for joint1..joint5.
# All revolute axes are (0, 0, -1) in their own frames (per URDF).
_JOINT_FIXED = [
    (make_T([0, 0, 0.0475],               [0, 0, 0]),      np.array([0, 0, -1.0])),  # joint1
    (make_T([0, 0, 0.0135],               [0, 1.5708, 0]), np.array([0, 0, -1.0])),  # joint2
    (make_T([-0.060997, -0.00064835, 0],  [0, 0, 0]),      np.array([0, 0, -1.0])),  # joint3
    (make_T([-0.060997, -0.00064835, 0],  [0, 0, 0]),      np.array([0, 0, -1.0])),  # joint4
    (make_T([-0.060128, -0.00063912, 0],  [0, 0, 0]),      np.array([0, 0, -1.0])),  # joint5
]

# link5 -> left_tooth fixed transform, prismatic held at 0 (gripper open).
_TOOTH_FIXED = make_T([-0.023867, -0.0002705, -0.013428], [0, 0, 0])
# Point of the tooth tip expressed in the left_tooth frame (along the tooth body).
_TIP_LOCAL = np.array([-0.0151884, -0.00016144, 0.0, 1.0])


def _fk_frame(q):
    """Full 4x4 transform of the left_tooth frame for joint vector q."""
    Tc = np.eye(4)
    for (Tj, ax), qi in zip(_JOINT_FIXED, q):
        Tc = Tc @ Tj @ axis_rot(ax, qi)
    return Tc @ _TOOTH_FIXED


def fk_tip(q):
    """Forward kinematics: 5 revolute angles -> left tooth tip position (3,)."""
    return (_fk_frame(q) @ _TIP_LOCAL)[:3]


# Unit direction, in the tooth's own frame, pointing from the tooth frame origin
# out through the physical tip. Derived from _TIP_LOCAL rather than hardcoded, so
# it stays correct if the tip offset is ever remeasured. The tip sits at NEGATIVE
# x in the tooth frame, so this comes out as -x -- getting this sign backwards
# aims the back of the wrist at the target and leaves the teeth pointing away.
_APPROACH_LOCAL = _TIP_LOCAL[:3] / np.linalg.norm(_TIP_LOCAL[:3])


def fk_approach(q):
    """Unit vector of the gripper approach axis (out through the teeth) in base_link."""
    return _fk_frame(q)[:3, :3] @ _APPROACH_LOCAL


def fk_full(q):
    """Return (tip position (3,), approach axis (3,))."""
    T = _fk_frame(q)
    return (T @ _TIP_LOCAL)[:3], T[:3, :3] @ _APPROACH_LOCAL


def _align_error(a, b):
    """
    Rotation vector that rotates unit vector `a` onto unit vector `b`.

    Direction = rotation axis, magnitude = angle (0..pi). Using the full angle
    rather than a bare cross product matters: cross(a, b) vanishes BOTH when the
    vectors are aligned and when they are exactly opposed, so a cross-product-only
    error has a stationary point at 180 deg and the solver gets trapped there with
    the teeth pointing straight up. This version returns a pi-magnitude kick in an
    arbitrary perpendicular direction instead, which escapes that trap.
    """
    c = np.cross(a, b)
    s = np.linalg.norm(c)
    d = float(np.dot(a, b))
    if s < 1e-9:
        if d > 0.0:
            return np.zeros(3)               # already aligned
        # exactly opposed: rotate pi about any axis perpendicular to a
        tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, tmp)
        return axis / np.linalg.norm(axis) * np.pi
    return (c / s) * np.arctan2(s, d)


def _residual(q, target):
    """
    6-element task residual, all of which we want driven to zero.

      rows 0-2 : position error (metres)  = target - tip
      rows 3-5 : orientation error        = w * (approach_axis  x  desired_dir)

    The cross product is the key bit: it yields a rotation AXIS whose direction
    tells the solver which way to twist the wrist, and whose magnitude is
    sin(angle-off-target). A plain dot product only gives the size of the error,
    not its direction, so the solver cannot act on it.
    """
    pos, appr = fk_full(q)
    e_pos = target - pos
    e_orient = ORIENT_WEIGHT * _align_error(appr, APPROACH_DIR)
    return np.concatenate([e_pos, e_orient])


def pos_jacobian(q, eps=1e-6):
    """Numerical position Jacobian d(tip)/dq, shape (3, 5)."""
    p0 = fk_tip(q)
    J = np.zeros((3, 5))
    for i in range(5):
        dq = q.copy()
        dq[i] += eps
        J[:, i] = (fk_tip(dq) - p0) / eps
    return J


def orient_jacobian(q, eps=1e-6):
    """Numerical Jacobian of the approach axis d(appr)/dq, shape (3, 5)."""
    a0 = fk_approach(q)
    J = np.zeros((3, 5))
    for i in range(5):
        dq = q.copy()
        dq[i] += eps
        J[:, i] = (fk_approach(dq) - a0) / eps
    return J


def _combined_jacobian(q, target, eps=1e-6):
    """Numerical Jacobian d(residual)/dq for the 6-row combined task, shape (6, 5)."""
    r0 = _residual(q, target)
    J = np.zeros((6, 5))
    for i in range(5):
        dq = q.copy()
        dq[i] += eps
        J[:, i] = (_residual(dq, target) - r0) / eps
    return J


def _solve_combined(target, seed, iters=300, damping=0.02):
    """
    STAGE 1: solve position and orientation together as one weighted task.
    This reliably finds a pose with the teeth pointing down, though the position
    may be slightly off if the target is near the edge of the teeth-down region.
    Its job is to land in the right basin for stage 2.
    """
    q = np.clip(np.asarray(seed, dtype=float), JOINT_LOWER, JOINT_UPPER)
    target = np.asarray(target, dtype=float)
    for _ in range(iters):
        r = _residual(q, target)
        J = _combined_jacobian(q, target)
        JJt = J @ J.T + (damping ** 2) * np.eye(6)
        dq = -J.T @ np.linalg.solve(JJt, r)
        q_new = np.clip(q + np.clip(dq, -0.2, 0.2), JOINT_LOWER, JOINT_UPPER)
        if np.linalg.norm(q_new - q) < 1e-10:
            return q_new
        q = q_new
    return q


def _solve_nullspace(target, seed, iters=500, tol=1e-4, damping=0.02):
    """
    Task-priority (null-space) Jacobian IK.

    PRIMARY task  : tip position -- solved exactly, always wins.
    SECONDARY task: point the teeth along APPROACH_DIR (down) -- pursued only
                    inside the null space of the position task, so chasing it can
                    never pull the tip off the target.

    With 5 joints and 3 position constraints the null space is 2-dimensional,
    which is exactly the number of DOF needed to aim a direction. So the arm
    reaches the target and then uses its leftover freedom to swing the upper links
    up and out of the gripper's way, as far as the joint limits permit.

    Returns (q, reached_bool, pos_error_metres, orient_cos).
    reached_bool means the POSITION was hit; orient_cos reports how close to
    straight-down the teeth ended up (1.0 = perfect).
    """
    q = np.clip(np.asarray(seed, dtype=float), JOINT_LOWER, JOINT_UPPER)
    target = np.asarray(target, dtype=float)

    for _ in range(iters):
        e_pos = target - fk_tip(q)

        Jp = pos_jacobian(q)
        # damped pseudo-inverse of the position Jacobian
        JJt = Jp @ Jp.T + (damping ** 2) * np.eye(3)
        Jp_pinv = Jp.T @ np.linalg.inv(JJt)

        # primary step: drive the tip onto the target
        dq = Jp_pinv @ e_pos

        # secondary step: rotate the approach axis toward APPROACH_DIR, but only
        # in directions that do not move the tip (null-space projection)
        appr = fk_approach(q)
        e_or = _align_error(appr, APPROACH_DIR)      # axis-angle tilt error
        Jo = orient_jacobian(q)
        dq_sec = ORIENT_WEIGHT * (Jo.T @ e_or)
        N = np.eye(5) - Jp_pinv @ Jp                 # null-space projector
        dq = dq + N @ dq_sec

        q_new = np.clip(q + np.clip(dq, -0.2, 0.2), JOINT_LOWER, JOINT_UPPER)
        step_size = np.linalg.norm(q_new - q)
        q = q_new

        # Stop once the joints have stopped moving meaningfully. If the position
        # is already met, a stalled step means the null space has no more room to
        # improve the aim, so further iterations are wasted.
        if step_size < 1e-7:
            break
        if step_size < 1e-5 and np.linalg.norm(target - fk_tip(q)) < tol:
            break

    pos_err = float(np.linalg.norm(target - fk_tip(q)))
    orient_cos = float(np.dot(fk_approach(q), APPROACH_DIR))
    return q, bool(pos_err < tol), pos_err, orient_cos


def solve_ik(target, seed, tol=1e-4):
    """
    Two-stage hybrid solve from a single seed.

    Stage 1 gets the teeth pointing down; stage 2 pulls the tip exactly onto the
    target using only null-space motion, so the downward aim survives.

    Returns (q, position_reached, pos_error_metres, orient_cos).
    """
    warm = _solve_combined(target, seed)
    return _solve_nullspace(target, warm, tol=tol)


# Furthest the tooth tip can ever get from the base origin: every fixed offset in
# the chain laid out in a straight line. Used as an instant reachability check.
MAX_REACH = (0.0475 + 0.0135
             + 0.060997 + 0.060997 + 0.060128
             + 0.023867 + 0.0151884) * 1.02   # +2% tolerance


def random_seed():
    return np.random.uniform(JOINT_LOWER, JOINT_UPPER)


def solve_best(target, attempts=SOLVE_ATTEMPTS, tol=1e-4):
    """
    Run the hybrid solve from many random seeds and return the best result.

    Selection rule: any solution that HITS the target beats one that does not;
    among those that hit it, the one whose teeth point most steeply downward wins.
    This is what keeps the upper links clear of the gripper.

    Returns (q, position_reached, pos_error_metres, orient_cos).
    """
    target = np.asarray(target, dtype=float)

    # Cheap sanity check: the tip can never be further from the base origin than
    # the arm is long, so reject wild targets instantly instead of grinding
    # through every random seed.
    if np.linalg.norm(target) > MAX_REACH:
        q = np.zeros(5)
        return (q, False, float(np.linalg.norm(target - fk_tip(q))),
                float(np.dot(fk_approach(q), APPROACH_DIR)))

    best = None
    cos_tol = np.cos(np.radians(ORIENT_TOL_DEG))
    stalled = 0

    for _ in range(attempts):
        q, reached, err, cos = solve_ik(target, random_seed(), tol=tol)
        # rank: position-valid first, then most-downward, then least error
        score = (0 if err < tol else 1,
                 -cos if err < tol else err)
        if best is None or score < best[0]:
            best = (score, q, reached, err, cos)
            stalled = 0
        else:
            stalled += 1
        # early out once we have a target-hitting, properly-downward pose
        if err < tol and cos > cos_tol:
            break
        # give up early on hopeless targets: if many fresh seeds in a row all fail
        # to improve and none has ever hit the position, it is out of reach
        if stalled >= STALL_LIMIT and best[3] > tol:
            break

    _, q, reached, err, cos = best
    return q, reached, err, cos


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------
class KeyboardIK(Node):
    def __init__(self):
        super().__init__("keyboard_ik")
        self.pub = self.create_publisher(JointTrajectory, CONTROLLER_TOPIC, 10)
        # Optional: track current joint state so we can report where we are.
        self.current = None
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        self.get_logger().info(
            f"keyboard_ik ready. Publishing to {CONTROLLER_TOPIC}\n"
            f"Type a target as:  X Y Z   (metres, base_link frame). Ctrl-C to quit."
        )

    def _js_cb(self, msg):
        self.current = dict(zip(msg.name, msg.position))

    def send(self, q_arm):
        traj = JointTrajectory()
        traj.joint_names = ALL_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q_arm] + GRIPPER_OPEN
        pt.time_from_start = Duration(
            sec=int(MOVE_TIME_SEC),
            nanosec=int((MOVE_TIME_SEC % 1) * 1e9),
        )
        traj.points = [pt]
        self.pub.publish(traj)

    def handle_target(self, x, y, z):
        target = np.array([x, y, z], dtype=float)

        q, reached, err, cos = solve_best(target)
        achieved = fk_tip(q)
        tilt = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        joints_str = " ".join(f"{v:+.3f}" for v in q)

        if not reached:
            self.get_logger().warn(
                f"Target out of reach. Closest tip -> "
                f"[{achieved[0]:.4f} {achieved[1]:.4f} {achieved[2]:.4f}] "
                f"(off by {err*1000:.1f} mm). Sending closest pose."
            )
        elif tilt > ORIENT_TOL_DEG:
            self.get_logger().warn(
                f"Reached target, but teeth are {tilt:.1f} deg off straight-down, "
                f"so the arm may sit in the gripper's way. This target is outside "
                f"the teeth-down workspace -- widening joints 2-5 in the URDF helps.\n"
                f"  tip -> [{achieved[0]:.4f} {achieved[1]:.4f} {achieved[2]:.4f}]\n"
                f"  joints (rad): {joints_str}"
            )
        else:
            self.get_logger().info(
                f"Solved. tip -> [{achieved[0]:.4f} {achieved[1]:.4f} {achieved[2]:.4f}] "
                f"(err {err*1000:.2f} mm, teeth {tilt:.1f} deg off-down)\n"
                f"  joints (rad): {joints_str}"
            )

        self.send(q)


def main():
    rclpy.init()
    node = KeyboardIK()

    try:
        while rclpy.ok():
            # process any pending joint_state callbacks without blocking input
            rclpy.spin_once(node, timeout_sec=0.0)
            try:
                line = input("target X Y Z > ").strip()
            except EOFError:
                break
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                node.get_logger().warn("Enter exactly three numbers: X Y Z")
                continue
            try:
                x, y, z = (float(p) for p in parts)
            except ValueError:
                node.get_logger().warn("Could not parse numbers.")
                continue
            node.handle_target(x, y, z)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()