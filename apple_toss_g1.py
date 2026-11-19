"""
Unitree G1 apple toss in MuJoCo
===============================
The G1 (29-DoF body, Dex3 three-finger hands, from MuJoCo Menagerie) picks an apple
off a table with its right hand, turns it palm-up, tosses it in an arc, and catches it
in its left hand. The left hand re-plans the catch point every control tick from the
apple's measured position/velocity (ballistic prediction), then cushions the impact.
The waist turns toward the working hand and back.

All motion is physical: joints are the G1's position servos (menagerie gains and torque
limits), the grasp is held by finger friction, and the apple is released because the
hand brakes harder than gravity.

Usage
-----
  python apple_toss_g1.py                     # -> apple_toss_g1.mp4
  python apple_toss_g1.py --camera wide --slowmo 3
  python apple_toss_g1.py --trials 20         # robustness test, no video
  python apple_toss_g1.py --view              # live viewer (needs a display)
"""
import argparse
import os
import sys

if sys.platform.startswith("linux"):   # headless OpenGL on Linux; Windows/macOS use the default backend
    os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

from g1_scene import build_model, APPLE_R, TABLE_TOP

G = 9.81


# ----------------------------------------------------------------- small math helpers
def frame(x, y):
    """rotation matrix whose columns are the hand axes x, y, z=x×y (given in world)"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    return np.column_stack([x, y, np.cross(x, y)])


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def slerp(q0, q1, s):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if q0 @ q1 < 0:
        q1 = -q1
    d = np.clip(q0 @ q1, -1, 1)
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)
    return q / np.linalg.norm(q)


def min_jerk(s):
    """smooth 0->1, zero velocity and acceleration at both ends"""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def predict_landing(p, v, z_catch):
    """ballistic: where the apple crosses z_catch on the way down"""
    a, b, c = -0.5 * G, v[2], p[2] - z_catch
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t = (-b - np.sqrt(disc)) / (2 * a)
    return p + v * t + np.array([0, 0, -0.5 * G * t * t])


# hand orientations.  Hand frame: x = fingers, palm normal = +y (right) / -y (left)
FWD, UP, DOWN, LEFT, RIGHT = np.eye(3)[0], np.eye(3)[2], -np.eye(3)[2], np.eye(3)[1], -np.eye(3)[1]
R_PALM_DOWN = mat2quat(frame(FWD, DOWN))
R_PALM_IN = mat2quat(frame(FWD, LEFT))     # right palm facing the body midline
R_PALM_UP = mat2quat(frame(FWD, UP))
L_PALM_IN = mat2quat(frame(FWD, LEFT))     # left palm (-y) facing the midline
L_PALM_UP = mat2quat(frame(FWD, DOWN))     # left palm (-y) facing up

# Dex3 finger poses for the RIGHT hand, joint order:
#   thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
HAND = {
    "open":  [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],    # thumb swung back, fingers straight
    "grasp": [0.0, -0.5, -0.9, 1.4, 1.2, 1.4, 1.2],  # thumb behind, two fingers in front
    "cup":   [0.0, -0.1, -0.4, 1.1, 0.5, 1.1, 0.5],  # palm-up cradle
    "toss":  [0.0, 0.0, -0.2, 0.45, 0.15, 0.45, 0.15], # flatter: nothing above the apple
    "catch": [0.0, 0.3, 0.0, 0.45, 0.4, 0.45, 0.4],  # wide cup waiting for the apple
}
FINGERS = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
MIRROR = np.array([1, -1, -1, -1, -1, -1, -1])         # left hand = mirrored joint signs


# ----------------------------------------------------------------- one arm + hand
class Arm:
    JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"]

    def __init__(self, task, side):
        m = task.model
        self.task, self.side = task, side
        name = "right" if side == "r" else "left"
        jids = [m.joint(f"{name}_{j}_joint").id for j in self.JOINTS]
        self.qadr, self.dadr = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
        self.lo, self.hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
        self.act = np.array([m.actuator(f"{name}_{j}_joint").id for j in self.JOINTS])
        self.kp = m.actuator_gainprm[self.act, 0]
        self.kv = -m.actuator_biasprm[self.act, 2]
        self.fing = np.array([m.actuator(f"{name}_hand_{f}_joint").id for f in FINGERS])
        self.mirror = MIRROR if side == "l" else np.ones(7)
        self.site = m.site(f"{side}_grip").id
        self.hand_bodies = {m.body(f"{name}_wrist_yaw_link").id} | \
            {m.body(f"{name}_hand_{f[:-2]}_{f[-1]}_link").id for f in FINGERS}
        self.geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in self.hand_bodies and m.geom_contype[g]}
        # natural posture for the null space: elbow bent, wrist neutral
        self.rest = np.array([0.2, -0.2 if side == "r" else 0.2, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.jacp, self.jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def reset(self):
        self.q_des = self.task.data.qpos[self.qadr].copy()
        self.qd_des = np.zeros(7)

    def grip_pos(self):
        return self.task.data.site_xpos[self.site].copy()

    def grip_quat(self):
        return mat2quat(self.task.data.site_xmat[self.site].reshape(3, 3))

    def ik(self, p_t, q_t, iters=40, lam=0.03, w_rot=0.5):
        """6-D damped-least-squares IK on the 7-DoF arm, warm-started from the last
        solution; the 1-D null space pulls toward a relaxed posture."""
        m, s = self.task.model, self.task.scratch
        s.qpos[:] = self.task.data.qpos
        q = self.q_des.copy()
        err_q, cur_q = np.zeros(3), np.zeros(4)
        for _ in range(iters):
            s.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, s)
            mujoco.mj_comPos(m, s)
            mujoco.mj_jacSite(m, s, self.jacp, self.jacr, self.site)
            Rm = s.site_xmat[self.site].reshape(3, 3)
            mujoco.mju_mat2Quat(cur_q, Rm.ravel())
            mujoco.mju_subQuat(err_q, q_t, cur_q)
            e = np.concatenate([p_t - s.site_xpos[self.site], w_rot * (Rm @ err_q)])
            J = np.vstack([self.jacp[:, self.dadr], w_rot * self.jacr[:, self.dadr]])
            JJt = J @ J.T + lam**2 * np.eye(6)
            Jpinv = J.T @ np.linalg.inv(JJt)
            dq = Jpinv @ e + 0.1 * (np.eye(7) - Jpinv @ J) @ (self.rest - q)
            q = np.clip(q + dq, self.lo, self.hi)
            if np.linalg.norm(e) < 2e-5:
                break
        return q

    def command(self, p_t, q_t, hand, dt):
        q_new = self.ik(p_t, q_t)
        # velocity feed-forward, low-pass filtered so IK noise can't excite the arm
        qd = (q_new - self.q_des) / dt
        self.qd_des = 0.7 * self.qd_des + 0.3 * qd
        self.q_des = q_new
        d = self.task.data
        d.ctrl[self.act] = self.q_des + self.kv / self.kp * self.qd_des
        d.ctrl[self.fing] = self.mirror * np.asarray(hand)


def blend(a, b, s):
    return (1 - s) * np.asarray(a) + s * np.asarray(b)


# ----------------------------------------------------------------- task parameters
APPLE0_Z = TABLE_TOP + APPLE_R            # apple centre when resting on the table
WIND = np.array([0.26, -0.12, 0.93])      # right palm-up hold before the throw
V_REL = np.array([0.0, 0.35, 1.60])       # right-hand velocity at release (m/s)
T_ACC, T_DEC = 0.25, 0.10                 # throw: accelerate, then brake (decel > g)
R_PARK = np.array([0.16, -0.22, 0.86])    # right hand gets out of the way
L_READY = np.array([0.26, 0.15, 0.95])    # left hand waits here
CATCH_Z = 0.95                            # apple-centre height where the catch happens
CUSHION = 0.08                            # left hand gives way this much while catching
T_CUSHION = 0.25                          # ...over this long (ease-out: fastest at contact)
L_SHOW = np.array([0.24, 0.08, 1.00])     # left hand presents the apple at the end
L_CATCH_Q = L_PALM_UP                     # flat palm: no surface to bounce the apple sideways


class AppleToss:
    CTRL_EVERY = 2  # controller at 250 Hz, physics at 500 Hz

    def __init__(self):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)   # IK works on a copy, never the live sim
        m = self.model
        self.dt_ctrl = m.opt.timestep * self.CTRL_EVERY
        self.R, self.L = Arm(self, "r"), Arm(self, "l")
        self.apple = m.body("apple").id
        self.apple_geom = m.geom("apple").id
        jid = m.joint("apple_free").id
        self.apple_q, self.apple_v = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        self.a_waist = m.actuator("waist_yaw_joint").id

    # ---------------------------------------------------- setup
    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        if seed is not None:   # jitter where the apple sits and how it is turned
            rng = np.random.default_rng(seed)
            self.data.qpos[self.apple_q:self.apple_q + 2] += rng.uniform(-0.012, 0.012, 2)
            yaw = rng.uniform(-np.pi, np.pi)
            self.data.qpos[self.apple_q + 3:self.apple_q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        self.R.reset(); self.L.reset()
        self.r_start = (self.R.grip_pos(), self.R.grip_quat())
        self.l_start = (self.L.grip_pos(), self.L.grip_quat())
        a0 = self.data.xpos[self.apple].copy()
        a0[2] = APPLE0_Z
        press = a0 + [0, 0, -0.006]           # palm presses lightly on the apple
        # right hand: (end time, grip point, orientation, finger pose)
        self.plan = [
            (1.2, a0 + [0, 0, 0.08], R_PALM_DOWN, "open"),    # hover over the apple
            (2.0, press, R_PALM_DOWN, "open"),                # come down onto it
            (2.6, press, R_PALM_DOWN, "grasp"),               # wrap the fingers
            (2.8, press, R_PALM_DOWN, "grasp"),
            (3.6, a0 + [0, 0, 0.15], R_PALM_DOWN, "grasp"),   # lift
            (4.6, WIND + [0, 0, 0.05], R_PALM_IN, "grasp"),   # roll the wrist...
            (5.6, WIND, R_PALM_UP, "grasp"),                  # ...until the palm is up
            (6.2, WIND, R_PALM_UP, "cup"),                    # loosen into a cradle
        ]
        self.t_throw = self.plan[-1][0]
        self.state, self.t_release, self.t_caught = "carry", None, None
        self.catch_target = L_READY.copy()
        self.free_ticks = 0
        self.trail, self.log = [], []

    # ---------------------------------------------------- right hand
    def right_target(self, t):
        if t < self.t_throw:
            (p0, q0), h0, t0 = self.r_start, "open", 0.0
            for t1, p1, q1, h1 in self.plan:
                if t <= t1:
                    s = min_jerk((t - t0) / (t1 - t0))
                    return p0 + s * (p1 - p0), slerp(q0, q1, s), blend(HAND[h0], HAND[h1], s)
                p0, q0, h0, t0 = p1, q1, h1, t1
        tau = t - self.t_throw
        if tau < T_ACC:                         # accelerate; fingers flatten under the apple
            hand = blend(HAND["cup"], HAND["toss"], min_jerk(tau / (0.5 * T_ACC)))
            return WIND + 0.5 * (V_REL / T_ACC) * tau**2, R_PALM_UP, hand
        p1 = WIND + 0.5 * V_REL * T_ACC
        tau -= T_ACC
        if tau < T_DEC:                         # brake hard (> g): the apple flies on
            return p1 + V_REL * tau - 0.5 * (V_REL / T_DEC) * tau**2, R_PALM_UP, HAND["toss"]
        p2 = p1 + 0.5 * V_REL * T_DEC
        s = min_jerk((tau - T_DEC - 0.1) / 0.7)  # then get out of the way
        return (p2 + s * (R_PARK - p2), slerp(R_PALM_UP, R_PALM_IN, s),
                blend(HAND["toss"], HAND["open"], s))

    # ---------------------------------------------------- left hand
    def left_target(self, t):
        if self.state == "carry":
            s = min_jerk((t - 4.0) / 1.6)
            p0, q0 = self.l_start
            return p0 + s * (L_READY - p0), slerp(q0, L_CATCH_Q, s), blend(HAND["open"], HAND["catch"], s)
        if self.state == "flight":
            return self.catch_target, L_CATCH_Q, HAND["catch"]
        tc = t - self.t_caught                  # caught: give way, close, present
        u = np.clip(tc / T_CUSHION, 0, 1)
        sink = self.catch_target - [0, 0, CUSHION * (1 - (1 - u) ** 3)]   # ease-out
        s = min_jerk((tc - 0.5) / 1.2)
        return (sink + s * (L_SHOW - sink), slerp(L_CATCH_Q, L_PALM_UP, s),
                blend(HAND["catch"], HAND["cup"], min(1.0, tc / 0.08)))

    # ---------------------------------------------------- whole body
    def waist_target(self, t):
        if self.state == "caught":
            return 0.18 * min_jerk((t - self.t_caught - 0.5) / 1.2)   # face the caught apple
        turn = -0.18 * min_jerk((t - 0.2) / 1.2)                      # lean into the pick
        return turn + 0.18 * min_jerk((t - 3.6) / 2.0)                # square up for the throw

    def touching(self, geoms):
        for c in self.data.contact[: self.data.ncon]:
            if (c.geom1 == self.apple_geom and c.geom2 in geoms) or \
               (c.geom2 == self.apple_geom and c.geom1 in geoms):
                return True
        return False

    def control(self):
        d, t = self.data, self.data.time
        ap = d.xpos[self.apple].copy()
        av = d.qvel[self.apple_v:self.apple_v + 3].copy()

        # release = right hand is braking and the apple has been free for a few ticks
        free = t > self.t_throw + 0.8 * T_ACC and av[2] > 0.3 and not self.touching(self.R.geoms)
        self.free_ticks = self.free_ticks + 1 if free else 0
        if self.state == "carry" and self.free_ticks >= 5:
            self.state, self.t_release = "flight", t
            self.log.append(f"t={t:5.2f}s  release  pos {ap.round(3)}  vel {av.round(2)}")
        if self.state == "flight":
            self.trail.append(ap)
            land = predict_landing(ap, av, CATCH_Z)     # re-plan every tick
            if land is not None:
                self.catch_target = land
            if self.touching(self.L.geoms):
                self.state, self.t_caught = "caught", t
                self.log.append(f"t={t:5.2f}s  caught   pos {ap.round(3)}  flight "
                                f"{t - self.t_release:.2f}s  apex {max(p[2] for p in self.trail):.2f} m")

        self.R.command(*self.right_target(t), self.dt_ctrl)
        self.L.command(*self.left_target(t), self.dt_ctrl)
        d.ctrl[self.a_waist] = self.waist_target(t)

    def step(self):
        if round(self.data.time / self.model.opt.timestep) % self.CTRL_EVERY == 0:
            self.control()
        mujoco.mj_step(self.model, self.data)

    def success(self):
        ap = self.data.xpos[self.apple]
        return self.state == "caught" and np.linalg.norm(ap - self.L.grip_pos()) < 0.05 and ap[2] > 0.85


# ----------------------------------------------------------------- rendering
def draw_trail(scene, trail):
    """the apple's flight path as small translucent spheres"""
    for p in trail[::4]:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.005, 0, 0]), np.asarray(p, float), np.eye(3).ravel(),
                            np.array([1.0, 0.55, 0.1, 0.6], np.float32))
        scene.ngeom += 1


def make_video(task, out, camera, duration, fps, slowmo, width=1280, height=720):
    import imageio
    renderer = mujoco.Renderer(task.model, height, width)
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    next_frame = 0.0
    while task.data.time < duration:
        task.step()
        t = task.data.time
        if t >= next_frame:
            renderer.update_scene(task.data, camera=camera)
            draw_trail(renderer.scene, task.trail)
            writer.append_data(renderer.render())
            slow = task.t_throw - 0.1 < t < (task.t_caught or 1e9) + 0.4   # slow-mo toss + catch
            next_frame += 1.0 / (fps * (slowmo if slow else 1))
    writer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="apple_toss_g1.mp4")
    ap.add_argument("--camera", default="front", choices=["front", "side", "wide", "eyes"])
    ap.add_argument("--duration", type=float, default=9.5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--slowmo", type=float, default=2.0, help="slow-motion factor for toss + catch")
    ap.add_argument("--seed", type=int, default=None, help="jitter the apple start pose")
    ap.add_argument("--trials", type=int, default=0, help="run N jittered trials headless and report")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--view", action="store_true", help="interactive viewer (needs a display)")
    args = ap.parse_args()

    task = AppleToss()
    if args.trials:
        ok = 0
        for k in range(args.trials):
            task.reset(seed=k)
            while task.data.time < args.duration:
                task.step()
            ok += task.success()
            print(f"trial {k:2d}: {'caught' if task.success() else 'MISSED'}", flush=True)
        print(f"\nsuccess {ok}/{args.trials}")
        return

    task.reset(seed=args.seed)
    if args.view:
        import time
        import mujoco.viewer
        with mujoco.viewer.launch_passive(task.model, task.data) as v:
            while v.is_running() and task.data.time < args.duration:
                t0 = time.time()
                task.step()
                v.sync()
                time.sleep(max(0, task.model.opt.timestep - (time.time() - t0)))
    elif args.no_video:
        while task.data.time < args.duration:
            task.step()
    else:
        make_video(task, args.out, args.camera, args.duration, args.fps, args.slowmo)
        print(f"saved {args.out}")
    print("\n".join(task.log))
    print("RESULT:", "apple caught and held" if task.success() else "missed")


if __name__ == "__main__":
    main()
