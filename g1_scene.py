"""Builds the apple-toss scene around the Unitree G1 (29-DoF + Dex3 hands) from MuJoCo Menagerie.

The G1 model is loaded unchanged from menagerie and edited in code with MjSpec:
  * the floating base is removed (pelvis fixed at standing height, legs servo to straight)
  * gravity compensation on every robot link (like the real robot's feed-forward)
  * robot links collide with the apple / table / floor, not with each other
  * a table, a 150 g apple, lights, cameras and two palm "grip" sites are added
"""
import os
import sys
if sys.platform.startswith("linux"):   # headless OpenGL on Linux; Windows/macOS use the default backend
    os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

MENAGERIE = os.environ.get(
    "MENAGERIE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_menagerie"))
G1_XML = os.path.join(MENAGERIE, "unitree_g1", "g1_with_hands.xml")

APPLE_R = 0.033                          # 6.6 cm apple
TABLE_POS = np.array([0.30, -0.13])      # table centre (x, y)
TABLE_TOP = 0.78


def _look(pos, target):
    """xyaxes for a camera at pos looking at target"""
    pos = np.asarray(pos, float)
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1]); x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.concatenate([x, y])


def _delete(spec, el):
    """remove an element (MjSpec API differs between MuJoCo 3.2/3.3 and newer)"""
    if hasattr(spec, "delete"):
        spec.delete(el)
    else:
        el.delete()


def build_model():
    if not os.path.exists(G1_XML):
        raise FileNotFoundError(
            f"{G1_XML} not found. Clone menagerie next to this file or set MENAGERIE=/path/to/mujoco_menagerie")
    spec = mujoco.MjSpec.from_file(G1_XML)

    # ---- fixed base, no keyframes (they assume the free joint)
    _delete(spec, spec.joint("floating_base_joint"))
    for k in list(spec.keys):
        _delete(spec, k)

    # ---- physics options: elliptic cones + high impratio = grasps that don't creep
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    for b in spec.bodies:
        if b.name != "world":
            b.gravcomp = 1.0
    for g in spec.geoms:            # robot collision geoms: hit the world, not each other
        if g.contype or g.conaffinity:
            g.contype, g.conaffinity = 2, 0
    # finger pads need good friction; palms too
    for side in ("left", "right"):
        for g in spec.body(f"{side}_wrist_yaw_link").geoms:
            if g.contype:
                g.friction = [1.5, 0.05, 0.001]

    # ---- grip sites: where the apple centre sits in each palm
    #      hand frame: x = along fingers, palm faces +y (right) / -y (left)
    spec.body("right_wrist_yaw_link").add_site(name="r_grip", pos=[0.092, 0.018 + APPLE_R, 0],
                                               size=[0.006, 0, 0], rgba=[0, 1, 0, 0], group=5)
    spec.body("left_wrist_yaw_link").add_site(name="l_grip", pos=[0.092, -0.018 - APPLE_R, 0],
                                              size=[0.006, 0, 0], rgba=[0, 1, 0, 0], group=5)

    # ---- visuals
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.55, 0.7, 0.85],
                     rgb2=[0.08, 0.1, 0.15], width=512, height=512)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.2, 0.3, 0.4],
                     rgb2=[0.1, 0.2, 0.3], mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.8, 0.8, 0.8],
                     width=300, height=300)
    mat = spec.add_material(name="grid", texrepeat=[5, 5], texuniform=True, reflectance=0.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.add_material(name="wood", rgba=[0.55, 0.35, 0.2, 1])

    w = spec.worldbody
    light = w.add_light(pos=[1, -0.5, 3.0], dir=[-0.3, 0.15, -1], castshadow=True, diffuse=[0.7, 0.7, 0.7])
    if hasattr(mujoco, "mjtLightType"):          # MuJoCo >= 3.3.1
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    else:
        light.directional = True
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid",
               contype=1, conaffinity=3)

    # ---- table
    t = w.add_body(name="table", pos=[TABLE_POS[0], TABLE_POS[1], 0])
    t.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, TABLE_TOP - 0.015], size=[0.1, 0.1, 0.015],
               material="wood", contype=1, conaffinity=3)
    t.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, (TABLE_TOP - 0.03) / 2],
               size=[0.025, (TABLE_TOP - 0.03) / 2, 0], material="wood", contype=1, conaffinity=3)

    # ---- apple: free body, 150 g
    a = w.add_body(name="apple", pos=[TABLE_POS[0], TABLE_POS[1], TABLE_TOP + APPLE_R * 0.93])
    a.add_freejoint(name="apple_free")
    a.add_geom(name="apple", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[APPLE_R, APPLE_R, APPLE_R * 0.93],
               mass=0.15, rgba=[0.8, 0.05, 0.05, 1], friction=[1.5, 0.05, 0.001], solref=[0.01, 1],
               condim=4, contype=1, conaffinity=3)
    a.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, APPLE_R * 0.93 + 0.004],
               size=[0.0025, 0.007, 0], rgba=[0.35, 0.2, 0.05, 1], contype=0, conaffinity=0, mass=0.001)

    # ---- cameras
    tgt = [0.2, 0.0, 0.98]
    for name, pos in [("front", [1.25, -0.6, 1.25]), ("side", [1.4, -1.2, 1.35]),
                      ("wide", [2.0, -0.9, 1.4])]:
        cam = w.add_camera(name=name, pos=pos)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = _look(pos, tgt)
    cam = spec.body("torso_link").add_camera(name="eyes", pos=[0.08, 0, 0.45])
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = [0, -1, 0, 0.45, 0, 0.89]

    return spec.compile()


if __name__ == "__main__":
    import imageio
    m = build_model()
    d = mujoco.MjData(m)
    for _ in range(300):
        mujoco.mj_step(m, d)
    print("nq", m.nq, "nu", m.nu, "apple", d.body("apple").xpos.round(3))
    r = mujoco.Renderer(m, 720, 1280)
    r.update_scene(d, camera="front")
    imageio.imwrite("g1_scene.png", r.render())
