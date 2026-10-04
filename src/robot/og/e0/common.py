"""E0 측정 공통: 빈 장면(또는 BEHAVIOR 장면)에 LIMO+OMX 를 limo_omx_eval.yaml 제어기 설정으로 띄운다(헤드리스).
conda env behavior 에서 실행. OmniGibson 소스는 건드리지 않는다."""
import os
import pathlib

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
HERE = pathlib.Path(__file__).resolve().parent
OG_DIR = HERE.parent
HOME_ARM = [0.0, -1.6, 1.45, 0.15, 0.0]

# omx_link5 프레임에서 잡는 점(손끝 패드 사이). E0 에서 측정(README.md "잡는 점" 참고).
GRASP_IN_LINK5 = (0.080, -0.00165, 0.0)


def robot_cfg(obs=("proprio",), fixed_base=False):
    from omegaconf import OmegaConf
    y = OmegaConf.to_container(OmegaConf.load(OG_DIR / "limo_omx_eval.yaml"), resolve=True)
    return dict(model="limo_omx", name="robot_limo", obs_modalities=list(obs), action_normalize=False,
                self_collisions=False, grasping_mode="physical", controller_config=y["controller_config"],
                reset_joint_pos=y["reset_joint_pos"], proprio_obs="default", fixed_base=fixed_base,
                sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": 240, "image_width": 320,
                                                                  "clipping_range": [0.05, 1e7]}}})


def make_env(scene=None, objects=(), obs=("proprio",), physics_hz=120, action_hz=30):
    import omnigibson as og
    from omnigibson.macros import gm
    gm.ENABLE_FLATCACHE = True
    import sys
    sys.path.insert(0, str(OG_DIR))
    import limo_eef_fix
    limo_eef_fix.install()
    scene_cfg = {"type": "Scene"} if scene is None else scene
    cfg = dict(env={"action_frequency": action_hz, "physics_frequency": physics_hz, "rendering_frequency": action_hz},
               scene=scene_cfg, robots=[robot_cfg(obs)], objects=list(objects))
    env = og.Environment(configs=cfg)
    r = env.robots[0]
    while isinstance(r, (list, tuple)):
        r = r[0]
    return og, env, r


def link(robot, name):
    return robot.links[name]


def pose(robot, name):
    p, q = robot.links[name].get_position_orientation()
    return p, q
