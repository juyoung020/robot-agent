"""LIMO+OMX 의 eef_link(잡는 점)을 손끝 패드 사이로 옮긴다 — 설치된 USD 파일은 고치지 않고 불러온 무대(stage)에서만.
옛 자리: omx_end_effector_link 원점(= omx_link5 + (0.09193, -0.0016, 0), URDF omx_end_effector_joint) = 손가락 끝 바로 앞.
그리퍼를 열면 손끝이 뒤로 물러나(45° 에서 link5 x 0.082) 이 점에 둔 물체는 손가락 사이에 들지 않는다(E0 시뮬: g=0.0919 0/3 성공).
새 자리: omx_link5 + (0.080, -0.0016, 0) — omx_end_effector_link 에서 손가락 축(eef z) 뒤로 0.0119 m. 시뮬 잡기 성공 범위
link5 x 0.055..0.085 의 손끝 쪽(작은 물체 10–40 mm 모두 성공, docs/map_vla/CURRICULUM_BEHAVIOR2026.md E0).
같은 값은 limo_omx_source_config.yaml(eef_vis_links offset)에도 넣었다 — 다시 가져오면(import_to_omnigibson.sh) 이 패치는 할 일이 없다.
"""
EEF_OFFSET_IN_EE_LINK = (-0.0119, 0.0, 0.0)     # omx_end_effector_link 프레임(x = 손가락 쪽)
JOINT = "omx_end_effector_link/omx_end_effector_link_eef_link_joint"


def fix_stage(robot_prim):
    """불러온 로봇 prim 에서 eef 고정 관절의 localPos0 와 eef_link 의 놓인 자리를 옮긴다. 이미 옮겼으면 그대로. 바꿨으면 True"""
    from pxr import Gf
    stage = robot_prim.GetStage()
    root = str(robot_prim.GetPath())
    j = stage.GetPrimAtPath(f"{root}/{JOINT}")
    if not j.IsValid():
        return False
    a = j.GetAttribute("physics:localPos0")
    cur = a.Get()
    want = Gf.Vec3f(*EEF_OFFSET_IN_EE_LINK)
    if cur is not None and (Gf.Vec3f(cur) - want).GetLength() < 1e-6:
        return False
    a.Set(want)
    ee = stage.GetPrimAtPath(f"{root}/omx_end_effector_link")
    eef = stage.GetPrimAtPath(f"{root}/eef_link")
    t_ee = ee.GetAttribute("xformOp:translate").Get()
    q_ee = ee.GetAttribute("xformOp:orient").Get()
    off = Gf.Rotation(q_ee).TransformDir(Gf.Vec3d(*EEF_OFFSET_IN_EE_LINK)) if q_ee is not None else Gf.Vec3d(*EEF_OFFSET_IN_EE_LINK)
    if t_ee is not None and eef.IsValid():
        eef.GetAttribute("xformOp:translate").Set(type(t_ee)(t_ee[0] + off[0], t_ee[1] + off[1], t_ee[2] + off[2]))
    return True


def install():
    """Robot._load 를 감싸 limo_omx 를 불러올 때마다 fix_stage 를 부른다(여러 번 불러도 한 번만 감쌈)"""
    import omnigibson.robots.robot as R
    from omnigibson.objects.usd_object import USDObject
    if getattr(R.Robot, "_limo_eef_fix", False):
        return

    import omnigibson as og

    def _load(self):
        prim = USDObject._load(self)
        if getattr(self, "model", None) != "limo_omx":
            return prim
        with og.sim.editing_usd():     # OmniGibson 은 이 문맥 밖의 USD 수정을 막는다
            changed = fix_stage(prim)
        if changed:
            print(f"[limo] eef_link -> omx_end_effector_link + {EEF_OFFSET_IN_EE_LINK} (link5 x 0.080)", flush=True)
        return prim

    R.Robot._load = _load
    R.Robot._limo_eef_fix = True
