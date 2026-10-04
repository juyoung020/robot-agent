"""가져온 limo_omx.usda 의 omx_gripper_joint_2 를 PhysX 미믹 관절로 만든다(joint_2 = -joint_1, 공식 URDF mimic multiplier=-1).
OmniGibson 임포터는 URDF <mimic> 을 지우므로 USD 에 직접 넣는다. PhysxMimicJointAPI 관계식:
  q2 + gearing * q1 + offset = 0   → gearing=1, offset=0 이면 q2 = -q1.
OmniGibson 은 미믹 관절의 드라이브 이득을 0 으로 두므로(JointPrim.set_control_type), MultiFingerGripperController
smooth 모드의 1차원 연속 행동이 joint_1 만 움직이고 joint_2 는 거울로 따라온다(+ = 벌림, - = 오므림).
여러 번 돌려도 된다(이미 있으면 그대로 둔다).
사용: python add_gripper_mimic.py <limo_omx.usda>
"""
import re
import sys

p = sys.argv[1]
t = open(p).read()
ROOT = re.search(r'defaultPrim = "([^"]+)"', t).group(1)
REF = "</%s/joints/omx_gripper_joint_1>" % ROOT
head = re.compile(r'(def PhysicsRevoluteJoint "omx_gripper_joint_2" \(\s*apiSchemas = \[)([^\]]*)(\]\s*\)\s*\{\n)')
m = head.search(t)
if not m:
    sys.exit("[mimic] omx_gripper_joint_2 정의를 못 찾음")
body_start = m.end()
body_end = t.index("\n        }", body_start)
body = t[body_start:body_end]
axis = re.search(r'physics:axis = "([XYZ])"', body).group(1)
inst = "rot" + axis
if "PhysxMimicJointAPI" in m.group(2):
    print("[mimic] 이미 있음:", re.search(r"physxMimicJoint:\w+:referenceJoint = <[^>]+>", body).group(0))
    sys.exit(0)
apis = m.group(2) + ', "PhysxMimicJointAPI:%s"' % inst
ind = re.match(r"\s*", body).group(0)
attrs = (
    "%sfloat physxMimicJoint:%s:gearing = 1\n" % (ind, inst)
    + "%sfloat physxMimicJoint:%s:offset = 0\n" % (ind, inst)
    + "%srel physxMimicJoint:%s:referenceJoint = %s\n" % (ind, inst, REF)
    + '%suniform token physxMimicJoint:%s:referenceJointAxis = "%s"\n' % (ind, inst, inst)
)
t = t[: m.start()] + m.group(1) + apis + m.group(3) + attrs + t[body_start:]
open(p, "w").write(t)
print("[mimic] omx_gripper_joint_2 = -omx_gripper_joint_1 (%s, ref %s)" % (inst, REF))
