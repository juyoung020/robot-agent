"""OmniGibson 재생 시험: 관측 키·seg_instance 정보 모양 확인(몇 스텝만). flock /tmp/claude-1000/og.lock 아래에서."""
import sys, time, json
import omnigibson as og
from omnigibson.envs import HDF5PlaybackWrapper
from omnigibson.macros import gm
from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES
sys.path.insert(0, '/home/juyoung/robot-agent/src/behavior-2026/BEHAVIOR-1K/OmniGibson/scripts/learning')
import replay_obs as ro

gm.HEADLESS = True
gm.RENDER_VIEWER_CAMERA = False
gm.ENABLE_TRANSITION_RULES = False
task = 'turning_on_radio'
at = ro._load_challenge_available_tasks()
scene_model = at[task][0]['scene_model']
t0 = time.time()
env = HDF5PlaybackWrapper.create_from_hdf5(
    input_path=sys.argv[1], output_path='/tmp/claude-1000/og_probe.hdf5',
    full_scene_file=ro._find_full_scene_file(task, scene_model), load_room_instances=ro._load_room_instances(task),
    robot_sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": 360, "image_width": 360}},
                         "zed_link:Camera:0": {"sensor_kwargs": {"horizontal_aperture": 40.0, "image_height": 360, "image_width": 360}}},
    include_sensor_names=["zed_link"], n_render_iterations=1, flush_every_n_traj=1, include_robot_control=False,
    robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
    robot_obs_modalities=sys.argv[2].split(","), include_contacts=False)
print('load s', time.time() - t0)
env.load_observation_space()
seen = {}

def parse(self, action, obs, reward, terminated, truncated, info):
    if not seen:
        for k, v in obs.items():
            print('OBS', k, getattr(v, 'shape', type(v)), getattr(v, 'dtype', ''))
        print('INFO keys', list(info.keys())[:20])
        oi = info.get('obs_info', info)
        print('INFO sample', json.dumps(oi, default=str)[:3000])
        seen['x'] = 1
    self._n = getattr(self, '_n', 0) + 1
    if self._n >= 30:
        raise KeyboardInterrupt
    return {}
HDF5PlaybackWrapper._parse_step_data = parse
ids = sorted(int(k.split('_')[1]) for k in env.input_hdf5['data'].keys() if k.startswith('demo_'))
t1 = time.time()
try:
    env.playback_episode(episode_id=ids[-1], record_data=True)
except KeyboardInterrupt:
    pass
print('30 steps s', time.time() - t1)
rob = env.scene.robots[0]
print('robot', rob.name, rob.get_position_orientation())
print('sensors', list(rob.sensors.keys()))
cam = [s for n, s in rob.sensors.items() if 'zed' in n][0]
print('K', cam.intrinsic_matrix)
og.shutdown()
