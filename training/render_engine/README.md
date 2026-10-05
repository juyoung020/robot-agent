# render_engine — 학습용 GPU 일괄 렌더(RenderBatch)

BC 영상 학생·`app_table`·`render_bench` 가 읽기만 하는 렌더 코드. 원래 옛 시뮬 저장소의 `src/sim/engine`(core/render, core/common 일부, cuda/render)에 있던 것을 10-06 robot-agent 로 옮겼다(옛 저장소 분리). `training/BC/CMakeLists.txt` 의 `ENGINE_DIR` 기본값이 이 폴더다.
