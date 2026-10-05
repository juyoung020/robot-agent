# realbag — 공개 실제 로봇 ROS bag 으로 지각 파이프라인 돌리기

ROS 를 설치하지 않고 실제 로봇 bag 을 우리 파이프라인(ovdet 검출 + scenemap SLAM·물체 기억·장면 그래프)에 넣고, sgview 로 실시간처럼 다시 본다.
잰 값과 실패 사례는 robot-agent `docs/map_vla/MAP_STATE_PLAN.md` "실제 데이터" 절.

| 파일 | 하는 일 |
|---|---|
| `bag2stream.py` | (오프라인 변환만, 파이썬 `rosbags`) bag → 스트림 폴더: `rgb/`·`depth/`(uint16 mm, 컬러에 맞춘 깊이) PNG, `frames.csv`(영상 시각·채점용 정답), `odom.csv`(바퀴 오도메트리 전부), `meta.json`(내부 파라미터, base ← 카메라 `T_bc`). 종류 `openloris`(TF 그대로) · `tum_pioneer`(TF 가 이름뿐이라 깊이 바닥 평면으로 높이·기울기). `--refit-only` 는 `T_bc` 만 다시. `--scan-only`(10-06): 영상은 그대로 두고 2D 라이다만 — `scans.bin`(SCN1)·`gt.csv`(정답 전부)·meta `T_bl`(base ← 라이다)·`scan_dt`(정답 없이 오도메트리 yaw 속도와의 상호상관으로 잰 라이다 시계 어긋남, OpenLORIS 1-6·1-7 은 +0.7 s) → `--pose carto` 의 입력 |
| `realbag_run.cpp` | 스트림 → 검출(`--det fastsam`(기본) = 이름 없는 분할 엔진 + SigLIP 2 이름 — 엔진 기본은 ObjectSAM `yolo26n-seg-obj-416.plan`(YOLO26n 학생, `runtime/src/objprob_front.hpp` `kDefaultEngine`), 원래 FastSAM-s 는 `--engine …/FastSAM-s-416.plan`, 버린 FastSAM-s 재학습은 보관 `models/ovdet/archive/x86_sm120/FastSAM-s-416-obj.plan`, `yoloe`, `yolo`(닫힌 어휘, 기본 yolo26s-seg-416), `none`; `--namer siglip\|engine` — 기본은 fastsam siglip·나머지 engine) → scenemap(`--robot limo_omx` 기본, `--pose carto\|odom\|gt` — 기본 carto = Cartographer(`../../slam_carto`, scans.bin + 오도메트리), 스캔이 없는 스트림은 odom. 옛 slam2d 는 archive). 카메라 외부 자세는 `sm_set_cam_extrinsic`, 오도메트리는 LIMO proprio 0–5(영상 시각에 보간 하나 더). 검출 캐시 `--dump/--load`, 정답 비교(ATE: SE(2) 맞춤·첫 프레임 맞춤, 오도메트리만 대비), `--ref-map`(정답 자세로 만든 지도와 점유 칸 비교), `objects.csv` 는 물체 노드 전부(`structural` 1 = 큰 것·고정 종류 — metrics `live`·`dup_pairs` 는 예전처럼 작은 것만, `*_all` 이 전부), 여러 판을 한 지도에(`a,b,…` + `--pose gt`), 재생 판 `--sg <run>`(stream.sgs·memory·cam·meta.json, 학습 뷰어 group `real_bags`), 실시간 `--live host:port` |
| `sgrec2stream.py` | sgrt 기록(`SGRT_RECORD` rec.bin, LIMO 시뮬 판) → 같은 스트림 폴더(RGB-D 5 Hz·바퀴 오도메트리 30 Hz·정답 베이스 자세, `T_bc` = 시뮬 정답 카메라 자세, `gt_objects.json`) — 같은 시뮬 프레임을 여러 검출기로 |
| `detcmp_run.sh` · `detcmp_eval.py` | 검출기 비교: 한 검출기를 slam·gt 자세로(검출은 한 번, gt 판은 캐시), 판마다 프로세스 GPU 최대. 채점 = 정답 물체(보인 것만, 짝·중복·이름·크기별 재현율·구조물 위 헛것), 벽(scenemap `walls.csv` 와 sgview `/api/walls`), 점유(정답 바닥 지도) |
| `realbag_run --objprob` (기본 켬) | 확률 물체 모델(objprob)(scenemap README "scenemap 확률 모드") — SigLIP 2 이름 길(`--det fastsam`·`--namer siglip`·RBD2 캐시)이면 기본으로 켜지고 `--no-objprob` 이면 옛 이름 규칙. 낱말 표·라벨 사전·매개변수 읽기는 libsgrt 와 같이 쓰는 `runtime/src/objprob_front.hpp`. 분할 조각 + SigLIP 2 조각 임베딩을 scenemap 에 주고(`sm_set_det_embeddings`), 통째 다시 담기 요청을 SigLIP 2 로 돌려줌. 배경 낱말(계단·문틀·창틀·덤불 → 구조물 쪽, `kVocabAp`)·라벨 크기 사전·상위어(`kApLabels`). `--label-prior label_prior.json`(아래 맞추기가 정답 없이 잰 라벨 사전). 검출 캐시는 RBD2(검출마다 임베딩 FP16 — `--load` 로 GPU 없이 objprob 를 다시 돌림, 통째 다시 담기만 GPU). `--engine` 으로 다른 분할 엔진 |
| `realbag_run --inspect` | 살펴본 정도(scenemap README "살펴본 정도", `sm_set_inspect`): `memory/view.json`·`scene.json` 물체에 `inspect`(가장 가까이 본 거리·본 시점 수·윗면 본 비율). `metrics.json` 에 `objmap_us`(objmap 단계 평균 µs/검출 keyframe)·`inspect` |
| `objprob_fit.py` | (오프라인) objprob 맞추기: RBD2 캐시 + 깊이·정답 자세 + 정답 물체 → κ(모습 품질), 같은 것 로지스틱(관측↔물체, 물체↔물체, 문턱 표·작은 것↔가구·같은 종류 이웃 거짓 같음), 라벨 사전(정답 없이 EM), 기하 구조물 표. `objprob_params.json`(엔진별 매개변수 파일 — 아래)도 씀 |
| `objprob_refit.sh <이름> <엔진.plan\|-> [r3 캐시] [office1-1 캐시] [office1-5 캐시]` | 엔진 하나의 objprob 다시 맞추기·비교: 캐시가 없으면 그 엔진으로 radio r3(det-every 1)·OpenLORIS office1-1·1-5(det-every 3) 검출을 한 번(GPU, 짧게), `objprob_fit.py`, 문턱(same_p/merge_p) 몇 쌍 × 새 매개변수와 비교 기준(`PARAMS_BASE`, 기본 옛 엔진 파일)으로 r3 slam·gt 재생 → `objprob_eval.py`, OpenLORIS 재생 → 기하 대용. 결과 `~/datasets/objprob/refit/<이름>/` |
| `objprob_refit_table.py <out>[,<out>…]` | 위 결과 표(찾음·중복·잘못 합침·stuff 헛노드·문창계단·OpenLORIS) |
| `objprob_params/<엔진>.json` | 엔진별 objprob 매개변수: `obj_params`(`sm_set_obj_params` 문자열 — 로지스틱 `ap_w*`·`ap_wm*`, κ `kap_*`, 문턱, 그 밖 `ap_*`), `label_prior`(옆 파일). `realbag_run --objprob` 은 `--objprob-params` 가 없으면 엔진 파일 이름(.plan 뺀 것)으로 고름 — `--load` 캐시는 그 엔진을 `--engine` 으로 알려 줘야 함. `none` = 내장 기본값. `FastSAM-s-416.json` = 내장 기본값과 같음(radio r3 바이트 같음 확인). `FastSAM-s-416-obj.json` = 새 분할 엔진에 다시 맞춘 것(conf 0.25, 문턱 0.5/0.7 — 결과는 scenemap README "확률 모드"). `yolo26n-seg-obj-416.json` = ObjectSAM(YOLO26n 학생)에 다시 맞춘 것(conf 0.25, 문턱 0.5/0.7) — **기본 엔진**이라 `--engine` 없이도 이 파일이 실림. libsgrt(`SGRT_OBJPROB`)도 같은 폴더에서 고름 |
| `objprob_eval.py` | `detcmp_eval.py` 표 + 잘못 합침(노드 점 구름이 서로 다른 정답 둘 이상 — 작은 것+가구 / 같은 종류 이웃), 문·창·계단(찾음·맞는 이름·물체 이름 노드), 글 질의 R@1(μ·모습·이름) |
| `sgs_play.cpp` | 기록한 `stream.sgs` 를 벽시계에 맞춰 sgview(`--ingest`)로: 자세 60 Hz 보간, 지도·요약은 기록 그대로. `--rate`·`--loop`·`--ctl`(표준입력 seek·play·pause·rate 로 조종, 표준출력 상태 — 학습 뷰어 재생이 씀) |
| `record_live.mjs` · `make_video.sh` | 머리 없는 Chrome 으로 sgview 화면을 찍고(재생과 함께) bag RGB 를 작은 창으로 붙여 MP4·GIF |
| `rb_util.hpp` | JSON·폴더·JPEG·스트림 받기(Capture) |

```bash
python3 -m venv ~/realbag_venv && ~/realbag_venv/bin/pip install rosbags numpy opencv-python-headless
~/realbag_venv/bin/python src/scene_graph/tools/realbag/bag2stream.py openloris office1-1.bag ~/streams/ol_office1-1
cmake -S src/scene_graph/tools/realbag -B ~/realbag_build && cmake --build ~/realbag_build -j 4
~/realbag_build/realbag_run ~/streams/ol_office1-1 out/ol11 --dump dets/ol11.gz --sg ~/trainview_work/real_bags/ol_office1-1   # ObjectSAM + SigLIP 2 + objprob(기본)
sgview <run>/replays/<ep>.sg/memory --port 8080 --ingest 127.0.0.1:9001 &
~/realbag_build/sgs_play <run>/replays/<ep>.sg/stream.sgs 127.0.0.1:9001 --rate 1 --loop
```

데이터: OpenLORIS-Scene(CC BY-ND 4.0 — 원본 영상은 고치지 않고 출처 표시, 파생 데이터셋 배포 금지), TUM RGB-D fr2 pioneer(CC BY 4.0). bag·변환 결과는 커밋하지 않는다.
