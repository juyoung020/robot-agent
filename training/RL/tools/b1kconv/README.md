# b1kconv — BEHAVIOR 2026 장면·과제 → 장치 형식 (계획서 E1)

[CURRICULUM_BEHAVIOR2026.md](../../../../docs/map_vla/CURRICULUM_BEHAVIOR2026.md) 5절 E1. BEHAVIOR 2026 의 장면 7 개·과제 100 개·인스턴스 32,000 개를 **장면마다 작은 파일 하나(RASC v1)** 로 바꾸는 오프라인 Rust 도구와, 그것을 읽는 C++ 참조 리더다.
입력은 평문 JSON·PNG·BDDL 뿐이다(`src/behavior-2026` 서브모듈, 읽기만). **암호화 USD 는 읽지도 풀지도 않는다.** Python 은 쓰지 않는다.
학습 환경(`training/RL/env`, `map`)은 건드리지 않는다 — 붙이는 것은 E2.

## 구성
- `src/` Rust 변환기. `bddl.rs`(BDDL 파서·접지), `scene.rs`(장면 JSON·배치 PNG), `task.rs`(과제·템플릿·인스턴스), `feas.rs`(LIMO + OMX-F 가능성 표), `format.rs`(레코드 정의 = C 헤더 생성원), `write.rs`, `verify.rs`, `png.rs`(8 비트 회색 PNG 만)
- `cpp/rasc_format.h` — `b1kconv header` 가 만든 **생성 파일**(손으로 고치지 않음). 레코드 크기·오프셋 `static_assert`, 배치 해시 `RASC_LAYOUT_HASH`
- `cpp/rasc.h` — 헤더만인 C++17 로더 + CPU 참조 함수(`cell`, `room_at`, `free_at`, `in_box`, `arg_cands`, `lit_str`)
- `cpp/rasc_test.cpp` — 왕복·불변식·손상 파일 거절 시험

## 빌드·실행
```
cd training/RL/tools/b1kconv
cargo build --release -j 4
./target/release/b1kconv convert --out ~/ra_b1k          # 7 장면 → ~/ra_b1k/<장면>.rasc + .expect, task_table.md, feasibility.tsv, summary.txt (약 4 s, 최대 RSS 약 240 MB)
./target/release/b1kconv verify  --out ~/ra_b1k          # 파일을 다시 읽어 원본 JSON·BDDL 과 값 비교 (0 다름이어야 통과)
./target/release/b1kconv negative                        # 깨진 입력 3 종이 오류로 멈춰야 통과(종료 코드 0)
./target/release/b1kconv header --check cpp/rasc_format.h   # 헤더가 지금 배치와 같은지
cmake -S cpp -B ~/ra_b1kbuild && cmake --build ~/ra_b1kbuild -j 4 && ~/ra_b1kbuild/rasc_test ~/ra_b1k/*.rasc   # 또는 ctest
```
`--scene NAME` 으로 장면 하나만, `--b1k PATH` 로 BEHAVIOR-1K 위치, `--doc PATH` 로 비교할 문서를 바꾼다. 배치(레코드)를 고치면 `b1kconv header --out cpp/rasc_format.h` 로 헤더를 다시 만든다(해시가 바뀌어 옛 파일은 로더가 거절).

## RASC v1 형식
리틀 엔디언, 장면당 파일 하나. 머리(`RascFileHeader`, 688 B) 뒤에 64 B 정렬 구역 21 개. 구역 표가 오프셋·바이트·개수·원소 크기를 갖고, 로더는 매직·버전·머리 크기·배치 해시·구역 크기를 모두 확인한다.

| 구역 | 원소 | 내용 |
|---|---|---|
| STRINGS | char | 이름(NUL 끝). 레코드는 바이트 오프셋을 가짐 |
| CATS | `RascCatRec` 32 B | OmniGibson 종류, 종류 평균 질량·부피·밀도(`avg_category_specs.json`, 없으면 NaN) |
| ROOMS | `RascRoomRec` 48 B | 방 인스턴스. 이름은 OmniGibson `SegmentationMap` 과 같은 규칙(0.10 m 최근접 축소 → id 오름차순 → `<종류>_<k>`), 종류 = `room_categories.txt` 줄 번호, 넓이·중심·범위(0.01 m 원본) |
| OBJS | `RascObjRec` 128 B | `<장면>_best.json` 의 모든 물체(파일 순서): 종류, 모델, 자세(위치·쿼터니언), 척도, 상자(`bbox_size × scale`, 중심 = 위치 + R·(scale·`base_link_offset`)), 세계 AABB, yaw, 종류 평균 질량, 관절 위치, 방(in_rooms), 플래그(벽·바닥·천장·문·창·계단·고정·관절체·열림·토글·열원·물·붙이기·조명·기울어짐) |
| BOXES | `RascBoxRec` 48 B | 정적 충돌 상자 = 바닥·천장을 뺀 모든 장면 물체(벽 포함, 플래그로 구분). 중심·반치수·yaw·z 범위 |
| DOORS | `RascDoorRec` 48 B | 종류에 `door` 가 든 물체(door / sliding_door / garage_door / elevator_door), 두 방, 관절 |
| JOINTS | f32 | 관절 위치 묶음(장면 물체·인스턴스 물체) |
| ROOM_GRID | u8 [H][W] | 0.10 m 칸의 방(RoomRec 번호 + 1, 0 = 방 아님) = 10 × 10 화소 다수결 |
| TRAV, TRAV_NO_OBJ, TRAV_NO_DOOR, TRAV_OPEN_DOOR | 비트 [H][W] | `floor_trav_{,no_obj,no_door,open_door}_0.png`. 칸의 100 화소가 모두 255 면 1. NO_OBJ = 벽만(점유) |
| TASKS | `RascTaskRec` 96 B | 과제: 번호 0–99, 사람 길이·제한 시간(×1.5)·이동 거리, 아래 표들의 범위, 불러오는 방 비트(`B100_task_misc.csv`), 문서 기준 조건 수, q 상한 엄격·느슨, 스킬 비트 |
| TASK_OBJS | `RascTaskObjRec` 92 B | BDDL `:objects` 순서. 인스턴스·synset·OmniGibson 이름, 종류, 장면 물체 번호(아니면 −1), 척도·반치수·상자 오프셋, 질량, 템플릿 자세, 플래그(로봇·시스템(입자)·미래·와일드카드·안 맞음) |
| LITS | `RascLitRec` 16 B | init 과 goal 문자. 술어 id(`RASC_P_*`), 부정, 인자 2(과제 물체 번호 / `RASC_ARG_VAR` 변수 / `RASC_ARG_ROOM` 방 종류), or 묶음·가지, 플래그 DOC(문서 셈에 듦)·INIT_TRUE(BDDL init 닫힌 세계에서 이미 참)·IN_OR |
| VARS, CANDS | `RascVarRec` 8 B, u16 | 목표 변수: exists(후보 중 하나), forn·forpairs(같은 묶음끼리 서로 다른 물체). 후보 = 과제 물체 번호 |
| REMOVED | u16 | 과제 템플릿이 불러오지 않는 장면 물체(예: HSF `mailbox`) |
| INSTS | `RascInstRec` 48 B | 인스턴스: 과제, 나눔(0 학습 300 / 1 공개 평가 20, id 301–320), 로봇 시작 자세·yaw, 자세 범위 |
| POSES | `RascPoseRec` 36 B | 인스턴스마다 과제 물체 순서대로 자세·관절, 출처(1 인스턴스 파일, 2 장면 파일, 3 템플릿 — 알려진 결함만) |
| IN_ROOMS | u32 | 물체 `in_rooms` 이름 전부(문자열 오프셋). OBJS 의 `room_a/b` 는 앞 둘 |

- 층(level): 2026 장면은 모두 층 하나(`n_levels = 1`, 물체 `level = 0`). 2 층 집은 `_lower`·`_upper` 가 서로 다른 장면 파일이다.
- 좌표: 칸 (r, c) 중심 = `origin + ((c + .5)·0.1, (r + .5)·0.1)`. PNG 화소 (r, c) 는 `((c − N/2)·0.01, (r − N/2)·0.01)` (OmniGibson `world_to_map` 과 같음, 행 = y).
- 목표 문법: `forall` 은 펼치고, `exists`/`forn`/`forpairs` 는 변수, `or` 는 모든 가지를 묶음·가지 번호와 함께 넣는다. `not` 아래 `or`/`exists`, `imply`, `fornpairs` 는 2026 과제에 없고 나오면 오류.
- 넣지 않은 것: 거리장(문서 5.1 제안 — E2), 메시(암호화), 선·각속도, 관절 한계(USD 안에 있음).

## 크기 (잰 값, 2026-10-04)
| 장면 | 파일 | 물체 | 상자 | 문 | 방 | 격자 | 과제 | 인스턴스 | 가장 큰 구역 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| `house_single_floor` | 6.23 MB | 595 | 556 | 24 | 21 | 863² | 34 | 10,880 | POSES 4.33 MB, ROOM_GRID 0.74 MB |
| `house_double_floor_lower` | 4.37 MB | 260 | 241 | 7 | 6 | 594² | 32 | 10,240 | POSES 3.20 MB |
| `house_double_floor_upper` | 1.18 MB | 134 | 124 | 6 | 5 | 133² | 10 | 3,200 | POSES 0.94 MB |
| `restaurant_diner` | 1.39 MB | 184 | 171 | 3 | 5 | 238² | 8 | 2,560 | POSES 1.11 MB |
| `Rs_int` | 0.68 MB | 80 | 70 | 2 | 5 | 83² | 6 | 1,920 | POSES 0.51 MB |
| `hotel_suite_large` | 0.69 MB | 63 | 59 | 2 | 2 | 123² | 5 | 1,600 | POSES 0.56 MB |
| `office_cubicles_right` | 0.79 MB | 297 | 267 | 15 | 12 | 419² | 5 | 1,600 | POSES 0.37 MB |
| 합 | 15.3 MB | | | | | | 100 | 32,000 | |

## LIMO + OMX-F 가능성 표 (문서 1.3·1.5 다시 내기)
`feas.rs` 는 문서 표를 만든 일회성 스크립트의 규칙을 그대로 옮겼다(키워드 목록·순서 버릇까지 — 통과 기준이 "같은 숫자"라서). 입력은 공개 평가 템플릿(`scene_test/public/.../*_0_0_template-partial_rooms.json`, 없으면 `_template.json`)의 `inst_to_name`·자세·척도, `metadata.json` 의 `bbox_size`, 종류 평균 질량.
- 접지: `forall` 펼침, `exists` 1 개, `forn n` n 개, `forpairs` min(|A|, |B|), `or` 첫 가지 = **문자 550 개**.
- 문자마다 "옮길 물체를 집을 수 있나(높이·무게·폭·닫힌 곳 안) + 목적지에 놓을 수 있나(놓을 높이·열기·선반·붙이기)", 토글은 물체 윗면 높이, `not open` 은 "처음부터 참" 으로 못 셈, 나머지 술어(입자·요리·자르기)는 못 함.
- 엄격 = 집는 면 0.45 m·놓는 면 0.50 m·0.25 kg·최소 가로 폭 0.08 m·토글 0.55 m, 느슨 = 0.60·0.62·0.5·0.10·0.70.
- 높이는 문서와 같이 `루트 z ± bbox_z/2`(회전·상자 오프셋 무시). `feasibility.tsv` 에는 회전·오프셋을 넣은 세계 상자로 다시 잰 값(`q_*_world`)도 있다.

결과(`summary.txt`): 엄격 7 과제 [26, 27, 30, 54, 79, 80, 89]·q = 1 은 [80]·평균 0.0267, 느슨 14 과제·q = 1 [0, 1, 80, 87]·평균 0.0874, 문자 550, `not open` 31 개 / 22 과제, 스킬 과제 수 M 36·P 77·O 38·T 7·H 9·S 9·C 11·F 6·A 4·W 13, 1.3절 표 100 줄 열 9 개 + 막는 것 **0 다름** → 기준 ① 통과.

## 검증 (잰 값, 2026-10-04)
- 기준 ① 문서 숫자: 위와 같이 **통과**.
- 기준 ② `verify`: 7 장면 842,695 값(물체 이름·모델·종류·척도·자세·관절·방, 과제 물체·OmniGibson 이름, init 문자 원문, 인스턴스 로봇 자세·물체 자세·관절, 나눔 개수) **0 다름**. 실수는 f32 비트 비교.
- 기준 ③ `negative`: 인스턴스 파일에서 과제 물체 하나 지움 → 오류로 멈춤, 목표가 없는 물체를 가리킴 → 오류, 배치 해시 1 비트 바꾼 파일 → 거절. 3/3.
- C++ `rasc_test`: 7 파일 모두 통과. ① `.expect`(변환기가 쓴 구역 개수·FNV-1a 64) 와 C++ 가 본 바이트가 같음 ② 다시 쓰고 읽은 바이트가 같음 ③ 상자 수 = 물체 − 바닥·천장, 문 수, 종류별 합 ④ AABB: 유한, 반치수 ≥ 0, 중심 ⊂ AABB, 쿼터니언 크기 1, 상자가 중심을 품음, 물체 중심이 격자 안 ⑤ 방: 방마다 칸 > 0, 칸 수 합 일치, 방 표시 물체 중 자기 방 칸 위 ≥ 0.90·0.35 m + 반치수 안 ≥ 0.98 ⑥ 문자 인자·변수·후보·or 묶음·인스턴스 자세 범위가 모두 유효, 문서 문자 수 일치 ⑦ 손상 4 종(해시·잘림·구역 크기·매직) 거절.

## 데이터에서 찾은 것 (원본 그대로 두고 표시만)
- **2025 과제(0–49) 학습 인스턴스는 일반 로봇 자세가 없다**: `robot_poses` 가 `R1Pro`·`Fetch`·`R1`·`Stretch`·`Tiago` 별로만 있고 id 는 0–299 다(새 과제 50–99 는 `robot`, id 1–300). 평가기는 `robot` 이 없으면 로봇 모델 이름 열쇠를 찾으므로(`eval/evaluator.py` 483–490) LIMO 로는 이 파일들을 그대로 못 쓴다. 변환기는 R1Pro 자세를 넣고 `InstRec.flags` 비트 0 을 켠다(15,000 개). 공개 평가 2,000 개는 모두 `robot`.
- `bringing_in_wood`: BDDL·전체 템플릿·인스턴스는 `plywood`, **공개 `-partial_rooms` 템플릿만 옛 `firewood`** 다. 문서 표의 "? 3" 이 이것 때문이다(문서 규칙대로 partial 템플릿을 써서 그대로 다시 냄). 장치 형식은 partial 에 없는 물체를 전체 템플릿에서 가져온다.
- `cook_a_frozen_pie` 공개 인스턴스 20 개에 `tray.n.01_1` 이 없다(학습 파일에는 있음) → 템플릿 자세(출처 3, 20 개). 같은 파일의 `countertop` 은 장면 물체라 장면 자세(출처 2, 20 개).
- `house_double_floor_upper`: 문 `door_pxhbim_0` 의 `in_rooms` 에 있는 `bathroom_1` 이 `floor_insseg_0.png` 에 없다(방 PNG 는 5 개).
- `office_cubicles_right` 벽 5 개는 `in_rooms` 가 3 개(IN_ROOMS 에 모두 둠).
- 다닐 수 있는 칸 중 방에 속한 비율: HDL 0.997·HSF 0.929·RS 0.838·HSL 0.830·OCR 0.495·RD 0.492 — 바깥(마당·주차장 등)이 다닐 곳이지만 방이 아니다.
- 인스턴스 파일에 BDDL 에 없는 열쇠가 있다(예: `rearrange_your_room` 의 `stand.n.04_2..4`) — 세어서 `summary.txt` 에 적고 버린다.
- 알려진 결함 표(`task.rs` `KNOWN_DEFECTS`) 밖에서 과제 물체가 빠지면 오류로 멈춘다.
