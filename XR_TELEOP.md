# XR(Quest 2) 텔레옵 — 구현 내용 및 사용법

FALCON policy가 하체 균형·보행을 담당하고, XR 헤드셋(Quest 2 등) 컨트롤러로 상체(양팔)를 실시간
조작하는 기능. Mujoco sim2sim에서 동작 확인 완료. 1인칭 카메라(로봇 시점 영상)도 지원.

## 구현 파일

모든 경로는 `FALCON_experiment/` 기준.

| 파일 | 내용 |
| --- | --- |
| `sim2real/rl_policy/loco_manip/loco_manip_xr.py` | 신규. `LocoManipPolicy`를 상속해 XR 입력을 팔 목표로 연결하는 브리지 |
| `sim2real/utils/head_cam_shm.py` | 신규. 머리 카메라 프레임을 프로세스 간 전달하는 공유메모리 유틸 (seqlock) |
| `sim2real/sim_env/base_sim.py` | 수정. `--head_cam` 옵션 — 시뮬레이션 스레드에서 오프스크린 렌더링 후 공유메모리에 publish |
| `sim2real/sim_env/loco_manip.py` | 수정. `--head_cam` 인자를 `BaseSimulator`로 전달 |
| `sim2real/config/g1/g1_29dof_falcon_xr.yaml` | 신규. XR용 테스트 config (`use_upper_body_controller: true` 등) |

XR 입력 자체는 `xr_teleoperate/teleop/televuer`(서브모듈)의 `televuer` 패키지를 그대로 사용 —
xr_teleoperate 쪽 코드는 수정하지 않았음.

## 동작 원리

```
Quest 2 (Meta Quest Browser, WebXR)
    |  https://<PC-IP>:8012/?ws=wss://<PC-IP>:8012
televuer (xr_teleoperate/teleop/televuer, fcreal env에 설치)
    |  손목 포즈(SE3), 컨트롤러 버튼/트리거/그립
loco_manip_xr.py
    |  EE_left/right_{x,y,z} 갱신 (데드맨 스위치로 게이팅) → update_waypoints()
    |  ※ 이 변수들은 원래 조이스틱으로 팔을 조작하던 기존 경로(use_upper_body_controller)를 그대로 재사용
G1_29_ArmIK_NoWrists (pinocchio + casadi, 기존 IK)
    |  ref_upper_dof_pos (14 DoF)
FALCON policy (residual_upper_body_action)
    |  LowCmd (DDS)
Mujoco (FALCON_experiment/sim2real/sim_env/loco_manip.py)
    |  (--head_cam 사용 시) head_camera 렌더 → 공유메모리 → loco_manip_xr.py → render_to_xr() → Quest 2
```

핵심 통합 지점은 `LocoManipPolicy`가 이미 갖고 있던 `waypoints_left/right` + `update_waypoints()`
+ `G1_29_ArmIK_NoWrists`(조이스틱 팔 조작 경로, `use_upper_body_controller: true`)를 그대로 재사용한
것. XR 쪽에서 하는 일은 같은 변수를 조이스틱 델타 대신 손목 절대 위치로 채우는 것뿐이다.

## 설치 (한 번만)

```bash
conda activate fcreal
cd xr_teleoperate/teleop/televuer   # FALCON_BD_real/xr_teleoperate 서브모듈, 별도 클론 불필요
pip install -e .
pip install params_proto==2.13.2    # 필수: 3.x가 깔리면 vuer import가 깨짐 (엉뚱한 aiohttp 에러 메시지 출력)
```

설치 확인: `python -c "from televuer import TeleVuerWrapper; print('OK')"`

### SSL 인증서 (이미 생성돼 있음)
```
~/.config/xr_teleoperate/cert.pem
~/.config/xr_teleoperate/key.pem
```
없으면: `openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem -subj "/CN=$(hostname -I | awk '{print $1}')"`

### 방화벽
`sudo ufw allow 8012` (이미 적용됨)

## 실행

### 팔 조작만 (패스스루 — 헤드셋 너머로 모니터를 보며 조작)

```bash
# 터미널 1 — Mujoco
conda activate fcreal
cd FALCON_experiment/sim2real
python sim_env/loco_manip.py --config=config/g1/g1_29dof_falcon_xr.yaml

# 터미널 2 — 정책 + XR 서버
conda activate fcreal
cd FALCON_experiment/sim2real
python rl_policy/loco_manip/loco_manip_xr.py \
  --config=config/g1/g1_29dof_falcon_xr.yaml \
  --model_path=models/falcon/g1_29dof.onnx \
  --cert_file=$HOME/.config/xr_teleoperate/cert.pem \
  --key_file=$HOME/.config/xr_teleoperate/key.pem \
  --xr_debug
```

Quest 2 브라우저: `https://<PC-IP>:8012/?ws=wss://<PC-IP>:8012` → 인증서 경고 진행 → **Enter VR**.

### 1인칭 카메라도 함께 (양쪽에 옵션 추가)

```bash
# 터미널 1 (FALCON_experiment/sim2real 안에서)
python sim_env/loco_manip.py --config=config/g1/g1_29dof_falcon_xr.yaml --head_cam

# 터미널 2 (FALCON_experiment/sim2real 안에서)
python rl_policy/loco_manip/loco_manip_xr.py \
  --config=config/g1/g1_29dof_falcon_xr.yaml \
  --model_path=models/falcon/g1_29dof.onnx \
  --cert_file=$HOME/.config/xr_teleoperate/cert.pem \
  --key_file=$HOME/.config/xr_teleoperate/key.pem \
  --display_mode immersive --head_cam --xr_debug
```

> `immersive` 모드는 헤드셋을 쓰면 모니터/키보드가 안 보인다. **로봇을 먼저 세워둔 뒤** 헤드셋을
> 착용할 것. 비상정지는 오른손 **B**.


### 로봇 세우기 (헤드셋 착용 전, 키보드로)

| 순서 | 어디서 | 무엇을 |
| --- | --- | --- |
| 1 | Mujoco 창 (클릭해 포커스) | `8` 을 발이 땅에 닿을 때까지 여러 번 |
| 2 | 터미널 2 | `i` — 기본 자세로 보간 |
| 3 | 터미널 2 | `]` — 정책 시작 |
| 4 | Mujoco 창 | `9` — 탄성 밴드 해제. 로봇이 스스로 서야 정상 |

### 조작

| 입력 | 동작 |
| --- | --- |
| 양손 트리거 또는 그립 유지 | 팔 트래킹 ON (데드맨 스위치, 놓으면 그 자리에 정지) |
| 오른손 B | 정책 정지 — 비상정지 |
| 오른손 A | 정책 시작 |
| 왼손 X | 기본 자세로 초기화 |
| 왼손 Y | 제자리 ↔ 걷기 전환 |
| 왼쪽 썸스틱 | 전후/좌우 이동 (걷기 모드에서만) |
| 오른쪽 썸스틱 좌우 | 회전 (걷기 모드에서만) |

쥐는 순간의 손 위치가 로봇 팔의 현재 목표 위치로 영점 잡히므로(rising-edge offset), 팔이 튀지 않는다.

### 주요 CLI 옵션 (`loco_manip_xr.py`)

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--engage` | `any` | 데드맨 스위치: `any`(트리거 or 그립) / `trigger` / `squeeze` / `always`(테스트 전용, 실사용 금지) |
| `--display_mode` | `pass-through` | `immersive`면 로봇 1인칭 영상 (`--head_cam` 필요) |
| `--motion_scale` | `1.0` | 손목 이동 → 로봇 EE 이동 배율 |
| `--xr_mode` | `controller` | `hand`로 핸드 트래킹 (Quest 2는 품질 낮음) |
| `--xr_debug` | 꺼짐 | 주기적으로 원시 XR 상태 출력. 연결 문제 생기면 제일 먼저 켤 것 |

## 검증 완료 사항 (Mujoco sim2sim)

- Quest 2 컨트롤러 트래킹 → 팔 목표(EE waypoint) 갱신 → IK → policy residual → 실제 로봇 팔 이동 확인
- 데드맨 스위치(그립) engage/release 정상 작동
- `--head_cam` + `--display_mode immersive`: 헤드셋에 로봇 1인칭 시점 영상 표시 확인
- GLFW 뷰어와 `mujoco.Renderer`(오프스크린) 같은 스레드 공존 확인, FPS 저하 미미 (~196 → ~194)

### 구현 중 발견/수정한 버그
- `params_proto` 3.x가 설치되면 `vuer` import가 깨짐 (에러 메시지는 엉뚱하게 "aiohttp 설치하라"로 나옴) → `2.13.2`로 고정
- `display_mode="immersive"`는 `zmq=True`가 필수 (빠뜨리면 `ValueError`)
- `render_to_xr()`는 내부적으로 BGR→RGB 변환을 하므로, Mujoco의 RGB 출력을 미리 `[:, :, ::-1]`로
  뒤집어 넘겨야 헤드셋에서 색이 정상으로 보임
- **`multiprocessing.resource_tracker` 버그**: 공유메모리 구독자(`HeadCamSubscriber`) 프로세스가
  종료되면, 자신이 만들지 않고 열기만 한 공유메모리 블록까지 자동으로 삭제해버림. XR 브리지를
  재시작할 때마다 sim_env의 카메라 공유메모리가 사라지는 문제로 나타남 →
  `resource_tracker.unregister()`로 구독자를 등록 해제해서 해결
- sim_env와 policy 프로세스가 서로 다른 config(특히 `INTERFACE`)로 뜨면 DDS로 서로를 못 봐서
  팔이 전혀 안 움직임 — sim2sim에서는 **반드시 두 프로세스가 같은 config**를 써야 함

## 실물 로봇(sim2real) 적용 시

**팔 조작 경로는 코드 변경 없이 그대로 적용 가능.** `waypoints_left/right` → IK → policy residual은
policy 레벨 로직이라 시뮬레이션/실물과 무관. `sim_env` 없이 policy 프로세스만 실행하고, config의
`INTERFACE`를 실제 이더넷 인터페이스로 바꾸면 된다:

```bash
# FALCON_experiment/sim2real 안에서
python rl_policy/loco_manip/loco_manip_xr.py \
  --config=config/g1/g1_29dof_falcon.yaml \    # INTERFACE를 실제 이더넷으로
  --model_path=... --cert_file=... --key_file=...
```

**1인칭 카메라는 프레임 소스만 교체하면 됨.** `mujoco.Renderer`는 시뮬레이션 전용이라 실물에는
안 맞지만, 공유메모리 배관(`head_cam_shm.py`)과 `render_to_xr()` 호출부는 그대로 재사용 가능 —
`base_sim.py`의 렌더 호출 자리를 xr_teleoperate의 `teleimager`(실제 카메라 스트림)로 바꾸면 된다.
아직 미구현.

**실물 배포 전 반드시 강화할 것:**
- `--engage always`는 테스트 전용, 실사용 절대 금지 (데드맨 스위치 없이 항상 추종)
- 비상정지(오른손 B)가 단일 버튼이라 오조작 위험 있음 — 쿨다운이나 조합키 추가 권장
- 워크스페이스 클램프(`WORKSPACE_X/Y/Z`, `loco_manip_xr.py` 상단)는 시뮬레이션 기준으로 잡은 값,
  실물 팔 가동범위/충돌 여부 재검증 필요
- `FALCON_experiment/sim2real/README.md`의 표준 안전수칙(사람 대기, kp 낮춰서 시작, Low-level 모드 진입) 동일 적용
