# FALCON Sim2Real Deployment Guide

실물 Unitree G1 29DoF 로봇에 FALCON policy를 배포하기 위한 가이드.  
시뮬레이션 검증 → 실물 배포 순서로 진행.

---

## Table of Contents

1. [설치](#설치)
2. [네트워크 설정 (S30 연결)](#네트워크-설정-s30-연결)
3. [로봇 켜기 & Debugging Mode 진입](#로봇-켜기--debugging-mode-진입)
4. [시뮬레이션 테스트 (Sim2Sim)](#시뮬레이션-테스트-sim2sim)
5. [실물 배포 (Sim2Real)](#실물-배포-sim2real)
   - [Original FALCON (오픈소스, dim=575)](#1-original-falcon-오픈소스-dim575)
   - [BD Policy (Balance Descriptor, dim=578)](#2-bd-policy-balance-descriptor-dim578)
6. [키보드 / 조이스틱 명령어](#키보드--조이스틱-명령어)

---

## 설치

```bash
conda create -n fcreal python=3.10
conda activate fcreal

# Pinocchio (IK용)
conda install pinocchio=3.2.0 -c conda-forge

# Unitree SDK
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e . && cd ..

# 기타 패키지
cd sim2real
pip install -r requirements.txt
```

---

## 네트워크 설정 (S30 연결)

### 하드웨어 연결

```
로봇 S30 포트 (앞면 하단) → 이더넷 케이블 → 노트북/PC
```

> G1은 S30 포트가 메인 통신 포트. LAN 케이블로 직결.

### IP 설정

로봇 IP: `192.168.123.161` (고정)  
PC IP: `192.168.123.x` 대역으로 수동 설정 (예: `192.168.123.100`)

```bash
# 인터페이스 이름 확인
ip link show

# 예시: enp3s0 또는 enx00e04c680e0f (USB-LAN 어댑터)
sudo ip addr add 192.168.123.100/24 dev enp3s0
sudo ip link set enp3s0 up

# 연결 확인
ping 192.168.123.161
```

### config 파일 수정

`config/g1/g1_29dof_falcon.yaml`:

```yaml
# sim2sim: "lo"
# sim2real: 실제 이더넷 인터페이스 이름으로 변경
INTERFACE: "enp3s0"   # 또는 "enx00e04c680e0f" 등
```

---

## 로봇 켜기 & Debugging Mode 진입

### 1. 로봇 전원 ON

1. 로봇 등 쪽 전원 버튼 **길게 눌러** 켜기
2. 부팅 완료까지 약 30~60초 대기 (발화음 들림)
3. 로봇이 기본 서 있는 자세로 전환됨

### 2. Low-level Control Mode (Debugging Mode) 진입

> **반드시** 이 모드로 진입해야 policy joint command가 적용됨

**조이스틱(Unitree WirelessController) 사용:**

```
L2 + R2  →  동시에 누르기  →  로봇이 앉음  →  Low-level 모드 진입
```

확인 방법: 로봇이 `L2+R2` 입력 후 바닥에 앉으면 성공.

> [!WARNING]
> Low-level 모드에서는 로봇이 자체 균형 제어를 **하지 않음**.  
> policy가 실행 중이 아니면 로봇이 쓰러짐. 반드시 사람이 옆에 있어야 함.

### 3. 로봇 준비 자세

policy 실행 전 로봇을 손으로 기립 자세로 잡아주거나, `i` 키(init state)로 천천히 default 자세로 이동시킨 후 `]` 키로 policy 시작.

---

## 시뮬레이션 테스트 (Sim2Sim)

실물 배포 전 MuJoCo 시뮬레이터로 검증.

> **INTERFACE는 `"lo"` (루프백)로 설정된 상태에서 진행**

`cd sim2real` 후 실행.

### Original FALCON (오픈소스, dim=575)

```bash
# 터미널 1: MuJoCo 환경 실행
python sim_env/loco_manip.py \
  --config=config/g1/g1_29dof_falcon.yaml

# 터미널 2: Policy 실행
python rl_policy/loco_manip/loco_manip.py \
  --config=config/g1/g1_29dof_falcon.yaml \
  --model_path=models/falcon/g1_29dof.onnx
```

> `use_balance_descriptor: false` 확인 (config 기본값)

### BD Policy (Balance Descriptor, dim=578)

`config/g1/g1_29dof_falcon.yaml`에서:

```yaml
use_balance_descriptor: true
```

```bash
# 터미널 1: MuJoCo 환경 실행
python sim_env/loco_manip.py \
  --config=config/g1/g1_29dof_falcon.yaml

# 터미널 2: Policy 실행 (BD onnx 사용)
python rl_policy/loco_manip/loco_manip.py \
  --config=config/g1/g1_29dof_falcon.yaml \
  --model_path=models/falcon/g1_29dof.onnx
```

BD가 정상 작동 중이면 policy 터미널에 아래와 같이 출력됨:

```
[BD] u_D=(0.000,0.000) dbar_norm=0.000 delta_bar=0.0000 dp_mag=0.0002
[INFER] step=1  obs_dim=(1, 578)  BD=[0. 0. 0.]
```

> 정지 상태에서 BD obs가 모두 0인 것은 정상 — 외란이 없으면 ZMP ≈ CoM이므로 BD obs=0.

---

## 실물 배포 (Sim2Real)

> [!IMPORTANT]
> - sim2real에서는 **MuJoCo 환경을 실행하지 않음**. policy만 실행.
> - `INTERFACE`를 실제 이더넷 인터페이스로 변경했는지 반드시 확인.
> - 로봇이 Low-level 모드 진입 상태인지 확인 후 실행.

`cd sim2real` 후 실행.

### 1. Original FALCON (오픈소스, dim=575)

`config/g1/g1_29dof_falcon.yaml`:
```yaml
INTERFACE: "enp3s0"          # 실제 인터페이스로 변경
use_balance_descriptor: false
```

```bash
python rl_policy/loco_manip/loco_manip.py \
  --config=config/g1/g1_29dof_falcon.yaml \
  --model_path=models/falcon/g1_29dof.onnx
```

### 2. BD Policy (Balance Descriptor, dim=578)

`config/g1/g1_29dof_falcon.yaml`:
```yaml
INTERFACE: "enp3s0"          # 실제 인터페이스로 변경
use_balance_descriptor: true
```

```bash
python rl_policy/loco_manip/loco_manip.py \
  --config=config/g1/g1_29dof_falcon.yaml \
  --model_path=models/falcon/g1_29dof.onnx
```

### 배포 순서 체크리스트

```
[ ] 로봇 전원 ON 및 부팅 완료
[ ] L2+R2로 Low-level 모드 진입 (로봇 앉음 확인)
[ ] 이더넷 연결 및 ping 192.168.123.161 확인
[ ] config의 INTERFACE가 올바른 인터페이스로 설정됨
[ ] config의 use_balance_descriptor가 onnx와 일치 (dim 확인)
[ ] 로봇 기립 자세로 손으로 받쳐줌
[ ] policy 실행
[ ] i 키로 init state 이동 후 ] 키로 policy 시작
```

---

## 키보드 / 조이스틱 명령어

> 모든 키보드 명령은 **policy 터미널**에 포커스된 상태에서 입력.

### Policy 제어 (기본)

| 키보드 | 조이스틱 | 동작 |
|--------|----------|------|
| `]` | `start` | Policy 시작 (두 번째 누르면 속도 0으로 제자리 서기) |
| `o` | `B+Y` | Policy 정지 (joint command 0) |
| `i` | `A+X` | Init state: 천천히 default 자세로 이동 |

### 이동 명령

| 키보드 | 동작 |
|--------|------|
| `=` | 걷기/서기 전환 (stand_command 토글) |
| `w` | 전진 속도 +0.1 m/s |
| `s` | 전진 속도 -0.1 m/s |
| `a` | 좌측 속도 +0.1 m/s |
| `d` | 우측 속도 +0.1 m/s |
| `q` | 좌회전 -0.1 rad/s |
| `e` | 우회전 +0.1 rad/s |
| `z` | 속도 초기화 |
| `m` | 전진 0.5 m/s + 걷기 모드 (즉시 이동) |

### 자세 제어

| 키보드 | 조이스틱 | 동작 |
|--------|----------|------|
| `1` | `B+up` | Base height +0.1 m |
| `2` | `B+down` | Base height -0.1 m |
| `,` | `select+left` | Waist yaw -0.2 rad |
| `.` | `select+right` | Waist yaw +0.2 rad |
| — | `Y+up` | Waist pitch -0.1 rad |
| — | `Y+down` | Waist pitch +0.1 rad |

### Gain 조정

| 키보드 | 조이스틱 | 동작 |
|--------|----------|------|
| `4` | `Y+left` | kp scale -0.1 |
| `7` | `Y+right` | kp scale +0.1 |
| `5` | `A+left` | kp scale -0.01 |
| `6` | `A+right` | kp scale +0.01 |
| `0` | `A+Y` 또는 `A+B` | kp scale 1.0 (초기화) |

### MuJoCo 뷰어 키 (Sim2Sim 전용)

| 키 | 동작 |
|----|------|
| `3` | +x 방향 충격력 (20N, 0.4s) |
| `4` | -x 방향 충격력 (20N, 0.4s) |
| `9` | Elastic band 토글 |
| `backspace` | 시뮬레이션 리셋 |

---

## 비상 정지

| 방법 | 동작 |
|------|------|
| 키보드 `o` | policy action 정지 |
| 조이스틱 `B+Y` | policy action 정지 |
| `Ctrl+C` | policy 프로세스 종료 |
| 조이스틱 `L1+L2` | 로봇 긴급 정지 (Unitree 내장) |

---

## Troubleshooting

**DDS 연결 안 됨**
- `INTERFACE` 값 확인 (`ip link show`로 인터페이스 이름 조회)
- `ping 192.168.123.161` 확인

**obs dim mismatch (RuntimeError)**
- `use_balance_descriptor` 설정과 onnx 모델 input dim이 불일치
- `false` → dim=575 onnx, `true` → dim=578 onnx

**BD obs가 실물에서 폭발적으로 커짐 (h_dot_ns spike)**
- `bd_ema_alpha` 값을 0.1→0.05로 낮춰 smoothing 강화
- `bd_landing_freeze_n` 값을 3→5로 늘려 touchdown spike 억제

**로봇이 policy 시작 직후 쓰러짐**
- kp gain을 `5`/`6` 키로 낮춰서 시작 (예: `5` 연타로 0.5 수준에서 시작)
- init state(`i`)로 default 자세 먼저 잡은 후 policy 시작(`]`)
