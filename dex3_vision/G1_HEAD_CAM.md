# 실물 G1 헤드 카메라 → Quest 2 (1인칭 영상)

로봇이 실제로 보는 영상을 Quest 2에 띄우는 절차. XR 텔레옵 자체(팔 조작, 데드맨 스위치, 컨트롤러 매핑, Dex3 손)는
저장소 루트의 `XR_TELEOP.md`를 볼 것 — 이 문서는 **카메라 영상 경로만** 다룬다.

**2026-07-29에 이 로봇(PC2)에서 직접 설치·검증 완료.** 아래 절차는 `viewer/G1_HEAD_CAM.md`(teleimager
공식 README 기준으로만 작성됐던 미검증 버전)를 실제로 따라가며 부딪힌 문제들을 전부 반영한 것.

## 0. 전제 조건

- 조작 PC와 G1을 랜선(이더넷)으로 직결했고, 그 상태로 FALCON policy(`loco_manip_xr.py` 등)가 이미
  DDS로 로봇을 정상 제어하고 있다. (즉 조작 PC ↔ G1 내부 네트워크 통신은 이미 뚫려 있음 —
  카메라 연결에서 네트워크를 새로 설정할 필요 없음.)
- `XR_TELEOP.md` 절차대로 `~/.config/xr_teleoperate/cert.pem`, `key.pem`이 조작 PC에 이미 있다.
- G1의 **PC2**(Development Computing Unit — 로봇 몸체에 내장된 온보드 컴퓨터, 저수준
  실시간 컨트롤러와는 별개. 헤드 카메라가 물리적으로 여기 USB로 연결돼 있음)에 SSH로 접속 가능하다.
  기본 IP는 `192.168.123.164`, 기본 계정 `unitree`/`123`.
- **이 로봇의 헤드 카메라는 Intel RealSense D435i**다 (뎁스 카메라지만 RGB 컬러 스트림은 일반
  UVC 웹캠처럼도 노출됨 — 아래 2.4에서 이 방식으로 씀, `pyrealsense2` 빌드 불필요).

```bash
# 조작 PC에서 확인
ping 192.168.123.164          # PC2가 응답하는지
ssh unitree@192.168.123.164   # 접속 가능한지
```

## 1. 아키텍처

```
G1 헤드 카메라 (PC2에 USB로 연결, Intel RealSense D435i)
    ↓
PC2에서 teleimager의 image_server 실행 (venv, 매번 수동 실행 — 자동시작 설정 안 함)
    ├─ ZMQ (JPEG)  → 조작 PC의 ImageClient → RobotHeadCamSource.read_bgr() → render_to_xr() → Quest 2
    └─ WebRTC      → Quest 2 브라우저가 PC2에서 영상을 직접 당겨감 (조작 PC를 안 거침, 지연 더 낮음)
```

시뮬레이션 경로(`sim_env --head_cam` → 공유메모리 → 헤드셋)는 이 작업과 무관하게 **그대로 동작한다**
(`--head_cam_source`의 기본값이 `sim`).

프레임 형상(해상도, 좌우 스테레오 여부)은 코드에 하드코딩하지 않고, 접속 시점에 PC2의
`cam_config_server.yaml`에서 그대로 받아와 televuer 디스플레이를 거기에 맞춘다. 이 로봇의 헤드캠은
**단안(mono) 640x480**이라 스테레오가 아님 (G1 순정 스테레오 헤드캠 480x1280과는 다름 — RealSense로
교체된 개체).

| 파일 | 내용 |
| --- | --- |
| `sim2real/utils/head_cam_source.py` | `SimHeadCamSource`(공유메모리) / `RobotHeadCamSource`(teleimager ZMQ) — 둘 다 `img_shape`, `binocular`, `read_bgr()` 제공 |
| `sim2real/rl_policy/loco_manip/loco_manip_xr.py` | `--head_cam_source`로 소스 선택, 카메라 해상도·양안 여부를 이미지 서버에서 받아 televuer 디스플레이를 그에 맞춰 구성 |

---

## 2. PC2(G1 온보드 컴퓨터) 설정 — 최초 1회

### 2.1 SSH 접속 후 teleimager 설치

**이 PC2엔 conda가 없다.** 시스템 Python(3.8.10, Ubuntu 20.04 기본)로 충분 — teleimager는
`python>=3.8,<3.11`만 요구하므로 conda 설치할 필요 없이 venv만 쓰면 됨.

```bash
ssh unitree@192.168.123.164

# PC2 안에서 (venv 모듈이 없으면 먼저: sudo apt install -y python3.8-venv)
python3 -m venv ~/teleimager-venv
source ~/teleimager-venv/bin/activate
python -m pip install --upgrade pip     # 시스템 기본 pip는 pyproject.toml만 있는 editable install을 못 함

sudo apt install -y libusb-1.0-0-dev libturbojpeg-dev

git clone https://github.com/unitreerobotics/teleimager.git
cd teleimager
pip install -e ".[server]"     # 서버(카메라 캡처 + WebRTC)까지 필요하므로 [server] 포함
pip install psutil             # [server] extra에 안 잡혀있지만 image_server.py가 import함
```

> `xr_teleoperate/README.md`는 `silencht/teleimager` 포크를 언급하지만, 위 공식 저장소
> (`unitreerobotics/teleimager`)를 써도 된다. 이 리포의 `xr_teleoperate/teleop/teleimager`는
> **클라이언트 쪽(조작 PC)** 코드가 이미 서브모듈로 들어있는 것이고, PC2에는 이렇게 별도로
> 새로 클론한다.

**PC2에서 인터넷이 막혀있을 수 있음 (학교 와이파이 캡티브 포털):** 이 PC2는 `eth0`(로봇 내부망,
게이트웨이 없음)와 `wlan0`(학교 게스트 와이파이 `KUWIFI_GUEST`) 둘 다 붙어있는데, `KUWIFI_GUEST`는
캡티브 포털이 있어서 헤드리스 상태로는 브라우저 인증을 통과할 수 없다. `git clone`/`pip install`
할 때 DNS/TLS가 이상하게 실패하면 — 휴대폰 핫스팟을 켜고 `nmcli`로 임시 연결해서 설치만 하고, 끝나면
다시 `KUWIFI_GUEST`로 되돌리면 된다 (카메라 스트리밍 자체는 `192.168.123.x` 내부망만 쓰므로 평소엔
인터넷이 전혀 필요 없음):

```bash
sudo nmcli device wifi rescan ifname wlan0
nmcli device wifi list ifname wlan0
sudo nmcli device wifi connect "<핫스팟 SSID>" password "<비번>" ifname wlan0
# ... 설치 끝나면 ...
sudo nmcli connection up KUWIFI_GUEST
sudo nmcli connection delete <핫스팟 SSID>   # 임시 프로필 정리
```

### 2.2 카메라 장치 권한

```bash
# PC2, teleimager 리포 루트에서
bash setup_uvc.sh
```
`unitree` 계정을 `video` 그룹에 추가해준다 — **적용하려면 SSH 세션을 나갔다가 다시 접속해야 함**
(`exit` 후 재접속).

### 2.3 인증서 복사 (WebRTC 쓸 경우 필수)

Quest 2는 두 개의 서로 다른 HTTPS 엔드포인트에 접속한다 — 조작 PC의 televuer(8012)와 PC2의
teleimager WebRTC(기본 60001). 각자 같은 인증서가 있어야 한다:

```bash
# 조작 PC에서
scp ~/.config/xr_teleoperate/cert.pem ~/.config/xr_teleoperate/key.pem \
    unitree@192.168.123.164:~/teleimager

# PC2에서
mkdir -p ~/.config/xr_teleoperate/
cp ~/teleimager/cert.pem ~/teleimager/key.pem ~/.config/xr_teleoperate/
```

### 2.4 카메라 탐색 및 `cam_config_server.yaml` 작성

```bash
# PC2, teleimager 리포 루트, venv 활성화 상태에서
python -m teleimager.image_server --cf
```

**`--rs`(RealSense 전용 SDK) 플래그는 쓰지 않는다** — `pyrealsense2`를 소스 빌드해야 해서 오래
걸리는데, RealSense의 RGB 컬러 스트림은 일반 UVC 카메라로도 잡히므로 그럴 필요가 없다. `--cf`만
돌리면 RealSense가 내놓는 여러 `/dev/videoN` 중 "RGB로 보이는" 후보들이 나오는데, **teleimager의
휴리스틱이 뎁스(Z16)/적외선(GREY 등) 스트림까지 RGB로 오탐지하니 그대로 믿지 말고 직접
`v4l2-ctl --device=/dev/videoN --list-formats-ext`로 각 후보의 실제 픽셀 포맷을 확인할 것.**
`YUYV`가 진짜 컬러, `Z16`은 뎁스, `GREY`/`Y8I`는 스테레오 IR.

이 로봇의 경우: `video0`=Z16(뎁스), `video2`=GREY/UYVY/Y8I(IR), **`video4`=YUYV(진짜 RGB, 640x480@30fps
지원)** — 그래서 `video_id: 4`.

```yaml
head_camera:
  enable_zmq: true
  zmq_port: 55555
  enable_webrtc: true
  webrtc_port: 60001
  webrtc_codec: h264

  # 중요: RealSense RGB 스트림은 MJPG를 지원 안 하고 YUYV만 지원함.
  # type: uvc는 pyuvc로 MJPG 모드를 정확히 매칭해야 해서 여기선 무조건 실패
  # (에러 로그도 없이 카메라 스레드가 조용히 안 뜸 — ZMQ/WebRTC 포트 자체가 안 열림).
  # type: opencv는 순정 cv2.VideoCapture라 카메라가 주는 포맷을 그대로 받아써서 문제없음.
  type: opencv

  image_shape: [480, 640]   # [height, width], 단안이라 640이 그대로 폭
  binocular: false
  fps: 30

  video_id: 4
  serial_number: null        # 플레이스홀더 값을 null로 안 바꾸면 그게 먼저 매칭 시도되다 실패함
  physical_path: null
```

왼쪽/오른쪽 손목 카메라는 없으므로 `left_wrist_camera`/`right_wrist_camera`의 `enable_zmq`/
`enable_webrtc`를 `false`로 둔다.

### 2.5 자동 시작 — 설정 안 함 (의도적)

`setup_autostart.sh`로 부팅 시 자동 실행되게 할 수도 있지만, **이 로봇은 여러 연구실이 공유하는
장비라 실험할 때마다 수동으로 켜고 끄기로 결정함.** 매번 아래 2.6처럼 SSH 들어가서 직접 실행할 것.

### 2.6 image_server 실행 (매번, 실험 시작할 때)

```bash
# PC2에서
source ~/teleimager-venv/bin/activate
cd ~/teleimager
python -m teleimager.image_server
```

로그에 `[OpenCVCamera: head_camera] initialized with 480x640 @ 30 FPS`와
`[Image Server] head_camera is ready.`가 뜨는지 확인. 이 터미널은 실험 끝날 때까지 그대로 두고
(포그라운드로 계속 돎), 끝나면 `Ctrl+C`로 종료.

---

## 3. 조작 PC 설정 — 최초 1회

```bash
conda activate fcreal
cd xr_teleoperate/teleop/teleimager
pip install -e . --no-deps
```

확인:
```bash
python -c "from teleimager.image_client import ImageClient; print('OK')"
```

---

## 4. 실행 절차 (매번)

### 4.1 PC2에서 image_server 실행

2.6 그대로. 별도 SSH 세션으로 계속 띄워둔다.

### 4.2 조작 PC에서 카메라 링크만 먼저 확인 (정책·헤드셋 없이)

```bash
cd dex3_vision/sim2real   # 또는 dex3_vision (모듈 경로에 따라)
python -m sim2real.utils.head_cam_source --source robot --img_server_ip 192.168.123.164
```

연결되면 해상도/양안 여부/WebRTC 가능 여부가 출력되고 OpenCV 창에 실제 카메라 영상이 뜬다.
**여기가 안 되면 헤드셋 쪽도 절대 안 되므로 항상 이걸 먼저 통과시킬 것.** (`q`로 창 닫기)

### 4.3 조작 PC에서 FALCON policy + XR + 헤드캠 + Dex3 실행

```bash
# dex3_vision/sim2real 안에서
python rl_policy/loco_manip/loco_manip_xr.py \
  --config=config/g1/g1_29dof_dex3_real.yaml \
  --model_path=models/falcon/g1_29dof.onnx \
  --cert_file=$HOME/.config/xr_teleoperate/cert.pem \
  --key_file=$HOME/.config/xr_teleoperate/key.pem \
  --xr_mode controller --dex3 \
  --display_mode immersive --head_cam --head_cam_source robot \
  --img_server_ip 192.168.123.164
```

로그에 `head camera stream connected (source=robot, ...)`가 **초록색**으로 뜨는지, 노란 경고
("no frames yet")가 안 뜨는지 확인.

> `immersive` 모드는 헤드셋을 쓰면 모니터/키보드가 안 보인다. **로봇을 먼저 세워둔 뒤** 헤드셋을
> 착용할 것. 비상정지는 오른손 **B**. (로봇 세우는 절차·안전수칙은 `XR_TELEOP.md`,
> `falcon_experiment/sim2real/README.md` 참고)

### 4.4 Quest 2에서 인증서 신뢰 + 접속

헤드셋의 Meta Quest 브라우저에서 **반드시 이 순서로**:

**a) WebRTC 카메라 서버부터 먼저 (WebRTC 쓰는 경우만, `--head_cam_transport`가 `zmq`가 아니면 필요)**
```
https://192.168.123.164:60001
```
경고 화면이 뜨면 **고급 → 안전하지 않음(unsafe) 계속 진행**. 페이지가 뜨면 **start** 버튼을
눌러 카메라 미리보기가 나오는지 확인. **한 번 신뢰해두면 같은 인증서인 한 다음부턴 생략 가능.**

**b) 그 다음 실제 텔레옵 화면**
```
https://<조작 PC IP>:8012/?ws=wss://<조작 PC IP>:8012
```
(조작 PC IP는 `ifconfig`로 확인) 여기도 경고 → 계속 진행 → **Enter VR**. 로봇 1인칭 시점 영상이
(단안으로) 보이면 성공.

---

## 5. 추가된 CLI 옵션 (`loco_manip_xr.py`)

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--head_cam_source` | `sim` | `sim`=Mujoco 공유메모리(기존 동작), `robot`=실물 G1 헤드 카메라 |
| `--head_cam_transport` | `auto` | `auto`는 서버가 WebRTC를 켜뒀으면 WebRTC 우선, 아니면 ZMQ. `zmq`/`webrtc`로 강제 가능 |
| `--img_server_ip` | `192.168.123.164` | image_server가 도는 PC2 IP |
| `--img_server_port` | `60000` | 카메라 설정 요청 포트 |
| `--dex3` | 꺼짐 | Dex3 손가락 open/close (이 문서와 무관, `XR_TELEOP.md` 참고) |

`--head_cam_width/height`는 이제 **sim 소스 전용**이다(실물 해상도는 서버가 알려줌).

---

## 6. 트러블슈팅 체크리스트

- **PC2에서 `git clone`/`pip install`이 DNS/TLS 에러로 실패** → `KUWIFI_GUEST`(학교 게스트
  와이파이) 캡티브 포털에 막힌 것. 위 2.1의 핫스팟 우회 절차 참고. (`ping 8.8.8.8`이 처음엔
  "Destination Host Unreachable"이면 라우팅 우선순위 문제, 그다음 응답 없음/무응답이면 방화벽,
  `curl https://아무데나`가 엉뚱한 인증서를 주면 캡티브 포털 리다이렉트다.)
- **`ModuleNotFoundError: No module named 'psutil'`** → `[server]` extra에 안 잡힌 의존성,
  `pip install psutil`로 따로 설치.
- **`image_server` 실행은 되는데 ZMQ(55555)/WebRTC(60001) 포트가 전혀 안 열림, 에러도 없음** →
  `cam_config_server.yaml`의 `type: uvc`인데 카메라가 MJPG를 지원 안 하는 경우 (RealSense RGB는
  YUYV만 지원). `type: opencv`로 바꿀 것. `sudo ss -tlnp | grep <포트>`로 실제 바인딩 여부 확인.
- **`--cf`가 뎁스/IR 스트림도 "RGB"라고 보여줌** → teleimager의 오탐지, `v4l2-ctl
  --list-formats-ext`로 직접 픽셀 포맷 확인 후 진짜 컬러(YUYV/MJPG) 스트림의 `video_id` 선택.
- **4.2가 안 됨** → PC2에서 `image_server`가 실제로 떠 있는지, 방화벽에서 ZMQ 포트(기본
  55555)/WebRTC 포트(60001)가 막혀있지 않은지 확인. `ping 192.168.123.164`로 네트워크부터 재확인.
- **4.3 로그에 노란 경고("no frames yet")** → teleimager의 requester는 서버가 응답 없으면
  로컬에 캐시된 `cam_config_client.yaml`로 조용히 폴백한다. 그래서 "연결 성공"처럼 보이는데
  영상만 안 온다. `image_server`가 진짜 살아있는지, IP가 맞는지 재확인.
- **헤드셋 화면이 계속 까맣게 나옴** → 4.4-a(WebRTC 인증서 신뢰)를 건너뛰고 바로 4.4-b로 가면
  WebRTC offer fetch가 조용히 실패한다. **반드시 60001부터 먼저 방문.**
- 카메라가 없거나 서버에 못 붙어도 **정책 자체는 그대로 뜬다** (영상만 빠짐). 팔/손 조작은 계속
  동작하니 급하면 카메라 없이 진행하고 나중에 붙여도 된다.

---

## 7. 구현 세부 (참고용)

- **WebRTC vs ZMQ**: WebRTC면 프레임이 조작 PC 프로세스를 안 거치므로 지연이 낮고 50Hz 정책
  루프에 부담이 없다. ZMQ는 JPEG 디코드가 정책 프로세스에서 일어나 CPU를 조금 더 쓴다.
- **색 순서**: `render_to_xr()`가 항상 BGR→RGB 변환을 하므로 소스는 전부 BGR로 넘긴다.
  teleimager는 원래 BGR이라 그대로, Mujoco는 RGB라 `[:, :, ::-1]`로 뒤집어서 넘긴다.
- **중복 프레임 제거**: 카메라 30fps < 정책 루프 50Hz라 같은 프레임이 반복 조회된다.
  `RobotHeadCamSource`가 같은 프레임이면 `None`을 돌려줘서 헛된 렌더를 건너뛴다.

## 8. 검증 상태

- sim 경로: 공유메모리 → BGR 변환 → `render_to_xr()` 왕복 확인
- **robot 경로: 2026-07-29, 이 로봇(PC2 + RealSense D435i)에서 실물 하드웨어로 직접 검증 완료** —
  PC2 설치(venv 기반), 카메라 탐색, `cam_config_server.yaml` 작성, `image_server` 기동,
  조작 PC의 `head_cam_source.py --source robot`로 실제 프레임(480x640x3 BGR) 수신까지 확인.
  (아직 안 한 것: Quest 2 헤드셋에 실제로 띄워서 보는 것 — 4.3/4.4 단계는 다음에 이어서 검증.)
