# 실물 G1 헤드 카메라 → Quest 2 (1인칭 영상)

로봇이 실제로 보는 영상을 Quest 2에 띄우는 절차. XR 텔레옵 자체(팔 조작, 데드맨 스위치, 컨트롤러 매핑)는 저장소 루트의
`XR_TELEOP.md`를 볼 것 — 이 문서는 **카메라 영상 경로만** 다룬다.

## 0. 전제 조건

- 조작 PC와 G1을 랜선(이더넷)으로 직결했고, 그 상태로 FALCON policy(`loco_manip_xr.py` 등)가 이미
  DDS로 로봇을 정상 제어하고 있다. (즉 조작 PC ↔ G1 내부 네트워크 통신은 이미 뚫려 있음 —
  카메라 연결에서 네트워크를 새로 설정할 필요 없음.)
- `XR_TELEOP.md` 절차대로 `~/.config/xr_teleoperate/cert.pem`, `key.pem`이 조작 PC에 이미 있다.
- G1의 **PC2**(Development Computing Unit — 로봇 몸체에 내장된 온보드 컴퓨터, 저수준
  실시간 컨트롤러와는 별개. 헤드 카메라가 물리적으로 여기 USB로 연결돼 있음)에 SSH로 접속 가능하다.
  기본 IP는 `192.168.123.164`.

```bash
# 조작 PC에서 확인
ping 192.168.123.164          # PC2가 응답하는지
ssh unitree@192.168.123.164   # 접속 가능한지
```

## 1. 아키텍처

```
G1 헤드 카메라 (PC2에 USB로 연결)
    ↓
PC2에서 teleimager의 image_server 실행
    ├─ ZMQ (JPEG)  → 조작 PC의 ImageClient → RobotHeadCamSource.read_bgr() → render_to_xr() → Quest 2
    └─ WebRTC      → Quest 2 브라우저가 PC2에서 영상을 직접 당겨감 (조작 PC를 안 거침, 지연 더 낮음)
```

시뮬레이션 경로(`sim_env --head_cam` → 공유메모리 → 헤드셋)는 이 작업과 무관하게 **그대로 동작한다**
(`--head_cam_source`의 기본값이 `sim`).

프레임 형상(해상도, 좌우 스테레오 여부)은 코드에 하드코딩하지 않고, 접속 시점에 PC2의
`cam_config_server.yaml`에서 그대로 받아와 televuer 디스플레이를 거기에 맞춘다. G1 기본 헤드
카메라는 480x1280 양안이라, 맞게 설정돼 있어야 헤드셋에서 스테레오로 제대로 보인다.

| 파일 | 내용 |
| --- | --- |
| `sim2real/utils/head_cam_source.py` | `SimHeadCamSource`(공유메모리) / `RobotHeadCamSource`(teleimager ZMQ) — 둘 다 `img_shape`, `binocular`, `read_bgr()` 제공 |
| `sim2real/rl_policy/loco_manip/loco_manip_xr.py` | `--head_cam_source`로 소스 선택, 카메라 해상도·양안 여부를 이미지 서버에서 받아 televuer 디스플레이를 그에 맞춰 구성 |

---

## 2. PC2(G1 온보드 컴퓨터) 설정 — 최초 1회

### 2.1 SSH 접속 후 teleimager 설치

```bash
ssh unitree@192.168.123.164

# PC2 안에서
conda create -n teleimager python=3.10 -y
conda activate teleimager
sudo apt install -y libusb-1.0-0-dev libturbojpeg-dev

git clone https://github.com/unitreerobotics/teleimager.git
cd teleimager
pip install -e ".[server]"     # 서버(카메라 캡처 + WebRTC)까지 필요하므로 [server] 포함
```

> `xr_teleoperate/README.md`는 `silencht/teleimager` 포크를 언급하지만, 위 공식 저장소
> (`unitreerobotics/teleimager`)를 써도 된다. 이 리포의 `xr_teleoperate/teleop/teleimager`는
> **클라이언트 쪽(조작 PC)** 코드가 이미 서브모듈로 들어있는 것이고, PC2에는 이렇게 별도로
> 새로 클론한다.

### 2.2 카메라 장치 권한

```bash
# PC2, teleimager 리포 루트에서
bash setup_uvc.sh
```
(스크립트가 없는 버전이면 생략 가능 — 대신 카메라 접근 시 permission 에러가 나면 `sudo`로
`teleimager-server`를 실행해서 확인.)

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
# PC2, teleimager 리포 루트에서
python -m teleimager.image_server --cf
# RealSense도 있으면: python -m teleimager.image_server --cf --rs
```

출력에서 헤드 카메라에 해당하는 `video_id` / `serial_number` / `physical_path`와 지원 포맷
(예: `480x1280@30 MJPG`)을 확인한 뒤 `cam_config_server.yaml`의 `head_camera` 항목을 채운다:

```yaml
head_camera:
  enable_zmq: true
  zmq_port: 55555
  enable_webrtc: true
  webrtc_port: 60001
  webrtc_codec: h264
  type: uvc                 # 탐색 결과에 맞게: opencv / realsense / uvc
  image_shape: [480, 1280]  # 실제 카메라 해상도 (양안이면 폭이 두 눈 합친 값)
  binocular: true           # 좌우 스테레오면 true, 단안이면 false
  fps: 30
  video_id: 2               # 탐색 결과의 video_id (또는 serial_number / physical_path)
  serial_number: null
  physical_path: null
```

여러 식별자 중 하나만 쓰면 되는데, **재부팅/재연결에도 안 바뀌는 게 좋으면 `physical_path`**,
간단하게 하려면 `video_id`를 쓴다 (자세한 트레이드오프는 teleimager README 4.1절 참고).

왼쪽/오른쪽 손목 카메라가 없으면 `left_wrist_camera`/`right_wrist_camera`의
`enable_zmq`/`enable_webrtc`를 `false`로 둬도 무방 — 이 통합은 `head_camera`만 사용한다.

### 2.5 (선택) 부팅 시 자동 시작

```bash
# PC2, teleimager 리포 루트에서
bash setup_autostart.sh
```
설정해두면 PC2가 켜질 때마다 수동으로 `image_server`를 실행할 필요가 없다. 스크립트가 없는
버전이면 생략하고 매번 2.6처럼 수동 실행.

### 2.6 image_server 실행 확인 (자동시작 안 했다면 매번)

```bash
# PC2에서
python -m teleimager.image_server
```

---

## 3. 조작 PC 설정 — 최초 1회

```bash
conda activate fcreal
cd xr_teleoperate/teleop/teleimager
pip install -e .
```

확인:
```bash
python -c "from teleimager.image_client import ImageClient; print('OK')"
```

---

## 4. 실행 절차 (매번)

### 4.1 PC2에서 image_server 실행 (2.5에서 자동시작 설정했으면 생략)
```bash
ssh unitree@192.168.123.164 "cd teleimager && python -m teleimager.image_server"
```
또는 PC2에 별도 터미널로 접속해서 실행.

### 4.2 조작 PC에서 카메라 링크만 먼저 확인 (정책·헤드셋 없이)

```bash
cd viewer
python -m sim2real.utils.head_cam_source --source robot --img_server_ip 192.168.123.164
```

연결되면 해상도/양안 여부/WebRTC 가능 여부가 출력되고 OpenCV 창에 실제 카메라 영상이 뜬다.
**여기가 안 되면 헤드셋 쪽도 절대 안 되므로 항상 이걸 먼저 통과시킬 것.** (`q`로 창 닫기)

### 4.3 조작 PC에서 FALCON policy + XR + 헤드캠 실행

```bash
# viewer/sim2real 안에서
python rl_policy/loco_manip/loco_manip_xr.py \
  --config=config/g1/g1_29dof_falcon_xr.yaml \
  --model_path=models/falcon/g1_29dof.onnx \
  --cert_file=$HOME/.config/xr_teleoperate/cert.pem \
  --key_file=$HOME/.config/xr_teleoperate/key.pem \
  --display_mode immersive --head_cam --head_cam_source robot \
  --img_server_ip 192.168.123.164
```

로그에 `head camera stream connected (source=robot, ...)`가 **초록색**으로 뜨는지, 노란 경고
("no frames yet")가 안 뜨는지 확인.

> `immersive` 모드는 헤드셋을 쓰면 모니터/키보드가 안 보인다. **로봇을 먼저 세워둔 뒤** 헤드셋을
> 착용할 것. 비상정지는 오른손 **B**. (로봇 세우는 절차는 `XR_TELEOP.md` 참고)

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
(조작 PC IP는 `ifconfig`로 확인, 보통 `192.168.123.2`) 여기도 경고 → 계속 진행 →
**Enter VR**. 로봇 1인칭 시점 영상이 스테레오로 보이면 성공.

---

## 5. 추가된 CLI 옵션 (`loco_manip_xr.py`)

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--head_cam_source` | `sim` | `sim`=Mujoco 공유메모리(기존 동작), `robot`=실물 G1 헤드 카메라 |
| `--head_cam_transport` | `auto` | `auto`는 서버가 WebRTC를 켜뒀으면 WebRTC 우선, 아니면 ZMQ. `zmq`/`webrtc`로 강제 가능 |
| `--img_server_ip` | `192.168.123.164` | image_server가 도는 PC2 IP |
| `--img_server_port` | `60000` | 카메라 설정 요청 포트 |

`--head_cam_width/height`는 이제 **sim 소스 전용**이다(실물 해상도는 서버가 알려줌).

---

## 6. 트러블슈팅 체크리스트

- **4.2가 안 됨** → PC2에서 `image_server`가 실제로 떠 있는지, 방화벽에서 ZMQ 포트(기본
  55555)/WebRTC 포트(60001)가 막혀있지 않은지 확인 (`sudo ufw allow 55555`, `sudo ufw allow 60001`
  on PC2). `ping 192.168.123.164`로 네트워크부터 재확인.
- **4.3 로그에 노란 경고("no frames yet")** → teleimager의 requester는 서버가 응답 없으면
  로컬에 캐시된 `cam_config_client.yaml`로 조용히 폴백한다. 그래서 "연결 성공"처럼 보이는데
  영상만 안 온다. `image_server`가 진짜 살아있는지, IP가 맞는지 재확인.
- **헤드셋 화면이 계속 까맣게 나옴** → 4.4-a(WebRTC 인증서 신뢰)를 건너뛰고 바로 4.4-b로 가면
  WebRTC offer fetch가 조용히 실패한다. **반드시 60001부터 먼저 방문.**
- **양안이어야 하는데 한쪽 눈에만 나오거나 이미지가 찌그러짐** → `cam_config_server.yaml`의
  `binocular`/`image_shape`가 실제 카메라와 다른 것. 2.4의 `--cf` 탐색 결과와 다시 대조.
- 카메라가 없거나 서버에 못 붙어도 **정책 자체는 그대로 뜬다** (영상만 빠짐). 팔 조작은 계속
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
- robot 경로: **목(mock) image_server 기준으로만 검증** — config 협상, 양안/단안 형상 반영,
  BGR 프레임 전달, 중복 프레임 제거, WebRTC/ZMQ 선택, 서버 불통 시 무크래시 폴백
- **실물 G1 하드웨어에서는 아직 미검증** — 이 문서의 PC2 설치 절차(2.1~2.5)도 teleimager
  공식 README 기준으로 작성한 것이지 실제로 PC2에서 실행해 확인한 것은 아님. 진행하며 막히는
  부분이 있으면 이 문서를 업데이트할 것.
