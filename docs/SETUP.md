# 체스 로봇팔 — 학교 노트북 현장 세팅 가이드

실물 하드웨어(서보·PCA9685·카메라·전원)가 모두 준비된 상태에서,
학교 노트북에 **처음부터** 세팅해 로봇팔을 작동시키는 전체 절차.

---

## 0. USB에 담아갈 파일 ⭐

`git clone`을 하면 **코드는 다 받아지지만**, 아래 파일들은 `.gitignore`로 제외되어
**따라오지 않으므로 USB로 직접 가져가야** 한다.

```
USB/
├── stockfish/
│   └── stockfish-windows-x86-64-avx2.exe   ← 체스 엔진 (필수)
│
├── models/                                  ← RL 보정 모델 (선택, 있으면 정밀도↑)
│   ├── stage2_final.zip                     ← main.py가 찾는 파일
│   └── stage2_vecnorm.pkl                   ← (sim 데모용, 실전엔 불필요)
│
└── offline_backup/                          ← (선택) 인터넷 없을 때 대비
    ├── python-3.11.x-amd64.exe              ← Python 설치파일
    └── wheels/                              ← pip 패키지 오프라인 (아래 설명)
```

**무엇을 왜:**
| 파일 | 출처 | 필수? |
|---|---|---|
| `stockfish .exe` | stockfishchess.org | ✅ 필수 (없으면 로봇이 수를 못 둠) |
| `stage2_final.zip` | 구글드라이브 `MyDrive/chess_robot/correction_model/` | ⬜ 선택 (없어도 IK만으로 동작) |
| `stage2_vecnorm.pkl` | 위와 동일 | ⬜ sim 데모용 |
| `calibration.json` | **가져가지 말 것** | ❌ 현장에서 새로 생성 (카메라 위치마다 다름) |

> 💡 인터넷이 막힌 학교라면 `offline_backup`에 Python 설치파일과
> `pip download -r requirements.txt -d wheels` 로 받은 휠을 담아가면 오프라인 설치 가능.
> (집에서 미리: `pip download -r requirements.txt -d USB\offline_backup\wheels`)

---

## 1. 노트북 기본 환경 설치

### 1-1. Python
python.org → 3.10~3.12 → 설치 시 **"Add Python to PATH" 체크 필수**
```bash
python --version
```

### 1-2. Git
git-scm.com → 기본 설정 설치

### 1-3. Arduino IDE
arduino.cc/en/software → 설치 →
라이브러리 매니저에서 **`Adafruit PWM Servo Driver Library`** 설치 (의존 라이브러리도 함께)

---

## 2. 코드 받기

```bash
cd C:\Users\사용자
git clone https://github.com/2025kuhshw-design/chess_robotarm.git
cd chess_robotarm
git checkout claude/chess-robot-arm-LIojd
git pull origin claude/chess-robot-arm-LIojd
```

---

## 3. 가상환경 + 패키지 설치

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
> 매 작업마다 `venv\Scripts\activate` 먼저. torch 설치라 시간 걸림.
>
> 오프라인이면: `pip install --no-index --find-links USB\offline_backup\wheels -r requirements.txt`

---

## 4. USB 파일 배치

```bash
# RL 모델 (있으면)
mkdir models\correction_model
copy USB\models\stage2_final.zip   models\correction_model\
copy USB\models\stage2_vecnorm.pkl models\correction_model\

# Stockfish는 그냥 USB 경로 그대로 써도 되고, 노트북에 복사해도 됨
```

---

## 5. 아두이노 업로드

1. USB로 Uno 연결
2. Arduino IDE에서 `hardware/arduino/servo_control.ino` 열기
3. 툴 → 보드: **Arduino Uno**, 포트: **COM?** 선택
4. 업로드(→) → 시리얼 모니터(9600)에서 `READY` 확인
5. **시리얼 모니터 창은 닫기** (안 닫으면 Python이 포트 못 엶)

> 포트 번호: 장치 관리자 → 포트(COM and LPT) → `Arduino Uno (COM?)`

---

## 6. 하드웨어 배선

```
Uno 5V   → PCA VCC (로직 전원)
Uno A4   → PCA SDA
Uno A5   → PCA SCL
Uno GND  → PCA GND
외부 6V(+) → PCA V+ 터미널
외부 6V(−) → PCA GND 터미널   ← Uno GND와 공통 접지

PCA ch0 → joint1 서보(베이스)   ch4 → 펌프
PCA ch1 → joint2 서보(어깨)     ch5 → 밸브
PCA ch2 → joint3 서보(팔꿈치)
```

⚠️ **체크:**
- 서보 전원은 반드시 외부 6V 8A에서 (Uno 5V 핀 금지)
- 공통 GND 빠지면 서보가 떨리거나 안 움직임
- 펌프 정격이 3A 넘으면 PCA 채널 헤더 말고 외부전원 직결

---

## 7. 단계별 점검 (한 번에 켜지 말 것)

> 모든 명령 전 `venv\Scripts\activate`

```bash
# 7-1. 하드웨어 없이 로직 (전부 PASS여야 함)
python test_pipeline.py --test ik
python test_pipeline.py --test lagrange
python test_pipeline.py --test full

# 7-2. 시리얼 통신 (Uno 연결, 시리얼모니터 닫은 상태)
python test_pipeline.py --test serial --port COM3

# 7-3. RL 모델 (가져왔으면)
python test_pipeline.py --test rl
```

---

## 8. 카메라 캘리브레이션 (현장 필수)

카메라를 **보드 정중앙 위 40~60cm, 똑바로 아래** 보게 단단히 고정 후:

```bash
python vision\calibrate.py --camera 0
```
1. 체스판 **8×8 플레이 영역의 4모서리**를 클릭 (테두리 X)
2. 순서: **좌상→우상→우하→좌하** (시계방향)
3. 좌상 미리보기에 뜨는 "← ROBOT a1" 힌트대로,
   **로봇에서 가장 가까운 기준칸(a1)을 좌상으로**
4. `s` 키 → `vision/calibration.json` 저장

### 좌표 검증 (회전/반전 확인)
```bash
python main.py --mode vision --camera 0
```
- 각 칸에 좌표(a1~h8) 라벨이 뜸, **a1은 마젠타**
- **로봇 a1 칸에 기물 하나** 올림 → 화면 마젠타 a1 칸에 떠야 정상
- 옆칸(b1)에 올렸을 때 화면도 오른쪽 한 칸이면 축 정상
  (아래로 가면 방향 뒤집힘 → 캘리브 반대방향 재클릭)
- 흰/검 인식 안 맞으면 `vision/detect.py`의
  `PIECE_VAR_THRESH`(빈칸/기물) → `WHITE_THRESH`/`BLACK_THRESH`(색) 순으로 조정

---

## 9. 서보/흡착 방향 튜닝 (팔 조립 전 권장)

시리얼 모니터 또는 sim 명령으로 확인:
```
A90,90,90,0      → 홈(모든 서보 중앙). 팔이 수직이면 정상
A0,0,0,0         → 한쪽 끝 (떨림/발열 시 .ino의 SERVO_MIN/MAX_PWM 조정)
A180,180,180,0   → 반대 끝
A90,90,90,1      → 흡착 ON (펌프 작동)
A90,90,90,0      → 흡착 OFF (밸브 열림=해제)
```
- 서보 회전 방향이 반대면 `hardware/arm_controller.py:69-72`의 부호(+/−) 수정
- 흡착이 반대로 동작하면 `.ino`의 `VALVE_HOLD`/`VALVE_RELEASE` 값 교체

---

## 10. 최종 실행

```bash
venv\Scripts\activate
python main.py --mode real --port COM3 --stockfish H:\학교\체스로봇팔\stockfish\stockfish-windows-x86-64-avx2.exe --camera 0
```

- 사람(WHITE) 차례 → 수 두고 Enter → 카메라가 인식
- 로봇(BLACK) 차례 → Stockfish 계산 → IK → (모델 있으면)RL 보정 → 집기→이동→놓기→홈복귀

> 🛑 **첫 실전 안전수칙**: 전원 차단 스위치를 손에 두고, 서보 느린 속도로,
> 처음엔 RL 보정 끄고(모델 빼고) IK만으로 검증 후 켜기.

---

## 빠른 체크리스트

```
[집에서 미리]
□ USB: stockfish.exe / stage2_final.zip / (오프라인 휠)
□ Python·Git·Arduino IDE 설치 + Adafruit 라이브러리

[학교에서]
□ git clone + checkout + pull
□ venv + pip install -r requirements.txt
□ USB 파일 배치 (models/correction_model/)
□ .ino 업로드 → READY
□ 배선 (서보 외부전원 + 공통 GND)
□ test: ik / lagrange / full / serial  전부 PASS
□ 카메라 캘리브레이션 + --mode vision 좌표 검증
□ 서보/흡착 방향 튜닝
□ main.py --mode real  실전
```

가장 빠뜨리기 쉬운 것: **USB의 stage2_final.zip(모델)** 과 **현장 calibration.json(새로 생성)**.
