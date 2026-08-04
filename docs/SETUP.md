# 체스 로봇팔 — 세팅 & 운영 가이드

노트북에 처음부터 세팅해 로봇팔을 작동시키는 전체 절차 + 도구 사용법 + 문제 해결.

---

## 0. 💻 하드웨어 없이(컴퓨터만) 할 수 있는 것

로봇팔·아두이노가 없어도 아래는 전부 동작한다. 집에서 미리 해두면 현장 시간을 아낀다.

```bash
# 로직 검증 (하드웨어 불필요)
python test_pipeline.py --test ik          # 역기구학↔순기구학 왕복 오차
python test_pipeline.py --test lagrange    # 토크 계산·한계 확인
python test_pipeline.py --test full        # sim 모드 3수 전체 파이프라인
python test_pipeline.py --test rl          # RL 보정 모델 로드 + 64칸 추론

# 시뮬레이션 게임 (시리얼 명령이 화면에 출력됨, stockfish 필요)
python main.py --mode sim --stockfish <stockfish.exe 경로>

# 정적분 발표용 운동 데이터 (명령 궤적만 — 하드웨어 없이)
python analysis/collect_motion.py --sim --from-sq e2 --to-sq e4 --dt 0.05 \
    --out analysis/logs/sim_dt50.csv
python analysis/integrate.py --csv analysis/logs/sim_dt50.csv
```

| 가능 ✅ | 불가능 ❌ (하드웨어 필요) |
|---|---|
| IK/FK·토크·전체 파이프라인 검증 | 실제 팔 구동, 흡착 |
| sim 모드 게임 (명령 출력 확인) | 시리얼 통신 테스트 |
| RL 모델 로드·추론 | 카메라 캘리브레이션·인식 |
| 구분구적 적분·수렴 분석 | 실측 이동거리 측정 |
| 코드 수정·상수 조정 | 서보 캘리브레이션 |

---

## 1. USB에 담아갈 파일 ⭐

`git clone`으로 코드는 받아지지만, 아래는 `.gitignore` 제외라 **USB로 직접** 가져가야 한다.

```
USB/
├── stockfish/
│   └── stockfish-windows-x86-64-avx2.exe   ← 체스 엔진 (필수)
├── models/
│   ├── stage2_final.zip                     ← RL 보정 모델 (main.py가 로드)
│   └── stage2_vecnorm.pkl                   ← (sim 데모용, 실전 불필요)
└── offline_backup/                          ← (선택) 인터넷 막힌 경우
    ├── python-3.12.x-amd64.exe
    └── wheels/
```

| 파일 | 출처 | 필수? |
|---|---|---|
| `stockfish .exe` | stockfishchess.org | ✅ 필수 |
| `stage2_final.zip` | 구글드라이브 `MyDrive/chess_robot/correction_model/` | ⬜ 선택(없으면 IK만) |
| `calibration.json` | **가져가지 말 것** | ❌ 현장에서 새로 생성 |

> 오프라인 대비: 집에서 `pip download -r requirements.txt -d USB\offline_backup\wheels`

---

## 2. 노트북 기본 환경

### 2-1. Python — ⚠️ 버전 주의
**Python 3.11 또는 3.12를 쓸 것. 3.13/3.14는 안 된다.**
최신 버전은 numpy·pybullet 등의 사전빌드(wheel)가 없어 소스 컴파일을 시도하고, MSVC 빌드 도구가 없으면 설치가 통째로 실패한다.

- 3.11/3.12 최신 릴리스 페이지엔 .exe가 없을 수 있다(보안수정 단계). 이럴 땐:
  ```bash
  winget install Python.Python.3.12
  ```
  또는 python.org에서 **.exe가 있는** 예전 3.12 릴리스를 받는다.
- 설치 시 **"py launcher" 체크**

```bash
py -3.12 --version     # Python 3.12.x 확인
```

### 2-2. Git / Arduino IDE
- git-scm.com → 기본 설치
- arduino.cc → 설치 후 라이브러리 매니저에서 **`Adafruit PWM Servo Driver Library`** 설치

---

## 3. 코드 받기

```bash
cd C:\Users\사용자
git clone https://github.com/2025kuhshw-design/chess_robotarm.git
cd chess_robotarm
git checkout claude/chess-robot-arm-LIojd
git pull origin claude/chess-robot-arm-LIojd
```

---

## 4. 가상환경 + 패키지

```bash
py -3.12 -m venv venv          # ⚠️ 반드시 버전 명시 (python -m venv 금지)
venv\Scripts\activate
python --version               # 3.12.x 확인!
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> - 매 작업마다 `venv\Scripts\activate` 먼저
> - `pybullet`은 `requirements-sim.txt`로 분리됨 (시뮬 전용, 실제 작동엔 불필요)
> - 오프라인: `pip install --no-index --find-links USB\offline_backup\wheels -r requirements.txt`

설치 확인:
```bash
python -c "import serial, stable_baselines3, cv2, chess, chess.engine, numpy, scipy; print('전부 OK')"
```

---

## 5. USB 파일 배치

```bash
mkdir models\correction_model
copy USB경로\stage2_final.zip    models\correction_model\
copy USB경로\stage2_vecnorm.pkl  models\correction_model\
```
> 폴더명 **`correction_model`** 정확히. 틀리면 모델을 못 찾고 IK만으로 동작한다.
> Stockfish는 위치 자유 — 실행 시 `--stockfish <경로>`로 지정.

---

## 6. 아두이노 업로드

1. USB로 Uno 연결
2. Arduino IDE에서 `hardware/arduino/servo_control.ino` 열기
3. 툴 → 보드: **Arduino Uno**, 포트: **COM?**
4. 업로드 → 시리얼 모니터(9600)에서 `READY` 확인
5. **시리얼 모니터 창 닫기** (안 닫으면 Python이 포트를 못 연다)

> `.ino`를 수정했으면 **반드시 재업로드**. 파이썬 파일만 바뀐 경우는 불필요.
> 포트 번호: 장치 관리자 → 포트(COM and LPT)

---

## 7. 하드웨어 배선

```
[로직 — Uno ↔ PCA9685]
Uno 5V   → PCA VCC          (로직 전원)
Uno A4   → PCA SDA
Uno A5   → PCA SCL
Uno GND  → PCA GND

[서보 전원]
외부 6V(+) → PCA V+ 터미널
외부 6V(−) → PCA GND 터미널

[PCA 채널]
ch0 → joint1 서보 (베이스)
ch1 → joint2 서보 (어깨)
ch2 → joint3 서보 (팔꿈치)
ch4 → 펌프 모듈 신호
ch5 → 밸브 모듈 신호

[⭐ 흡착기 — 전원은 반드시 직결]
펌프/밸브 모듈 신호(주황) → PCA ch4 / ch5 의 신호핀
펌프/밸브 모듈 V+(빨강)   → 외부 6V(+) 에 직접   ← PCA 거치지 말 것!
펌프/밸브 모듈 GND(갈색)  → 공통 GND
```

### ⚠️ 치명적 주의사항
- **펌프 V+를 PCA 채널 헤더에 꽂으면 안 된다.** 펌프 인러시(순간 수 A)가 PCA의 얇은
  전원 트레이스를 통과해 **전압 강하 → 전체 brownout → 드라이버 손상**으로 이어진다.
  (실제로 이 방식으로 드라이버가 한 번 파손됨)
- **공통 GND 필수**: 전원GND · PCA GND · Uno GND · 흡착모듈 GND 전부 연결.
  직결이어도 GND가 안 이어지면 신호 기준이 없어 모듈이 안 켜진다.
- 6V 레일에 **대용량 전해 커패시터(1000µF↑)**를 달면 펌프 기동 시 전압강하 완충에 도움.
- 서보 전원은 반드시 외부 6V에서 (Uno 5V 핀 금지).

### 흡착 모듈에 대해
펌프/밸브 스위치 모듈(FZ006류)은 **서보 3선(신호/V+/GND) 입력**이며,
상시 HIGH가 아니라 **서보 펄스**로 켜고 끈다. `.ino`가 그렇게 보내도록 되어 있다.
- ON  = `DEV_ON_PULSE`  (≈2.4ms)
- OFF = `DEV_OFF_PULSE` (≈0.5ms)
- 반대로 동작하면 `.ino`의 이 두 값을 서로 바꾼다.

---

## 8. 점검 순서 (한 번에 켜지 말 것) 🛑

> 모든 명령 전 `venv\Scripts\activate`. 펌프 테스트 시 **전원 스위치를 손에** 둘 것.

```bash
# 8-1. 하드웨어 없이 로직 (전부 PASS여야 함)
python test_pipeline.py --test ik
python test_pipeline.py --test lagrange
python test_pipeline.py --test full

# 8-2. 통신만 (Uno 연결, 6V는 아직 꺼도 됨, 시리얼모니터 닫은 상태)
python test_pipeline.py --test serial --port COM5

# 8-3. 흡착기 단독 ⭐ (서보 3개는 PCA에서 뽑고 → 부하 격리)
python hardware/test_suction.py --port COM5

# 8-4. 서보 개별 (흡착 OK 후 서보 다시 연결)
python hardware/servo_jog.py --port COM5

# 8-5. 동시 부하 / 문제 격리
python hardware/diagnose.py --port COM5

# 8-6. RL 모델
python test_pipeline.py --test rl
```

---

## 9. 도구 레퍼런스

### 9-1. `servo_jog.py` — 서보 조그 (캘리브레이션용)
```bash
python hardware/servo_jog.py --port COM5
python hardware/servo_jog.py --port COM5 --step 1 --step-delay 0.05   # 더 천천히
python hardware/servo_jog.py --port COM5 --max-angle 200              # 180 초과 필요시
```
| 입력 | 동작 |
|---|---|
| `90 90 90` | 세 서보를 각각 그 각도로 |
| `s1 120` | 1번 서보만 120으로 |
| `suction 1` / `suction 0` | 흡착 ON / OFF |
| `q` | 종료 (**마지막 각도 그대로 유지**) |

- 목표까지 `--step`도씩 **부드럽게** 이동한다(충격·스톨 방지).
- 🛑 **긁는 소리가 나면 즉시 `Ctrl+C` → 전원 차단.** 스톨이며 기어가 갈린다.

### 9-2. `test_suction.py` — 흡착기 단독 테스트
```bash
python hardware/test_suction.py --port COM5
```
| 입력 | 동작 |
|---|---|
| `on` / `off` | 흡착 ON / OFF |
| `pulse 3` | ON 후 3초 뒤 자동 OFF |
| `q` | 종료 (자동 OFF) |

> 서보를 PCA에서 분리하고 하면 전원 문제를 격리할 수 있다.

### 9-3. `diagnose.py` — 하드웨어 격리 진단
```bash
python hardware/diagnose.py --port COM5
```
부품을 하나씩만 작동시켜 어디서 전원이 죽는지 좁혀낸다.
통신 → 서보 개별 → 펌프 단독 → 펌프+서보 동시 순으로 테스트하고 원인을 요약해준다.

### 9-4. `analysis/` — 정적분 발표용 운동 데이터
```bash
# 수집 (--sim이면 하드웨어 없이 명령 궤적만)
python analysis/collect_motion.py --port COM5 --from-sq e2 --to-sq e4 \
    --vmax 0.05 --accel 0.10 --dt 0.05 --out analysis/logs/run_dt50.csv

# 적분 + 실측 비교 + 수렴 시연
python analysis/integrate.py --csv analysis/logs/run_dt50.csv --measured-cm 5.3
```
- 사다리꼴 속도 프로파일(가속-등속-감속)로 이동하며 `(t, 관절각, 말단xyz, 속도)`를 CSV 기록
- 구분구적 4종(왼쪽/오른쪽/중점/사다리꼴) 적분 → 예측거리, 자 실측값과 비교, dt 수렴 표 출력
- ⚠️ **로그의 위치·속도는 "명령값"**이다. MG996R은 엔코더가 없어(개방루프) 실제 도달각을
  읽을 수 없다. "실제 이동거리"는 반드시 **자로 측정**해서 비교할 것.

---

## 10. 캘리브레이션

### 10-1. 카메라 (현장 필수)
카메라를 **보드 정중앙 위 40~60cm, 똑바로 아래** 보게 고정 후:
```bash
python vision\calibrate.py --camera 0
```
1. 체스판 **8×8 플레이 영역의 4모서리** 클릭 (테두리 X)
2. 순서: **좌상 → 우상 → 우하 → 좌하** (시계방향)
3. 좌상이 **로봇에서 가장 가까운 기준칸(a1)**이 되도록
4. 미리보기에 체스판이 **정사각형 가득** 떠야 정상 (까맣게 나오면 코너를 잘못 찍은 것 → `r`로 초기화)
5. `s` 키 → `vision/calibration.json` 저장

> 키가 안 먹으면: **한/영을 영문으로** + **영상 창을 클릭해 활성화**

좌표 검증:
```bash
python main.py --mode vision --camera 0
```
- 각 칸에 좌표(a1~h8) 라벨, **a1은 마젠타**
- 로봇 a1 칸에 기물 → 화면 마젠타 a1에 떠야 정상
- 옆칸(b1)에 놓았을 때 화면도 오른쪽 한 칸이면 축 정상 (아래로 가면 방향 뒤집힘 → 재클릭)

**인식 임계값 조정** (`vision/detect.py`)
게임은 "빈 칸 ↔ 찬 칸"만 사용하므로 **흰/검 색 오인식은 무방**하다.
- 빈 칸인데 기물로 인식 → `PIECE_VAR_THRESH` **올리기**
- 기물인데 빈 칸으로 인식 → `PIECE_VAR_THRESH` **내리기**
- 색 라벨까지 맞추려면 `WHITE_THRESH` / `BLACK_THRESH`

> 빛 반사(창문·조명)가 최대 적. 직사광을 피하고 무광 표면을 쓰면 인식이 크게 좋아진다.

### 10-2. 서보 (혼 재장착 없이 소프트웨어로)
`hardware/servo_jog.py`로 아래 3가지를 찾아 `hardware/arm_controller.py`에 넣는다.

**① 안전 각도 범위 → `SERVO_SAFE_MIN` / `SERVO_SAFE_MAX`**
각 관절을 조금씩 움직여 테이블·구조물에 **닿기 직전**까지. 5° 여유를 둔다.

**② 기준자세 → `SERVO1_HOME` / `SERVO2_HOME` / `SERVO3_HOME`**
| 관절 | IK 기준자세(q=0) |
|---|---|
| s1 베이스 | 팔이 **체스판 정면**을 똑바로 가리킴 |
| s2 어깨 | 상완이 **테이블과 수평** |
| s3 팔꿈치 | 전완이 상완과 **일직선**(쭉 편 상태) |

각도를 키웠을 때 관절이 반대로 돌면 해당 `SERVO*_DIR`의 부호를 뒤집는다.

**③ Z자 park 자세 → `PARK_POSE`**
게임 시작/휴식 시 팔이 접혀 있을 각도. 카메라 시야를 안 가리고 토크 부하도 줄인다.

---

## 11. 조정 상수 한눈에

### `hardware/arm_controller.py`
| 상수 | 의미 |
|---|---|
| `SERVO1/2/3_HOME` | IK 기준자세(q=0)에 해당하는 서보 각도 |
| `SERVO1/2/3_DIR` | 관절 회전 방향 (+1 / −1) |
| `SERVO_SAFE_MIN/MAX` | 관절별 소프트 각도 제한 (충돌 방지) |
| `PARK_POSE` | 시작/휴식 Z자 자세 |
| `RAMP_STEP_DEG` / `RAMP_DELAY` | 이동 부드러움 (작을수록 부드럽고 느림) |
| `LIFT_HEIGHT` / `LIFT_MIN` | 기물 들어올리는 높이 (먼 칸은 자동 축소) |

### `hardware/arduino/servo_control.ino`
| 상수 | 의미 |
|---|---|
| `CH_JOINT1/2/3` | 서보 채널 (0/1/2) |
| `CH_PUMP` / `CH_VALVE` | 펌프 ch4 / 밸브 ch5 |
| `SERVO_MIN_PWM` / `SERVO_MAX_PWM` | 서보 펄스 범위 (떨림·발열 시 조정) |
| `ANGLE_HARD_MAX` | 확장 최대각 (기본 200) |
| `DEV_ON_PULSE` / `DEV_OFF_PULSE` | 흡착 모듈 ON/OFF 펄스 (반대면 교체) |
| `VALVE_HOLD` / `VALVE_RELEASE` | 밸브 동작 (반대면 교체) |

### `utils/ik_solver.py`
| 상수 | 의미 |
|---|---|
| `L1_DEFAULT` 0.140 / `L2_DEFAULT` 0.155 / `L3_DEFAULT` 0.075 | 링크 길이 (m) |
| `BOARD_ORIGIN_X/Y` | 체스판 원점(a1 코너) 위치 |
| `CELL_SIZE` 0.029125 | 한 칸 크기 (m) |

---

## 12. 최종 실행

```bash
venv\Scripts\activate
python main.py --mode real --port COM5 ^
    --stockfish C:\...\stockfish\stockfish-windows-x86-64-avx2.exe --camera 0
```

- 사람(WHITE) 차례 → **실제로 말을 옮기고** Enter → 카메라가 인식
  - ⚠️ `e2e4` 같은 **글자를 치는 게 아니다.** 물리적으로 옮겨야 한다.
  - 첫 Enter는 현재 상태를 기준으로 저장하는 용도다.
- 로봇(BLACK) 차례 → Stockfish → IK → (모델 있으면) RL 보정 → 집기→이동→놓기→park 복귀
- `--mode real`은 카메라 창을 띄우지 않는다. 인식을 눈으로 보려면 별도 창에서 `--mode vision`.

> 🛑 첫 실전: 전원 차단 스위치를 손에, 처음엔 RL 보정 없이 IK만으로 검증 후 켜기.

---

## 13. 트러블슈팅

### 설치·환경
| 증상 | 원인 / 해결 |
|---|---|
| `No module named 'serial'` | `pip install pyserial` (패키지명 ≠ 임포트명) |
| `pybullet` 빌드 실패 (MSVC 요구) | 시뮬 전용이라 불필요. `requirements.txt`에서 이미 분리됨 |
| numpy/기타 빌드 실패 | **Python 3.13/3.14 사용 중.** 3.11/3.12로 venv 재생성 |
| `UnicodeDecodeError: cp949` (pip) | requirements 파일에 한글 주석 → 영문으로 (해결됨) |
| venv가 `pythoncore-3.14` 못 찾음 | base Python을 삭제함. `py -3.12 -m venv venv`로 재생성 |
| 패키지가 계속 없다고 뜸 | `pip install -r requirements.txt`가 중간에 실패해 이후가 안 깔린 것. `Successfully installed`로 끝나는지 확인 |

### 시리얼·아두이노
| 증상 | 원인 / 해결 |
|---|---|
| `PermissionError` (액세스 거부) | **Arduino IDE 시리얼 모니터가 포트 점유 중** → 닫기 |
| 응답 `'READY'` | 리셋 시 초기 메시지. `reset_input_buffer()`로 처리됨(수정 완료) |
| 응답 `''` (빈 값) | `setup()`에서 멈춤 = **PCA9685와 I2C 통신 실패.** VCC/SDA/SCL/GND 배선 확인 |
| 각도가 몇 초 뒤 원래대로 | 예전 타임아웃 자동복귀. 현재 코드는 제거됨 → **재업로드** 필요 |

### 전원·구동
| 증상 | 원인 / 해결 |
|---|---|
| 펌프 켜자 **서보·흡착 전부 힘빠짐** (Uno/PCA LED는 켜짐) | 6V 레일 brownout. **펌프 V+를 외부전원 직결**로, 커패시터 추가 |
| 서보에서 **칠판 긁는 소리** | 스톨(막힌 상태로 계속 밀어붙임) → **즉시 전원 차단.** 기어 손상 원인 |
| 서보가 힘없이 떨림 | 외부 6V가 PCA **V+ 터미널**에 안 들어옴 / 공통 GND 누락 |
| MG996R인데 플라스틱 기어 | 클론 제품일 가능성. 정품은 전 금속 기어 |

### 소프트웨어 동작
| 증상 | 원인 / 해결 |
|---|---|
| `ValueError: 도달 불가능한 좌표` | 리프트 높이가 팔 길이 초과. 적응형 `_safe_lift`로 해결됨 |
| RL 모델 로드 시 `unsupported operand type(s) for *` | 학습 환경과 버전 불일치. `PPO.load(custom_objects=...)`로 해결됨 |
| RL 로드 중 조용히 죽음(트레이스백 없음) | Python 3.14 등 부적합 버전. 3.11/3.12로 |
| `이동 감지 실패 (사라짐=[], 나타남=[])` | **말을 실제로 안 옮김.** 물리적으로 옮기고 Enter |
| 캘리브 미리보기가 까맣게 | 코너를 잘못/순서 틀리게 클릭 → `r`로 초기화 후 재클릭 |
| 불법 이동이라고 거부됨 | 대부분 보드 상태 오독. 대문자=백, 소문자=흑 |

---

## 14. 빠른 체크리스트

```
[집에서 미리]
□ Python 3.11 또는 3.12 설치 (3.13+ 금지) + py launcher
□ Git · Arduino IDE + Adafruit PWM Servo Driver 라이브러리
□ USB: stockfish.exe / stage2_final.zip / (오프라인 휠)
□ 컴퓨터만으로: test ik / lagrange / full / rl 전부 PASS 확인

[현장 — 순서대로]
□ git clone + checkout + pull
□ py -3.12 -m venv venv → activate → pip install -r requirements.txt
□ USB 파일 배치 (models/correction_model/)
□ 배선: 로직(VCC/SDA/SCL/GND) + 서보전원(V+) + ⭐펌프 V+ 직결 + 공통 GND
□ .ino 업로드 → READY → 시리얼 모니터 닫기
□ test serial → OK
□ test_suction.py (서보 뽑고) → 펌프 on/off, 전원 안 죽는지
□ servo_jog.py → 서보 개별 확인, 안전범위 파악
□ 서보 캘리브레이션 (SERVO*_HOME, SAFE_MIN/MAX, PARK_POSE)
□ 카메라 캘리브레이션 + --mode vision 좌표 검증
□ diagnose.py → 동시 부하 확인
□ main.py --mode real  실전
```

가장 빠뜨리기 쉬운 것: **USB의 stage2_final.zip**, **현장 calibration.json(새로 생성)**,
**펌프 V+ 직결**, **공통 GND**.
