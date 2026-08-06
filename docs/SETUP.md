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
python main.py --mode sim --stockfish C:\Users\hjh20\chess_robotarm\stockfish\stockfish.exe

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
│   └── (RL 모델은 재학습 후에 넣는다 — 9-A-2 참고)
└── offline_backup/                          ← (선택) 인터넷 막힌 경우
    ├── python-3.12.x-amd64.exe
    └── wheels/
```

| 파일 | 출처 | 필수? |
|---|---|---|
| `stockfish .exe` | stockfishchess.org | ✅ 필수 |
| `stage2_final.zip` + `stage2_vecnorm.pkl` | 구글드라이브 `MyDrive/chess_robot/correction_model/` | ⭕ 재학습 완료본 사용 |
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
4. 업로드 → 시리얼 모니터(9600)에서 **`READY fw=3`** 확인
5. **시리얼 모니터 창 닫기** (안 닫으면 Python이 포트를 못 연다)

> 포트 번호: 장치 관리자 → 포트(COM and LPT)

### 언제 재업로드해야 하나
| 바뀐 파일 | 조치 |
|---|---|
| `hardware/arduino/servo_control.ino` | `git pull` **+ 재업로드** |
| 그 외 전부 (`.py`, `.md`) | `git pull` 만 |

아두이노에는 `.ino` 하나만 올라간다. 파이썬은 노트북에서 도는 것이라 무관하다.

### 펌웨어 버전 자동 확인
`.ino`를 고칠 때마다 `FW_VERSION`이 올라가고, 파이썬이 연결 시 대조한다.

```
[ArmController] 펌웨어 fw=3 ✓                 ← 정상, 그냥 진행
```
```
⚠️ 아두이노 펌웨어가 최신이 아닙니다 (감지: 2 / 필요: 3)
   Arduino IDE에서 ... 반드시 다시 업로드하세요.        ← 재업로드 필요
```
**이 한 줄만 확인하면 된다.** 구버전은 부팅 시 서보를 90,90,90으로 전속력
이동시켜 기어가 손상될 수 있다.

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

# 8-5. 칸 단위 동작 (서보 캘리브레이션 끝난 뒤)
python hardware/test_square.py --port COM5 --step 1 --step-delay 0.1

# 8-6. 동시 부하 / 문제 격리
python hardware/diagnose.py --port COM5

# 8-7. RL 모델
python test_pipeline.py --test rl
```

---

## 8-A. ⭐ 기동 절차 — 팔을 움직이는 모든 작업의 시작

**서보에는 위치 센서가 없다.** 그래서 코드는 팔이 지금 어디 있는지 모른다.
아두이노는 부팅 시 서보를 잡지 않으므로(슬램 방지) 팔은 **중력으로 처져 있고**,
그 상태에서 아무 위치나 명령하면 서보가 **자기 최대 속도로 튄다**(기어 손상).
`--step`/`--step-delay`로도 이 **첫 동작만은 늦출 수 없다**(출발점을 모르므로).

그래서 `test_square.py`와 `main.py --mode real`은 시작 시 기동 절차를 거친다.

### 가장 쉬운 방법
1. **팔을 손으로 Z자(PARK) 자세로 잡는다**
2. `engage` 입력 → "지금 PARK에 있다"고 선언. 실제와 맞으니 **안 움직인다**
3. 손을 뗀다 (이제 서보가 잡고 있음)
4. `start` → 완료

### 명령
| 명령 | 동작 |
|---|---|
| `engage` | 팔이 Z자(PARK)에 있다고 선언 + 그 값 전송 (**안 움직임**) |
| `engage 30 150 175` | 다른 각도에 있으면 그 값으로 선언 |
| `s2 145` | 그 관절만 천천히 이동 |
| `40 130 170` | 세 관절 모두 천천히 이동 |
| `park` | PARK 자세로 천천히 (세 관절 동시) |
| `start` | 세팅 완료 → 진행 |
| `q` | 중단 |

> ⚠️ `engage` 값이 실제와 다르면 그 차이만큼 튄다. 크게 움직였다면 값이 틀린
> 것이니, **멈춘 그 위치를 다시 `engage`** 하면 된다(서보가 그 위치를 잡고 있으므로).
>
> 팔이 이미 PARK에 잡혀 있으면 `--no-startup`으로 건너뛸 수 있다.

---

## 8-B. 이동 속도 조절 (`--step`, `--step-delay`)

서보는 명령받은 위치로 자기 최대 속도로 간다. 늦추는 유일한 방법은
**중간 위치를 잘게 나눠 보내는 것**이다.

| 옵션 | 뜻 |
|---|---|
| `--step` | 한 번에 움직일 **각도(도)**. 크면 빠르고 거칠다 |
| `--step-delay` | 스텝 사이 **대기(초)**. 크면 느리고 부드럽다 |

| 설정 | 속도 | 60° 이동 |
|---|---|---|
| `--step 1 --step-delay 0.1` | 8°/초 | 7.5초 (테스트용, 아주 안전) |
| `--step 2 --step-delay 0.05` | 27°/초 | 2.3초 (**기본값**) |
| `--step 3 --step-delay 0.03` | 55°/초 | 1.1초 (검증 끝난 뒤 실전용) |

- 캘리브레이션·첫 테스트는 느리게, 동작이 검증되면 기본값으로.
- `--step`이 5를 넘으면 램프 효과가 사라진다.
- 시리얼 왕복(약 25ms)이 항상 끼므로 delay를 0으로 해도 무한정 빨라지진 않는다.
- **긁는 소리·떨림이 나면 즉시 낮출 것.**

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

### 9-2A. `test_square.py` — 칸 단위 동작 테스트 ⭐
조그가 '서보 각도'를 다룬다면 이 도구는 '**체스 좌표**'를 다룬다.
서보 캘리브레이션과 체스판 배치가 맞는지 검증하는, 실전 게임 직전 단계.

```bash
python hardware/test_square.py --port COM5 --step 1 --step-delay 0.1
```
시작하면 **기동 절차**(8-A)를 먼저 거친다.

| 입력 | 동작 |
|---|---|
| `e5` | e5 칸 **위 안전높이**로 이동 |
| `down` / `up` | 현재 칸으로 하강 / 상승 |
| `pick e7` | 집기 (위→하강→흡착ON→**흡착 유지한 채** 상승) |
| `place e5` | 놓기 (흡착 유지한 채 이동·하강→흡착OFF→상승) |
| `move e7 e5` | 게임과 **동일한 전체 시퀀스** |
| `suction 1` / `0` | 흡착만 토글 (**팔은 안 움직임**) |
| `zoff 8` | 하강 깊이를 8mm로 조정 (아래 참고) |
| `set 85 140 180` | 현재 위치 가정만 교정 (전송 안 함) |
| `a 85 140 185` | 서보 각도로 직접 이동 (한 축만 바꿔 격리 진단) |
| `limit` | 소프트 제한 보기 / 임시 변경 (`limit max s3 200`) |
| `probe s3 180 200` | 그 관절의 **실제 기계 한계**를 5°씩 올려가며 찾는다 |
| `home` | PARK 자세로 복귀 |
| `q` | 종료 (팔은 현재 위치 유지) |

**하강 깊이 `zoff`** — 흡착컵은 고무라 기물에 **살짝 눌려야** 진공이 걸린다.
기물 윗면보다 얼마나 더 내려갈지를 mm로 지정한다(기본 4mm).
```
e5 → down → (덜 눌리면) zoff 8 → down → (여전히면) zoff 12 → down
```
적당한 값을 찾으면 `hardware/arm_controller.py`의 `TOUCH_PRESS`에 넣어 고정.
어깨높이·링크길이 실측 오차도 이 값이 함께 흡수한다.

> ⚠️ 랭크1~2(사람 진영)는 현재 배치에서 팔이 닿지 않는다. 랭크3~8로 테스트할 것.

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

## 9-A. 정확도 보정 — 두 가지 방법

이름이 비슷해 헷갈리기 쉬운데 **성격이 완전히 다르다.**

| | ① 실측 보정 | ② 시뮬 RL 재학습 |
|---|---|---|
| 도구 | `hardware/calibrate_board.py` | `sim/train_correction.py` |
| 무엇을 학습하나 | **실제 로봇의 오차** | 해석적 IK ↔ PyBullet 동역학의 차이 |
| 실제 정확도 개선 | ✅ **직접적** | ❓ 보장 없음 (실물을 본 적 없음) |
| 걸리는 시간 | **20분** | 30분~수시간 |
| 하드웨어 필요 | 필요 | 불필요 (컴퓨터만) |
| 방법 | 실측 → 최소제곱 회귀 | 강화학습(PPO) |

> **정확도를 올리는 건 ①이다.** ②는 "강화학습을 적용했다"는 서술과 시뮬 지표를 위한 것.
> 보고서에는 **둘 다 하고 비교**하는 구성이 가장 정직하고 내용도 좋다.
> "시뮬 RL만으로는 실제 오차가 줄지 않아 실측 보정이 필요했다"는 발견 과정 자체가 좋은 재료다.

### 9-A-1. 실측 보정 ⭐ (정확도가 가장 크게 오름)

**카메라를 쓰지 않는다.** 카메라로 재면 카메라 캘리브레이션 오차가 측정에 섞인다.
대신 **체스판에 인쇄된 격자를 눈금으로** 쓴다 — 한 칸 29.1mm라 눈으로 2~3mm까지
읽히고, 로봇 오차는 보통 1~2cm라 충분하다.
**기물도 흡착 성공도 필요 없다.** 빈 팔을 내려 흡착컵 위치만 보면 된다.

```bash
python hardware/calibrate_board.py --port COM5 --step 1 --step-delay 0.1
```
1. 기동 절차(8-A) → `engage` → `start`
2. 9개 칸(`a8 d8 h8 a5 d5 h5 b3 e3 f6`)으로 자동 이동
3. 각 칸에서 흡착컵이 **칸 중심에서 벗어난 양**을 mm로 입력
   - 부호: 앞(로봇에서 멀어짐)=**+x**, 왼쪽=**+y**
   - 예) 5mm 앞·3mm 오른쪽 → `5 -3`
   - `s`=건너뛰기, `q`=측정 종료하고 피팅
4. 끝나면 `vision/board_fit.json` 저장 → **다음 실행부터 자동 적용**

구해지는 것: **전체 밀림(offset) · 축척(scale) · 기울어짐(skew)**.
계통 오차의 대부분이라 6~9점이면 충분하다.

```
측정 9점 → affine 보정식
보정 전 평균 오차:  16.1 mm
보정 후 잔차     :   1.1 mm
→ 93% 감소
```
> 보정을 끄려면 `vision/board_fit.json`을 지우면 된다(자동으로 무보정 동작).
> 카메라 캘리브레이션과 무관한 별개 파일이다.

### 9-A-2. 시뮬 RL 재학습

⚠️ **기존 `stage2_final.zip`은 못 쓴다.** 학습 당시와 지금의 관절각 범위가
**전혀 겹치지 않는다**(elbow-up→down으로 바뀌어 q3 부호까지 반대). 모델이 한 번도
본 적 없는 입력이라 나오는 보정값은 근거 없는 외삽이다. **재학습이 필요하다.**

학습 환경이 `inverse_kinematics`를 그대로 호출하므로 **수정된 기구학이 자동 반영**된다.

```bash
pip install pybullet             # 학교 노트북엔 안 깔릴 수 있음(개인 PC 권장)
python sim/benchmark.py          # ① 내 컴퓨터 속도부터 측정 (2~3분)
python sim/train_correction.py --stage 1
```

**속도**: GPU는 거의 안 쓰인다(물리 시뮬은 CPU, 정책망은 작은 MLP).
**CPU가 병목**이고 `--device cpu`가 보통 같거나 빠르다. `benchmark.py`가
목표 스텝별 예상 시간을 표로 알려준다.

**몇 스텝이 필요한가**
| 스텝 | 의미 |
|---|---|
| 300,000 | 효과가 보이기 시작 (최소) |
| **500,000** | **안정적 개선 (권장 하한)** |
| 1,000,000 | 충분 |
| 2,000,000 | `STAGE1_STEPS` 기본값 |

- **50,000 스텝마다 체크포인트** 저장 → `Ctrl+C`로 끊고 **같은 명령 재실행하면 이어서** 학습
- 시간을 줄이려면 `sim/train_correction.py`의 `STAGE1_STEPS`를 낮춘다
- 결과: `models/correction_model/stage1_final.zip` → `models/correction_model/`에 두면 `main.py`가 자동 로드

**Colab에서 돌리려면** (pybullet 설치가 안 될 때)
```python
!git clone -b claude/chess-robot-arm-LIojd https://github.com/2025kuhshw-design/chess_robotarm.git
%cd chess_robotarm
!pip install pybullet stable-baselines3[extra] gymnasium
!python sim/train_correction.py --stage 1
```
결과가 구글드라이브 `MyDrive/chess_robot/correction_model/`에 저장된다.

---

## 10. 캘리브레이션

### 10-1. 카메라 (현장 필수)
카메라를 **보드 정중앙 위 40~60cm, 똑바로 아래** 보게 고정 후:
```bash
python vision\calibrate.py --camera 0
```
1. 체스판 **8×8 플레이 영역의 4모서리** 클릭 (테두리 X)
2. 순서: **좌상 → 우상 → 우하 → 좌하** (시계방향)
3. ⭐ **코너 3·4 변(아래쪽)이 로봇과 가까운 쪽**이 되도록 클릭
   (미리보기 아래쪽에 `ROBOT THIS SIDE` 마젠타 선이 뜬다)
4. 미리보기에 체스판이 **정사각형 가득** 떠야 정상 (까맣게 나오면 코너를 잘못 찍은 것 → `r`로 초기화)
5. `s` 키 → `vision/calibration.json` 저장

> 키가 안 먹으면: **한/영을 영문으로** + **영상 창을 클릭해 활성화**

### 보드 방향 설정 — `ROBOT_SIDE`
화면에서 로봇이 **어느 변에 있는지**를 `vision/detect.py`의 `ROBOT_SIDE`로 지정한다.
코드가 이 값으로 화면 격자 ↔ 체스/로봇 좌표를 변환한다.

| 값 | 로봇 위치 (탑뷰 기준) | a1이 오는 위치 |
|---|---|---|
| `"left"` ← **현재 설정** | 왼쪽 (코너 1·4 변) | 우하 |
| `"right"` | 오른쪽 (코너 2·3 변) | 좌상 |
| `"top"` | 위쪽 (코너 1·2 변) | 좌하 |
| `"bottom"` | 아래쪽 (코너 3·4 변) | 우상 |

카메라 위치나 보드 배치를 바꾸면 이 값만 고치면 된다.
**`calibration.json`은 다시 만들 필요 없다** — 거기엔 클릭한 4점의 픽셀
좌표만 들어 있고, 방향 변환은 코드 쪽에서 처리한다.

> **배치 전제**: 사람(WHITE)은 로봇(BLACK) **맞은편**에 앉는다.
> 랭크1이 사람 쪽, 랭크8이 로봇 쪽이어야 백 폰이 로봇 쪽으로 전진하고
> 엔진이 사람 수를 정상 판정한다. a1은 **사람 기준 왼쪽 앞** 칸이다.
> 네 설정 모두 로봇 좌표계와 같은 오른손계로 검증되어 있다(거울상 아님).

좌표 검증:
```bash
python main.py --mode vision --camera 0
```
- 각 칸에 좌표(a1~h8) 라벨, **a1은 마젠타**, 로봇 쪽 변에 `ROBOT THIS SIDE` 표시
- 로봇 a1 칸에 기물 → 화면 마젠타 a1에 떠야 정상
- **b1**(a1의 옆칸)에 옮겼을 때 화면에서도 `b1` 라벨 칸으로 가면 축 정상
- 안 맞으면 `ROBOT_SIDE`를 다른 값으로 바꿔가며 확인 (4가지뿐)

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

> 💡 **기준자세를 직접 만들 필요는 없다.** 만들기 쉬운 자세에서 재고 환산하면 된다.
> 예: 상완이 **수직**일 때 `q2=+90°`이므로 `SERVO2_HOME = (수직일 때 값) − DIR×90`.
> 팔꿈치는 어깨 위치와 무관하므로 **일직선일 때 값을 그대로** 쓰면 된다.

**③ Z자 park 자세 → `PARK_POSE`**
게임 시작/휴식 시 팔이 접혀 있을 각도. 카메라 시야를 안 가리고 토크 부하도 줄인다.
**이 자세가 기동 절차(8-A)의 기준**이기도 하다.

**④ 현재 확정값** (이 팔 기준, 실측 완료)
```python
SERVO1_HOME = 85;  SERVO1_DIR = +1   # 정면일 때 85 (2차 교체 서보, 실측)
SERVO2_HOME = -10; SERVO2_DIR = +1   # 수직 80 → 80-90 = -10
SERVO3_HOME = 50;  SERVO3_DIR = -1   # 일직선일 때 50
PARK_POSE   = (85, 140, 180)         # Z자
```
`utils/ik_solver.py` 쪽:
```python
SHOULDER_HEIGHT = 0.089   # 어깨 관절이 보드면보다 8.9cm 위 (실측)
ELBOW_SIGN      = -1      # elbow-down (서보 가동범위에 맞는 쪽)
PIECE_Z         = 0.007   # 기물 높이 7mm (실측)
```

> **`SHOULDER_HEIGHT`와 `ELBOW_SIGN`은 원래 코드에 없던 값**이다. 어깨가 보드면과
> 같은 높이라고 가정했고 팔꿈치 해도 한쪽만 썼는데, 실측해 보니 둘 다 틀려서
> 서보 가동범위를 벗어났다. 팔을 바꾸면 이 둘부터 다시 확인할 것.

### 10-3. 체스판 물리 배치 ⭐

캘리브레이션 값이 정해지면, **체스판을 팔이 닿는 위치에 놓아야** 한다.
기준점 **(0,0) = 로봇 베이스(s1) 회전축**, `+x` = 로봇 정면, `+y` = 로봇 왼쪽.

| 항목 | 위치 |
|---|---|
| 8×8 격자 **가까운 변**(랭크8, 로봇 쪽) | x = **11.0 cm** |
| 8×8 격자 **먼 변**(랭크1, 사람 쪽) | x = 34.3 cm |
| 8×8 격자 **오른쪽 변**(파일 a) | y = **−6.0 cm** |
| 8×8 격자 **왼쪽 변**(파일 h) | y = +17.3 cm |
| **보드 중심** (제일 재기 쉬움) | x = **22.7**, y = **+5.7 cm** |
| 한 칸 | 2.91 cm |

- 측정 지점은 **나무 테두리가 아니라 8×8 칸이 시작되는 선**.
- 보드가 **로봇 왼쪽으로 5.7cm 치우친다.** (베이스 서보를 교체하기 전에는
  정면이 `s1=38`이라 우회전 여유가 38°뿐이어서 이렇게 배치했다. 교체 후
  정면은 `s1=90`이라 좌우 여유가 같지만, 보드는 테이프로 고정돼 있고
  캘리브레이션도 이 배치 기준이므로 그대로 둔다.)
- 캡처 구역(잡은 기물 놓는 곳)은 보드 **왼쪽 바깥 y=20~23cm**. 비워둘 것.

**도달 가능 범위** (현재 배치, 46/64칸)
```
        a  b  c  d  e  f  g  h
 랭크8  O  O  O  O  O  O  O  O   ← 로봇 쪽 (흑 기물)
 랭크7  O  O  O  O  O  O  O  O
 랭크6  O  O  O  O  O  O  O  O
 랭크5  O  O  O  O  O  O  O  O
 랭크4  O  O  O  O  O  O  O  O
 랭크3  O  O  O  O  O  O  .  .
 랭크2  .  .  .  .  .  .  .  .
 랭크1  .  .  .  .  .  .  .  .   ← 사람 쪽 (백 기물)
```
랭크1~2는 못 닿지만 **로봇은 흑만 옮기고 사람은 손으로 두므로** 진행에 지장 없다.
중앙 4×4와 흑 진영(랭크7·8)은 전부 도달 가능.

---

## 11. 조정 상수 한눈에

### `hardware/arm_controller.py`
| 상수 | 의미 |
|---|---|
| `SERVO1/2/3_HOME` | IK 기준자세(q=0)에 해당하는 서보 각도 |
| `SERVO1/2/3_DIR` | 관절 회전 방향 (+1 / −1) |
| `SERVO_SAFE_MIN/MAX` | 관절별 소프트 각도 제한 (충돌 방지) |
| `PARK_POSE` | 시작/휴식 Z자 자세 |
| `RAMP_STEP_DEG` / `RAMP_DELAY` | 이동 속도 기본값 (`--step`/`--step-delay`로 덮어씀) |
| `LIFT_HEIGHT` / `LIFT_MIN` | 기물 들어올리는 높이 (먼 칸은 자동 축소) |
| `TOUCH_PRESS` | 하강 시 기물 윗면보다 더 내려갈 깊이 (흡착 밀착용) |
| `CAPTURE_X0/Y0/SPACING` | 잡은 기물 놓는 구역 위치 |
| `EXPECTED_FW` | 요구 펌웨어 버전 (`.ino`의 `FW_VERSION`과 일치해야) |

### `vision/board_fit.json` (실측 보정 결과)
`calibrate_board.py`가 만든다. 있으면 `chess_square_to_xyz`가 자동 적용,
없으면 무보정. 지우면 보정이 꺼진다. git에는 올리지 않는다(환경마다 다름).

### `hardware/arduino/servo_control.ino`
| 상수 | 의미 |
|---|---|
| `CH_JOINT1/2/3` | 서보 채널 (0/1/2) |
| `CH_PUMP` / `CH_VALVE` | 펌프 ch4 / 밸브 ch5 |
| `FW_VERSION` | 펌웨어 버전 (고칠 때마다 올릴 것) |
| `SERVO_MIN_PWM` / `SERVO_MAX_PWM` | 서보 펄스 범위 (떨림·발열 시 조정) |
| `ANGLE_HARD_MAX` | 확장 최대각 (기본 200) |
| `DEV_ON_PULSE` / `DEV_OFF_PULSE` | 흡착 모듈 ON/OFF 펄스 (반대면 교체) |
| `VALVE_HOLD` / `VALVE_RELEASE` | 밸브 동작 (반대면 교체) |

### `utils/ik_solver.py`
| 상수 | 의미 |
|---|---|
| `L1_DEFAULT` 0.140 / `L2_DEFAULT` 0.155 / `L3_DEFAULT` 0.075 | 링크 길이 (m) |
| `SHOULDER_HEIGHT` | 어깨 관절이 보드면보다 높은 거리 (m) |
| `ELBOW_SIGN` | 팔꿈치 자세 (+1=up / −1=down) |
| `BOARD_ORIGIN_X/Y` | 체스판 원점(a1 코너) 위치 |
| `CELL_SIZE` 0.029125 | 한 칸 크기 (m) |

---

## 11-B. 기물 인식 점검 ⭐

캘리브레이션(격자)이 맞아도 **기물 판정**은 따로 확인해야 한다.
기물을 시작 배치로 놓고:

```bash
python vision/check_board.py --camera 1                # 지금 설정으로 판독
python vision/check_board.py --camera 1 --all-sides    # 판 방향까지 자동 탐색
python vision/check_board.py --camera 1 --thresh 12    # 임계값 바꿔 시험
```

`test_square.py` 안에서는 `board` 명령으로 같은 점검을 할 수 있다.

맞을 때의 출력:
```
  8 b b b b b b b b
  7 b b b b b b b b
  6 . . . . . . . .
  ...
  2 w w w w w w w w
  1 w w w w w w w w
    a b c d e f g h
시작 배치와 64/64 칸 일치
```

### 어떻게 판정하나
셀 중앙 60%의 **평균 밝기**를 재고, **그 칸이 비었을 때의 밝기**와 비교한다.
밝으면 흰 기물, 어두우면 검은 기물, 비슷하면 빈 칸.
기준 밝기는 "지금 비어 있어야 할 칸들"에서 매번 새로 잡으므로 조명이 변해도 따라간다.

> ⚠️ 예전에는 셀 중앙의 **분산**으로 판정했다. 이는 틀렸다 — 기물이 셀을 덮으면
> 그 부분이 오히려 균일해져 분산이 0에 가까워진다. 그래서 32개 기물이 **전부**
> 빈 칸으로 읽혔다.

### 안 맞을 때
| 증상 | 조치 |
|---|---|
| 기물이 있는데 `.` 로 읽힘 | `OCC_DIFF_THRESH` 를 낮춘다 (`--thresh 12`) |
| 빈 칸이 기물로 읽힘 | `OCC_DIFF_THRESH` 를 올린다 (`--thresh 25`) |
| 기물은 잡히는데 엉뚱한 칸 | `--all-sides` 로 `ROBOT_SIDE` 확인 |
| 흑백이 뒤바뀜 | 판이 180° 돌아 있는지 확인 (`--all-sides`) |

### 기물 만들기 — 크기·두께·색

| 항목 | 권장 | 이유 |
|---|---|---|
| 모양 | 정사각 또는 원 (상관없음) | 실측 결과 인식률 차이 없음 |
| 크기 | **23~25mm** | 칸 29.1mm에 3mm씩 여유. 26mm도 되지만 여유가 1.6mm뿐 |
| 두께 | 4~5mm (**윗면이 평평할 것**) | 흡착컵이 밀착돼야 진공이 걸린다 |
| 윗면 색 | **진한 색 두 가지** (예: 초록 / 파랑) | 아래 참고 |

**⚠️ 기물을 바꾸면 `utils/ik_solver.py` 의 `PIECE_Z` 를 다시 재서 넣을 것.**
이 값이 실제보다 크면 흡착컵이 기물에 닿기 전에 멈춰 진공이 안 걸린다.
현재 설정: `PIECE_Z = 0.0045` (4.5mm), `TOUCH_PRESS = 0.003` (3mm 더 눌림).

**기물 수가 32개가 아니면** `--fen` 으로 실제 배치를 알려줘야 한다.
안 그러면 카메라가 본 배치와 엔진이 아는 배치가 달라 모든 수가 어긋난다.

```bash
# 16개(양쪽 킹1+룩1+폰6)를 랭크4·6에 놓는 배치 — 전부 팔 도달권
python main.py --mode real --port COM5 --camera 1 --view \
    --fen "8/8/krpppppp/8/KRPPPPPP/8/8/8 w - - 0 1"
```
```
    a b c d e f g h
  6 k r p p p p p p     ← 로봇(흑)
  4 K R P P P P P P     ← 사람(백)
```
> 랭크1·2 와 g3·h3 는 팔이 안 닿으므로 그 칸을 쓰는 배치는 피할 것.
> `check_board.py` 에도 같은 `--fen` 을 주면 그 배치로 대조한다.

---

### ⭐⭐ 가장 확실한 방법 — 기물 윗면에 색 붙이기

카메라는 기물의 **윗면만** 본다. 그러니 윗면을 **체스판에 없는 색**으로 만들면
판 색깔과 무관하게 100% 구분된다. 알고리즘을 아무리 고쳐도 안 되던 게
테이프 한 장으로 끝난다.

**색 고르는 법 — 진한 색 두 가지**

| | 쓸 것 | 쓰면 안 되는 것 | 이유 |
|---|---|---|---|
| 흰쪽(사람) | 초록 | ❌ **흰색** | 밝은 칸(연한 나무색)과 같아짐 |
| 검은쪽(로봇) | 파랑 | ❌ **검은색** | 어두운 칸과 같아짐 |
| 둘 다 | — | ❌ **노랑** | 흡착컵 마커 색과 겹침 |

합성 영상 실측 (기물 지름 12mm, 아주 작은 경우):

| 조합 | 결과 |
|---|---|
| 흰색 / 빨강 | 40/64 ❌ |
| **초록 / 파랑** | **64/64** ✅ |
| **노랑 / 빨강** | 64/64 (단 마커 색을 바꿔야 함) |

```bash
python vision/pick_pieces.py --camera 1
#  1 누르고 → 흰쪽 기물 윗면 클릭 (여러 개)
#  2 누르고 → 검은쪽 기물 윗면 클릭
#  s 저장 (PIECE_COLOR_MODE 가 자동으로 켜진다)
```

화면 아래에 `cells white=16 black=16` 이 뜨면 성공.
기물 윗면만 칠해지고 판은 안 칠해져야 한다.

> 기물이 작아도 된다. 지름 12mm에서도 64/64가 나왔다.
> 잘 안 잡히면 `detect.py` 의 `PIECE_COLOR_MIN_FRAC` 를 낮춘다(기본 0.08).

---

### 차선책 — 빈 판 기준 영상 찍기

**밝기만으로는 검은 기물과 어두운 칸을 구분할 수 없다.** 둘 다 어둡기 때문이다.
실제로 검은 기물이 **어두운 칸에서만 통째로 안 잡혔고**, 임계값을 낮추면 이번엔
빈 칸이 기물로 잡히기 시작했다(맞바꿈이라 임계값으로는 해결 불가).

해결은 **기물을 다 치운 판을 한 번 찍어두고 그것과 비교**하는 것이다.
검은 기물이 어두운 칸에 있어도 나뭇결·테두리·그림자가 달라지므로 잡힌다.
평균이 아니라 **달라진 픽셀의 비율**로 판단하므로 기물이 작아도 된다.

```bash
# 1) 체스판에서 기물을 전부 치우고, 팔은 park(Z자)로 둔다
python vision/check_board.py --camera 1 --capture-empty

# 2) 기물을 시작 배치로 놓고 확인
python vision/check_board.py --camera 1
```

`test_square.py` 안에서는 `refcap` 명령으로도 찍을 수 있다.

> 만들어진 `vision/empty_ref.npz` 는 설치 환경마다 다르므로 git에 올리지 않는다.
> **체스판이나 카메라를 옮기면 다시 찍어야 한다.**

| 증상 | 조치 |
|---|---|
| 기물 있는 칸이 안 잡힘 | `--frac 0.10` (비율 임계 낮추기) |
| 빈 칸이 기물로 잡힘 | `--frac 0.25` (비율 임계 올리기) |
| 여전히 이상 | `--all-sides` 로 `ROBOT_SIDE` 확인 |

좋은 값을 찾았으면 `vision/detect.py` 의 `OCC_AREA_FRAC` 에 적는다.

---

### 마커가 안 잡힐 때 ⭐

`mark` 가 계속 "못 찾음"이면 색 범위(`MARKER_HSV_RANGES`)가 실제 마커 색과
안 맞는 것이다. 숫자를 눈으로 추측하지 말고 **화면에서 직접 재라**:

```bash
# 1) 먼저 팔을 체스판 위로 (마커가 카메라에 보이게)
python hardware/test_square.py --port COM5
square> e5
square> q          # ⚠️ 반드시 종료 — 카메라를 두 프로그램이 동시에 못 쓴다

# 2) 색 고르기
python vision/pick_marker.py --camera 1
```

창에서 **마커를 클릭**하면 그 픽셀의 HSV를 재서 범위를 만든다.
잡힌 영역이 초록으로 칠해지므로 **마커만 초록**이 되도록 몇 번 더 클릭한다.
`s` 를 누르면 `vision/detect.py` 에 바로 저장된다.

| 화면 표시 | 뜻 |
|---|---|
| `blobs=1 max=350px (OK)` | 성공 — 마커 하나만 알맞은 크기로 잡힘 |
| `(BAD)` | 덩어리가 너무 크거나 작다 → 다른 곳이 같이 잡힌 것 |
| `blobs=0` | 아무것도 안 잡힘 → 마커를 다시 클릭 |

- 팔이 체스판 밖에 있으면 `--raw` 로 원본 화면에서 고를 수 있다
- `r` 초기화, `q` 저장 없이 종료

---

`--thresh` 로 좋은 값을 찾았으면 `vision/detect.py` 의 `OCC_DIFF_THRESH` 에 적는다.

---

## 12. 최종 실행

```bash
venv\Scripts\activate
python main.py --mode real --port COM5 ^
    --stockfish C:\...\stockfish\stockfish-windows-x86-64-avx2.exe --camera 1 ^
    --view --step 1 --step-delay 0.1
```

실행하면 **기동 절차**(8-A)가 먼저 뜬다 → 팔을 Z자로 잡고 `engage` → `start`.
동작이 검증된 뒤에는 속도 옵션을 빼거나 올려도 된다(8-B).


- 시작 시 **시작 배치를 제대로 읽는지 자동 점검**한다.
  `64칸 중 N칸 일치`가 60 미만이면 인식·방향 설정이 잘못된 것이다.
- `--view` 를 주면 **카메라 창이 계속 떠 있다** (test_square의 `show`와 같다).
  라이브 뷰 스레드가 카메라를 독점하고, 인식은 그 최신 프레임을 빌려 쓴다.
- 사람(WHITE) 차례 → **실제로 말을 옮기고** Enter → 카메라가 인식
  - ⚠️ `e2e4` 같은 **글자를 치는 게 아니다.** 물리적으로 옮겨야 한다.
  - Enter 대신 `u` = 수를 직접 입력, `s` = 카메라가 본 배치를 표로 확인.
  - 인식은 **합법수 대조** 방식이다. 지금 국면의 합법수(보통 20~40개)마다
    "그 수를 뒀다면 판이 어떻게 보일지"를 만들어 화면과 견주고, 가장 잘
    맞는 것을 고른다. 칸 몇 개를 잘못 읽어도 정답이 살아남는다.
  - 1등과 2등이 비슷하면 **후보 5개를 보여주고 사람이 고른다.** 게임이 멈추지 않는다.
- 로봇(BLACK) 차례 → Stockfish → IK → (모델 있으면) RL 보정 → 집기→이동→놓기→park 복귀

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
| **시작하자마자 팔이 빠르게 이상한 자세로 튐** | ① `.ino` 재업로드 누락(구버전이 부팅 시 90,90,90으로 감) → `fw=3 ✓` 확인 ② 기동 절차에서 `engage` 값이 실제와 다름 |
| 팔이 전원 켜도 축 처져 있음 | **정상**. 부팅 시 서보를 잡지 않도록 한 것(슬램 방지). `engage`로 힘을 넣는다 |
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
| **흡착이 안 걸림 / 기물이 안 붙음** | 하강이 얕은 것. `zoff 8` → `zoff 12`로 늘려가며 찾고 `TOUCH_PRESS`에 반영 |
| **집자마자 기물을 놓아버림** | 수정 완료(흡착이 이동 중 유지됨). 구버전이면 `git pull` |
| `suction 1` 했더니 팔이 움직임 | 수정 완료. `git pull` |
| `서보 가동범위 밖: s3=190` | 그 칸은 현재 배치에서 도달 불가. 도달 지도(10-3) 참고 |
| `up`/`down`이 안 먹음 | 칸을 먼저 지정해야 한다 (예: `e5` → `down`) |

---

## 14. 빠른 체크리스트

```
[집에서 미리]
□ Python 3.11 또는 3.12 설치 (3.13+ 금지) + py launcher
□ Git · Arduino IDE + Adafruit PWM Servo Driver 라이브러리
□ USB: stockfish.exe / (오프라인 휠)   ※ RL 모델은 재학습 필요
□ 컴퓨터만으로: test ik / lagrange / full / rl 전부 PASS 확인

[현장 — 순서대로]
□ git clone + checkout + pull
□ py -3.12 -m venv venv → activate → pip install -r requirements.txt
□ USB 파일 배치 (models/correction_model/)
□ 배선: 로직(VCC/SDA/SCL/GND) + 서보전원(V+) + ⭐펌프 V+ 직결 + 공통 GND
□ .ino 업로드 → "READY fw=3" → 시리얼 모니터 닫기
□ test serial → OK
□ test_suction.py (서보 뽑고) → 펌프 on/off, 전원 안 죽는지
□ servo_jog.py → 서보 개별 확인, 안전범위 파악
□ 서보 캘리브레이션 → SERVO*_HOME / SERVO*_DIR / PARK_POSE
□ 어깨높이 실측 → SHOULDER_HEIGHT,  팔꿈치 방향 → ELBOW_SIGN
□ 체스판 배치 (격자 중심 x=22.7cm, y=+5.7cm) + 캡처 구역(y=20~23cm) 확보
□ 카메라 캘리브레이션 + --mode vision 좌표 검증 (a1 = 사람 쪽)
□ test_square.py → engage → start → 칸 정확도 확인
□ zoff 로 하강 깊이 맞추기 → TOUCH_PRESS 에 반영
□ pick / place / move 로 실제 집기·놓기 확인
□ ⭐ calibrate_board.py → 실측 보정 (20분, 정확도 크게 향상)
□ diagnose.py → 동시 부하 확인
□ main.py --mode real  실전

[여유 있으면]
□ sim/benchmark.py → 학습 속도 확인
□ sim/train_correction.py --stage 1 → RL 재학습 (기존 모델은 못 씀)
```

**가장 빠뜨리기 쉬운 것**
- USB의 `stockfish.exe`, 현장 `calibration.json`(새로 생성)
- **펌프 V+ 직결**, **공통 GND**
- **`.ino` 재업로드** → 실행 시 `fw=3 ✓` 한 줄로 확인
- **기동 절차**: 팔을 Z자로 잡고 `engage` → `start` (안 하면 첫 명령에 튐)

---

## 15. 알아둘 설계 제약

| 제약 | 이유 / 대응 |
|---|---|
| **랭크1~2에 팔이 안 닿음** | 팔 길이(29.5cm) 한계 — 베이스 회전이 아니라 거리 문제. 로봇은 흑만 옮기므로 진행엔 무방 |
| **첫 동작은 속도를 못 늦춤** | 서보에 위치 센서가 없어 출발점을 모름 → 기동 절차(8-A)로 해결 |
| **기존 RL 모델을 못 씀** | elbow-up으로 학습됐는데 현재는 elbow-down이라 관절각 범위가 전혀 겹치지 않음. 재학습 필요(9-A-2) |
| **시뮬 RL은 실제 오차를 못 줄임** | 실물을 본 적이 없어 원리상 불가. 실제 정확도는 실측 보정(9-A-1)으로 올린다 |
| **s3가 180 근처까지 감** | 팔꿈치 혼 장착 위치 때문. 여유가 필요하면 혼을 30°쯤 돌려 끼우면 됨 |
| **기물 색 오인식은 무해** | 게임은 "빈 칸 ↔ 찬 칸"만 사용 |
