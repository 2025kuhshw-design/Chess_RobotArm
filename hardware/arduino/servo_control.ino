/*
 * 체스 로봇팔 아두이노 제어 (PCA9685 16채널 PWM 드라이버 사용)
 *
 * 하드웨어 구성:
 *   - Arduino Uno (로직/시리얼만 담당, USB 전원)
 *   - PCA9685 서보 드라이버 (I2C: SDA=A4, SCL=A5, VCC=Uno 5V)
 *   - 외부 6V 8A 어댑터 → PCA9685 V+ 터미널 (서보/펌프/밸브 구동 전원)
 *   - 공통 GND: 외부전원 GND ─ PCA GND ─ Uno GND 모두 연결
 *
 * PCA9685 채널 배치:
 *   ch0 = joint1 서보 (베이스, MG996R)
 *   ch1 = joint2 서보 (어깨,   MG996R)
 *   ch2 = joint3 서보 (팔꿈치, MG996R)
 *   ch3 = 펌프   (인라인 PWM MOSFET 보드)
 *   ch4 = 밸브   (인라인 PWM MOSFET 보드)
 *
 * 시리얼 명령(기존과 동일): "A{각도1},{각도2},{각도3},{흡착기}\n"
 *   예: "A120,85,60,1\n"  (흡착기 1=집기 / 0=놓기)
 * 응답: "OK\n" 또는 "ERR\n"
 *
 * ※ Adafruit PWM Servo Driver Library 설치 필요
 *   (Arduino IDE → 라이브러리 매니저 → "Adafruit PWM Servo Driver" 검색)
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ─────────────────────────────────────────
// PCA9685 채널 / 상수
// ─────────────────────────────────────────
#define CH_JOINT1   0
#define CH_JOINT2   1
#define CH_JOINT3   2
#define CH_PUMP     3
#define CH_VALVE    4

#define BAUD_RATE     9600
#define ANGLE_MIN     0
#define ANGLE_MAX     180
#define NEUTRAL_ANGLE 90
#define TIMEOUT_MS    5000   // 5초 이상 명령 없으면 중립 복귀

// 서보 PWM 펄스 범위 (PCA9685 12비트 카운트, 50Hz 기준)
// 50Hz → 주기 20ms → 4096카운트. 1ms≈205, 2ms≈410.
// MG996R 실측에 맞춰 SERVO_MIN/MAX를 미세 조정하세요(과구동 시 서보 떨림/발열).
#define SERVO_FREQ    50
#define SERVO_MIN_PWM 110   // 약 0.54ms (0도)
#define SERVO_MAX_PWM 510   // 약 2.49ms (180도)

// 펌프/밸브: 인라인 MOSFET 보드에 풀듀티 ON / 0 OFF
#define PWM_FULL_ON   4095
#define PWM_OFF       0

// 밸브 동작 정의 (밸브 종류에 맞게 조정)
//   기본: 평상시 닫힘(NC) + "전원 인가 시 열림" 가정
//   → 집기(흡착ON): 펌프 ON, 밸브 OFF(닫힘=진공 유지)
//   → 놓기(흡착OFF): 펌프 OFF, 밸브 ON(열림=진공 해제, 기물 낙하)
//   밸브가 NO(평상시 열림)이면 VALVE_HOLD/RELEASE 값을 서로 바꾸세요.
#define VALVE_HOLD    PWM_OFF      // 집기 중 밸브 상태(진공 유지)
#define VALVE_RELEASE PWM_FULL_ON  // 놓기 시 밸브 상태(진공 해제)

// ─────────────────────────────────────────
// 전역 변수
// ─────────────────────────────────────────
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();   // 기본 주소 0x40
unsigned long lastCmdTime = 0;
String inputBuffer = "";

// ─────────────────────────────────────────
// 각도(0~180) → PCA9685 PWM 카운트
// ─────────────────────────────────────────
int angleToPulse(int angle) {
  if (angle < ANGLE_MIN) angle = ANGLE_MIN;
  if (angle > ANGLE_MAX) angle = ANGLE_MAX;
  return map(angle, ANGLE_MIN, ANGLE_MAX, SERVO_MIN_PWM, SERVO_MAX_PWM);
}

// ─────────────────────────────────────────
// setup
// ─────────────────────────────────────────
void setup() {
  Serial.begin(BAUD_RATE);

  Wire.begin();
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);   // 내부 오실레이터(라이브러리 권장값)
  pwm.setPWMFreq(SERVO_FREQ);             // 서보용 50Hz

  // 전원 켜면 전 서보 중립(90도), 펌프/밸브 OFF
  goNeutral();
  lastCmdTime = millis();

  Serial.println("READY");
}

// ─────────────────────────────────────────
// loop
// ─────────────────────────────────────────
void loop() {
  // 시리얼 수신
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }

  // (타임아웃 자동 중립복귀 제거 — 마지막 명령 위치를 그대로 유지)
}

// ─────────────────────────────────────────
// 명령 파싱 및 실행
// ─────────────────────────────────────────
void processCommand(String cmd) {
  cmd.trim();

  // 명령 형식 검증: "A{a1},{a2},{a3},{suction}"
  if (cmd.length() < 2 || cmd.charAt(0) != 'A') {
    Serial.println("ERR");
    return;
  }

  String body = cmd.substring(1);   // "A" 제거
  int v[4];
  int idx = 0;
  int start = 0;

  for (int i = 0; i <= body.length() && idx < 4; i++) {
    if (i == body.length() || body.charAt(i) == ',') {
      v[idx++] = body.substring(start, i).toInt();
      start = i + 1;
    }
  }

  if (idx < 4) {
    Serial.println("ERR");
    return;
  }

  int a1      = v[0];
  int a2      = v[1];
  int a3      = v[2];
  int suction = v[3];

  // 각도 유효 범위 검증
  if (a1 < ANGLE_MIN || a1 > ANGLE_MAX ||
      a2 < ANGLE_MIN || a2 > ANGLE_MAX ||
      a3 < ANGLE_MIN || a3 > ANGLE_MAX ||
      suction < 0 || suction > 1) {
    Serial.println("ERR");
    return;
  }

  // 서보 이동 (PCA9685 채널 PWM)
  pwm.setPWM(CH_JOINT1, 0, angleToPulse(a1));
  pwm.setPWM(CH_JOINT2, 0, angleToPulse(a2));
  pwm.setPWM(CH_JOINT3, 0, angleToPulse(a3));

  // 흡착기: 비트 1개 → 펌프 + 밸브 두 채널로 변환
  if (suction == 1) {
    pwm.setPWM(CH_PUMP,  0, PWM_FULL_ON);   // 펌프 ON (진공 생성)
    pwm.setPWM(CH_VALVE, 0, VALVE_HOLD);    // 밸브 닫힘 (진공 유지)
  } else {
    pwm.setPWM(CH_PUMP,  0, PWM_OFF);       // 펌프 OFF
    pwm.setPWM(CH_VALVE, 0, VALVE_RELEASE); // 밸브 열림 (진공 해제 → 기물 낙하)
  }

  lastCmdTime = millis();
  Serial.println("OK");
}

// ─────────────────────────────────────────
// 중립 위치로 복귀 (서보 90도, 펌프/밸브 OFF)
// ─────────────────────────────────────────
void goNeutral() {
  pwm.setPWM(CH_JOINT1, 0, angleToPulse(NEUTRAL_ANGLE));
  pwm.setPWM(CH_JOINT2, 0, angleToPulse(NEUTRAL_ANGLE));
  pwm.setPWM(CH_JOINT3, 0, angleToPulse(NEUTRAL_ANGLE));
  pwm.setPWM(CH_PUMP,  0, PWM_OFF);
  pwm.setPWM(CH_VALVE, 0, PWM_OFF);   // 중립: 밸브도 비활성(전력 절약)
}
