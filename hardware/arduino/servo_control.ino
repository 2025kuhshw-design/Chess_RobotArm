/*
 * 체스 로봇팔 아두이노 서보 제어
 * 시리얼 명령: "A{각도1},{각도2},{각도3},{흡착기}\n"
 * 예: "A120,85,60,1\n"
 * 응답: "OK\n" 또는 "ERR\n"
 */

#include <Servo.h>

// ─────────────────────────────────────────
// 핀 설정 / 상수
// ─────────────────────────────────────────
#define PIN_JOINT1    3    // 베이스 (MG90S)
#define PIN_JOINT2    5    // 어깨   (MG996R)
#define PIN_JOINT3    6    // 팔꿈치 (MG996R)
#define PIN_SUCTION   10   // 흡착기 솔레노이드

#define BAUD_RATE     9600
#define ANGLE_MIN     0
#define ANGLE_MAX     180
#define NEUTRAL_ANGLE 90
#define TIMEOUT_MS    5000   // 5초 이상 명령 없으면 중립 복귀

// ─────────────────────────────────────────
// 전역 변수
// ─────────────────────────────────────────
Servo joint1, joint2, joint3;
unsigned long lastCmdTime = 0;
String inputBuffer = "";

// ─────────────────────────────────────────
// setup
// ─────────────────────────────────────────
void setup() {
  Serial.begin(BAUD_RATE);

  joint1.attach(PIN_JOINT1);
  joint2.attach(PIN_JOINT2);
  joint3.attach(PIN_JOINT3);

  pinMode(PIN_SUCTION, OUTPUT);
  digitalWrite(PIN_SUCTION, LOW);

  // 전원 켜면 전 서보 중립(90도)
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

  // 타임아웃: 5초 이상 명령 없으면 중립 복귀
  if (millis() - lastCmdTime > TIMEOUT_MS) {
    goNeutral();
    lastCmdTime = millis();
  }
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

  // 서보 이동
  joint1.write(a1);
  joint2.write(a2);
  joint3.write(a3);

  // 흡착기
  digitalWrite(PIN_SUCTION, suction == 1 ? HIGH : LOW);

  lastCmdTime = millis();
  Serial.println("OK");
}

// ─────────────────────────────────────────
// 중립 위치로 복귀
// ─────────────────────────────────────────
void goNeutral() {
  joint1.write(NEUTRAL_ANGLE);
  joint2.write(NEUTRAL_ANGLE);
  joint3.write(NEUTRAL_ANGLE);
  digitalWrite(PIN_SUCTION, LOW);
}
