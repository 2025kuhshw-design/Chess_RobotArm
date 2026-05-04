"""
RL 보정 모델 학습 스크립트 (PPO, stable-baselines3)
1단계: env_simple → 500,000 스텝
2단계: env_full  → 1,000,000 스텝 (1단계 이어서)

Google Colab A100 실행 지원:
  - 드라이브 마운트 자동 처리
  - 세션 재시작 시 최신 체크포인트 자동 로드
"""

import os
import sys
import glob

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수
# ─────────────────────────────────────────
STAGE1_STEPS   = 1_500_000
STAGE2_STEPS   = 1_000_000
LEARNING_RATE  = 3e-4
N_STEPS        = 2048
BATCH_SIZE     = 64
ENT_COEF       = 0.005
CHECKPOINT_FREQ= 50_000   # 체크포인트 저장 간격

# 로컬 저장 경로
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "correction_model")

# Colab 드라이브 경로 (Colab에서만 유효)
COLAB_DRIVE_DIR = "/content/drive/MyDrive/chess_robot/correction_model"

IS_COLAB = "google.colab" in sys.modules or os.path.exists("/content")


# ─────────────────────────────────────────
# Colab 드라이브 마운트
# ─────────────────────────────────────────
def mount_drive_if_colab():
    if not IS_COLAB:
        return LOCAL_MODEL_DIR
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        os.makedirs(COLAB_DRIVE_DIR, exist_ok=True)
        print(f"[Colab] 드라이브 마운트 완료: {COLAB_DRIVE_DIR}")
        return COLAB_DRIVE_DIR
    except Exception as e:
        print(f"[Colab] 드라이브 마운트 실패: {e} → 로컬 저장")
        return LOCAL_MODEL_DIR


# ─────────────────────────────────────────
# 최신 체크포인트 자동 탐색
# ─────────────────────────────────────────
def find_latest_checkpoint(model_dir: str, prefix: str = "rl_model") -> str | None:
    pattern = os.path.join(model_dir, f"{prefix}_*_steps.zip")
    files   = glob.glob(pattern)
    if not files:
        return None
    # 파일명에서 스텝 수 추출 후 최대값
    def extract_steps(path):
        basename = os.path.basename(path)
        try:
            return int(basename.split("_")[-2])
        except Exception:
            return 0
    return max(files, key=extract_steps)


# ─────────────────────────────────────────
# 체크포인트 콜백
# ─────────────────────────────────────────
def make_checkpoint_callback(save_dir: str, prefix: str):
    from stable_baselines3.common.callbacks import CheckpointCallback
    os.makedirs(save_dir, exist_ok=True)
    return CheckpointCallback(
        save_freq    = CHECKPOINT_FREQ,
        save_path    = save_dir,
        name_prefix  = prefix,
        verbose      = 1,
    )


# ─────────────────────────────────────────
# TensorBoard 커스텀 콜백 (평균 오차·성공률)
# ─────────────────────────────────────────
class MetricsCallback:
    """에피소드 info에서 dist_cm, 성공 여부 집계."""

    def __init__(self):
        from stable_baselines3.common.callbacks import BaseCallback
        import numpy as np

        class _Inner(BaseCallback):
            def __init__(self_inner):
                super().__init__(verbose=0)
                self_inner._ep_dists    = []
                self_inner._ep_success  = []
                self_inner._ep_rewards  = []

            def _on_step(self_inner) -> bool:
                infos = self_inner.locals.get("infos", [])
                dones = self_inner.locals.get("dones", [])
                for info, done in zip(infos, dones):
                    if done and "dist_cm" in info:
                        self_inner._ep_dists.append(info["dist_cm"])
                        self_inner._ep_success.append(float(info["dist_cm"] < 2.0))
                    if done and "episode" in info:
                        self_inner._ep_rewards.append(info["episode"]["r"])

                if len(self_inner._ep_dists) >= 10:
                    import numpy as np
                    self_inner.logger.record("custom/mean_dist_cm",   float(np.mean(self_inner._ep_dists)))
                    self_inner.logger.record("custom/success_rate",   float(np.mean(self_inner._ep_success)))
                    if self_inner._ep_rewards:
                        self_inner.logger.record("custom/mean_reward", float(np.mean(self_inner._ep_rewards)))
                    self_inner._ep_dists.clear()
                    self_inner._ep_success.clear()
                    self_inner._ep_rewards.clear()
                return True

        self.callback = _Inner()


# ─────────────────────────────────────────
# 1단계 학습
# ─────────────────────────────────────────
def train_stage1(model_dir: str) -> str:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from sim.env_simple import ChessArmEnvSimple

    print("\n" + "="*50)
    print("1단계 학습 시작 (env_simple, PPO)")
    print("="*50)

    env = Monitor(ChessArmEnvSimple())

    ckpt_path = find_latest_checkpoint(model_dir, "stage1")
    if ckpt_path:
        print(f"[체크포인트 발견] 이어서 학습: {ckpt_path}")
        model = PPO.load(ckpt_path, env=env)
        trained_steps = int(os.path.basename(ckpt_path).split("_")[-2])
        remaining     = max(0, STAGE1_STEPS - trained_steps)
    else:
        print("[새로 학습 시작]")
        model = PPO(
            "MlpPolicy", env,
            learning_rate = LEARNING_RATE,
            n_steps       = N_STEPS,
            batch_size    = BATCH_SIZE,
            ent_coef      = ENT_COEF,
            tensorboard_log = os.path.join(model_dir, "tb_logs"),
            verbose       = 1,
        )
        remaining = STAGE1_STEPS

    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([
        make_checkpoint_callback(model_dir, "stage1"),
        MetricsCallback().callback,
    ])

    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False)

    save_path = os.path.join(model_dir, "stage1_final")
    model.save(save_path)
    env.close()
    print(f"\n✅ 1단계 학습 완료: {save_path}")
    return save_path


# ─────────────────────────────────────────
# 2단계 학습
# ─────────────────────────────────────────
def train_stage2(stage1_path: str, model_dir: str) -> str:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from sim.env_full import ChessArmEnvFull

    print("\n" + "="*50)
    print("2단계 학습 시작 (env_full, Domain Randomization)")
    print("="*50)

    env = Monitor(ChessArmEnvFull())

    ckpt_path = find_latest_checkpoint(model_dir, "stage2")
    if ckpt_path:
        print(f"[체크포인트 발견] 이어서 학습: {ckpt_path}")
        model = PPO.load(ckpt_path, env=env)
        trained_steps = int(os.path.basename(ckpt_path).split("_")[-2])
        remaining     = max(0, STAGE2_STEPS - trained_steps)
    else:
        print(f"[1단계 모델 로드] {stage1_path}")
        model = PPO.load(stage1_path, env=env)
        remaining = STAGE2_STEPS

    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([
        make_checkpoint_callback(model_dir, "stage2"),
        MetricsCallback().callback,
    ])

    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False)

    save_path = os.path.join(model_dir, "stage2_final")
    model.save(save_path)
    env.close()
    print(f"\n✅ 2단계 학습 완료: {save_path}")
    return save_path


# ─────────────────────────────────────────
# 단독 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="학습 단계 (1: env_simple, 2: env_full)")
    parser.add_argument("--stage1-model", type=str, default=None,
                        help="2단계 시작 시 사용할 1단계 모델 경로")
    args = parser.parse_args()

    model_dir = mount_drive_if_colab()
    os.makedirs(model_dir, exist_ok=True)

    if args.stage == 1:
        stage1_path = train_stage1(model_dir)
    else:
        s1_path = args.stage1_model or os.path.join(model_dir, "stage1_final")
        if not os.path.exists(s1_path + ".zip"):
            print("[경고] 1단계 모델 없음 → 1단계부터 자동 실행")
            s1_path = train_stage1(model_dir)
        train_stage2(s1_path, model_dir)
