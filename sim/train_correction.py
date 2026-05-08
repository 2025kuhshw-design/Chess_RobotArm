"""
RL 보정 모델 학습 스크립트 (PPO, stable-baselines3)
1단계: env_simple → 5,000,000 스텝
2단계: env_full  → 1,500,000 스텝 (1단계 이어서)

Google Colab A100 실행 지원:
  - 드라이브 마운트 자동 처리
  - 세션 재시작 시 최신 체크포인트 자동 로드
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim.env_simple import REACH_FINE

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수
# ─────────────────────────────────────────
STAGE1_STEPS    = 2_000_000
STAGE2_STEPS    = 1_500_000
N_ENVS          = 4           # 병렬 환경 수 (샘플 다양성 ↑)
N_STEPS         = 2048        # 환경당 롤아웃 길이
BATCH_SIZE      = 256         # 4envs × 2048 / 32 minibatches
ENT_COEF_S1     = 0.01        # 1단계: 탐색 강화
ENT_COEF_S2     = 0.005       # 2단계: 수렴 안정화
EVAL_FREQ       = 50_000      # EvalCallback 평가 주기 (스텝)
N_EVAL_EPS      = 10          # 평가 에피소드 수
CHECKPOINT_FREQ = 50_000      # 체크포인트 저장 주기

LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "correction_model")
COLAB_DRIVE_DIR = "/content/drive/MyDrive/chess_robot/correction_model"
IS_COLAB        = "google.colab" in sys.modules or os.path.exists("/content")


# ─────────────────────────────────────────
# 학습률 선형 감소 스케줄
# ─────────────────────────────────────────
def linear_schedule(initial_value: float):
    """progress_remaining: 1.0(시작) → 0.0(종료) 선형 감소."""
    def func(progress_remaining: float) -> float:
        return max(progress_remaining * initial_value, 1e-5)
    return func


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
# 최신 체크포인트 탐색
# ─────────────────────────────────────────
def find_latest_checkpoint(model_dir: str, prefix: str) -> str | None:
    pattern = os.path.join(model_dir, f"{prefix}_*_steps.zip")
    files   = glob.glob(pattern)
    if not files:
        return None
    def extract_steps(path):
        try:
            return int(os.path.basename(path).split("_")[-2])
        except Exception:
            return 0
    return max(files, key=extract_steps)


# ─────────────────────────────────────────
# VecEnv 생성 (병렬 환경 + 보상 정규화)
# ─────────────────────────────────────────
def _make_env(env_class, rank: int, render: bool = False):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        # render=True면 모든 env에 동일한 render_mode 전달 (SB3 DummyVecEnv 요구사항)
        # 실제 GUI 창은 env_simple._gui_open 플래그로 첫 번째 env만 열림
        mode = "human" if render else None
        env = Monitor(env_class(render_mode=mode))
        env.reset(seed=rank)
        return env
    return _init


def load_vec_env(env_class, n_envs: int, norm_path: str, render: bool = False):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    venv = DummyVecEnv([_make_env(env_class, i, render=render) for i in range(n_envs)])
    if os.path.exists(norm_path):
        venv = VecNormalize.load(norm_path, venv)
        venv.training = True
        print(f"[VecNormalize] 통계 로드: {norm_path}")
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)
        print("[VecNormalize] 새로 시작")
    return venv


# ─────────────────────────────────────────
# 체크포인트 + VecNormalize 동시 저장 콜백
# ─────────────────────────────────────────
def make_checkpoint_callback(save_dir: str, prefix: str, vec_env):
    from stable_baselines3.common.callbacks import BaseCallback
    os.makedirs(save_dir, exist_ok=True)
    _last_save = [0]

    class _Callback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=1)

        def _on_step(self) -> bool:
            if self.num_timesteps - _last_save[0] >= CHECKPOINT_FREQ:
                _last_save[0] = self.num_timesteps
                path = os.path.join(save_dir, f"{prefix}_{self.num_timesteps}_steps")
                self.model.save(path)
                vec_env.save(os.path.join(save_dir, f"{prefix}_vecnorm.pkl"))
                print(f"  체크포인트 저장: {path}.zip")
            return True

    return _Callback()


# ─────────────────────────────────────────
# EvalCallback (과적합 감지 + best model 저장)
# ─────────────────────────────────────────
def make_eval_callback(env_class, model_dir: str):
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    # 학습 env와 동일하게 VecNormalize로 감싸되 training=False, norm_reward=False
    # → 관측 정규화 통계 동기화 허용 + 실제 보상값 그대로 측정
    eval_env = DummyVecEnv([lambda: Monitor(env_class())])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)
    return EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(model_dir, "best"),
        log_path             = os.path.join(model_dir, "eval_logs"),
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = N_EVAL_EPS,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )


# ─────────────────────────────────────────
# TensorBoard 커스텀 메트릭 콜백
# ─────────────────────────────────────────
def make_metrics_callback():
    from stable_baselines3.common.callbacks import BaseCallback

    class _Callback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._ep_dists   = []
            self._ep_success = []
            self._ep_rewards = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for info, done in zip(infos, dones):
                if done and "dist_cm" in info:
                    self._ep_dists.append(info["dist_cm"])
                    self._ep_success.append(float(info["dist_cm"] < REACH_FINE * 100))
                if done and "episode" in info:
                    self._ep_rewards.append(info["episode"]["r"])

            if len(self._ep_dists) >= 20:
                self.logger.record("custom/mean_dist_cm", float(np.mean(self._ep_dists)))
                self.logger.record("custom/success_rate", float(np.mean(self._ep_success)))
                if self._ep_rewards:
                    self.logger.record("custom/mean_reward", float(np.mean(self._ep_rewards)))
                self._ep_dists.clear()
                self._ep_success.clear()
                self._ep_rewards.clear()
            return True

    return _Callback()


# ─────────────────────────────────────────
# 1단계 학습
# ─────────────────────────────────────────
def train_stage1(model_dir: str, render: bool = False) -> str:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from sim.env_simple import ChessArmEnvSimple

    print("\n" + "="*50)
    print("1단계 학습 시작 (env_simple, PPO)")
    print("="*50)

    norm_path = os.path.join(model_dir, "stage1_vecnorm.pkl")
    ckpt_path = find_latest_checkpoint(model_dir, "stage1")
    env       = load_vec_env(ChessArmEnvSimple, N_ENVS, norm_path, render=render)

    from stable_baselines3.common.utils import get_schedule_fn

    from stable_baselines3.common.logger import configure as sb3_configure

    if ckpt_path:
        print(f"[체크포인트 발견] 이어서 학습: {ckpt_path}")
        model = PPO.load(ckpt_path, env=env)
        model.learning_rate = linear_schedule(3e-4)
        model.lr_schedule   = get_schedule_fn(model.learning_rate)
        model.ent_coef      = ENT_COEF_S1
    else:
        print("[새로 학습 시작]")
        model = PPO(
            "MlpPolicy", env,
            learning_rate   = linear_schedule(3e-4),
            n_steps         = N_STEPS,
            batch_size      = BATCH_SIZE,
            ent_coef        = ENT_COEF_S1,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            max_grad_norm   = 0.5,
            verbose         = 1,
        )

    # 체크포인트 재시작마다 같은 폴더에 기록 (PPO_1, PPO_2 분산 방지)
    model.set_logger(sb3_configure(
        os.path.join(model_dir, "tb_logs", "stage1"), ["stdout", "tensorboard"]
    ))
    remaining = max(0, STAGE1_STEPS - model.num_timesteps)

    callbacks = CallbackList([
        make_checkpoint_callback(model_dir, "stage1", env),
        make_metrics_callback(),
        make_eval_callback(ChessArmEnvSimple, model_dir),
    ])

    try:
        if remaining > 0:
            # total_timesteps = 절대 목표값: num_timesteps가 STAGE1_STEPS에 도달할 때까지 학습
            model.learn(total_timesteps=STAGE1_STEPS, callback=callbacks, reset_num_timesteps=False)

        save_path = os.path.join(model_dir, "stage1_final")
        model.save(save_path)
        env.save(norm_path)
    finally:
        env.close()

    print(f"\n✅ 1단계 학습 완료: {save_path}")
    return save_path


# ─────────────────────────────────────────
# 2단계 학습
# ─────────────────────────────────────────
def train_stage2(stage1_path: str, model_dir: str, render: bool = False) -> str:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from sim.env_full import ChessArmEnvFull

    print("\n" + "="*50)
    print("2단계 학습 시작 (env_full, Domain Randomization)")
    print("="*50)

    # 2단계 절대 목표 스텝 = 1단계 종료 시점 + 2단계 추가 스텝
    # model.num_timesteps는 stage1 완료 후 STAGE1_STEPS이므로 이 기준으로 remaining 계산
    total_target = STAGE1_STEPS + STAGE2_STEPS

    norm_path = os.path.join(model_dir, "stage2_vecnorm.pkl")
    ckpt_path = find_latest_checkpoint(model_dir, "stage2")
    env       = load_vec_env(ChessArmEnvFull, N_ENVS, norm_path, render=render)

    from stable_baselines3.common.utils import get_schedule_fn

    from stable_baselines3.common.logger import configure as sb3_configure

    if ckpt_path:
        print(f"[체크포인트 발견] 이어서 학습: {ckpt_path}")
        model = PPO.load(ckpt_path, env=env)
    else:
        print(f"[1단계 모델 로드] {stage1_path}")
        model = PPO.load(stage1_path, env=env)

    model.learning_rate = linear_schedule(3e-4)
    model.lr_schedule   = get_schedule_fn(model.learning_rate)
    model.ent_coef      = ENT_COEF_S2

    # 체크포인트 재시작마다 같은 폴더에 기록 (PPO_1, PPO_2 분산 방지)
    model.set_logger(sb3_configure(
        os.path.join(model_dir, "tb_logs", "stage2"), ["stdout", "tensorboard"]
    ))
    remaining = max(0, total_target - model.num_timesteps)

    callbacks = CallbackList([
        make_checkpoint_callback(model_dir, "stage2", env),
        make_metrics_callback(),
        make_eval_callback(ChessArmEnvFull, model_dir),
    ])

    try:
        if remaining > 0:
            # total_timesteps = 절대 목표값: 1단계 이어서 total_target에 도달할 때까지 학습
            model.learn(total_timesteps=total_target, callback=callbacks, reset_num_timesteps=False)

        save_path = os.path.join(model_dir, "stage2_final")
        model.save(save_path)
        env.save(norm_path)
    finally:
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
    parser.add_argument("--render", action="store_true",
                        help="env[0]을 PyBullet GUI로 열어 학습 과정 실시간 시각화")
    args = parser.parse_args()

    model_dir = mount_drive_if_colab()
    os.makedirs(model_dir, exist_ok=True)

    if args.stage == 1:
        train_stage1(model_dir, render=args.render)
    else:
        s1_path = args.stage1_model or os.path.join(model_dir, "stage1_final")
        if not os.path.exists(s1_path + ".zip"):
            # stage1_final 없으면 최신 체크포인트로 대체 (중간 중단 시 대비)
            ckpt = find_latest_checkpoint(model_dir, "stage1")
            if ckpt:
                s1_path = ckpt[:-4]  # .zip 제거 (PPO.load 호환)
                print(f"[stage1_final 없음] 최신 체크포인트로 대체: {ckpt}")
            else:
                print("[경고] 1단계 모델 없음 → 1단계부터 자동 실행")
                s1_path = train_stage1(model_dir, render=args.render)
        train_stage2(s1_path, model_dir, render=args.render)
