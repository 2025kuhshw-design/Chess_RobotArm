"""
학습된 PPO 모델 데모 실행 (GUI 시각화)
python sim/demo.py
python sim/demo.py --model models/correction_model/stage2_final
python sim/demo.py --episodes 5
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "correction_model")


def find_best_model(model_dir: str) -> tuple[str | None, str | None]:
    """사용 가능한 모델 중 우선순위대로 탐색. (모델경로, vecnorm경로) 반환."""
    import glob

    candidates = [
        ("stage2_final",   "stage2_vecnorm.pkl"),
        ("best/best_model","stage2_vecnorm.pkl"),
        ("stage1_final",   "stage1_vecnorm.pkl"),
        ("best/best_model","stage1_vecnorm.pkl"),
    ]
    for model_rel, norm_rel in candidates:
        model_path = os.path.join(model_dir, model_rel)
        norm_path  = os.path.join(model_dir, norm_rel)
        if os.path.exists(model_path + ".zip"):
            return model_path, norm_path if os.path.exists(norm_path) else None

    # 체크포인트 중 가장 최신 것
    for prefix in ["stage2", "stage1"]:
        files = glob.glob(os.path.join(model_dir, f"{prefix}_*_steps.zip"))
        if files:
            best = max(files, key=lambda p: int(os.path.basename(p).split("_")[-2]))
            norm = os.path.join(model_dir, f"{prefix}_vecnorm.pkl")
            return best[:-4], norm if os.path.exists(norm) else None

    return None, None


def run_demo(model_path: str, vecnorm_path: str | None, n_episodes: int):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from sim.env_full import ChessArmEnvFull

    print(f"[모델] {model_path}.zip")
    print(f"[VecNorm] {vecnorm_path or '없음 (정규화 미적용)'}")

    env = DummyVecEnv([lambda: Monitor(ChessArmEnvFull(render_mode="human"))])

    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training   = False
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=False, training=False)

    model = PPO.load(model_path, env=env, custom_objects={
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
        "exploration_schedule": lambda _: 0.0,
    })

    print("\n[조작법]  S: 속도 토글(느림↔빠름)  ←→↑↓: 카메라 회전  Ctrl+C: 종료\n")

    ep = 0
    try:
        while n_episodes == 0 or ep < n_episodes:
            obs  = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_reward += float(reward[0])
                steps += 1

            dist_cm      = info[0].get("dist_cm", 0.0)
            threshold_cm = info[0].get("reach_fine_cm", 2.0)
            success      = dist_cm < threshold_cm
            print(f"  ep {ep+1:3d} | {steps:3d}스텝 | 거리 {dist_cm:.2f}cm "
                  f"(기준 {threshold_cm:.0f}cm) | {'✅ 성공' if success else '❌ 실패'} "
                  f"| 보상 {ep_reward:.0f}")
            ep += 1

    except KeyboardInterrupt:
        print("\n[종료]")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO 보정 모델 데모")
    parser.add_argument("--model",    default=None, help="모델 경로 (.zip 제외)")
    parser.add_argument("--episodes", type=int, default=0, help="에피소드 수 (0=무한)")
    args = parser.parse_args()

    model_dir = MODEL_DIR

    if args.model:
        model_path  = args.model
        vecnorm_path = os.path.join(model_dir, "stage2_vecnorm.pkl")
        if not os.path.exists(vecnorm_path):
            vecnorm_path = os.path.join(model_dir, "stage1_vecnorm.pkl")
    else:
        model_path, vecnorm_path = find_best_model(model_dir)

    if model_path is None or not os.path.exists(model_path + ".zip"):
        print("[오류] 사용 가능한 모델 파일이 없습니다.")
        print(f"  찾은 위치: {model_dir}")
        print("  학습을 먼저 완료하거나 --model 옵션으로 경로를 지정하세요.")
        sys.exit(1)

    run_demo(model_path, vecnorm_path, args.episodes)
