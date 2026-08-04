"""
학습 속도 측정 — 재학습에 몇 시간 걸릴지 내 컴퓨터에서 직접 확인.

PPO 학습 루프를 짧게 돌려 초당 스텝 수(FPS)를 재고, 목표 스텝까지의
예상 시간을 계산한다. 2~3분이면 끝난다.

실행:
  python sim/benchmark.py                 # CPU (권장)
  python sim/benchmark.py --device cuda   # GPU와 비교해보고 싶을 때

참고: 이 학습은 GPU 이득이 거의 없다. 정책망이 작은 MLP라 GPU 전송
오버헤드가 연산 이득을 상쇄하고, 물리 시뮬(PyBullet)은 애초에 CPU에서 돈다.
보통 device=cpu가 같거나 더 빠르다.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--n-envs", type=int, default=None, help="기본: 학습 설정값")
    ap.add_argument("--measure-steps", type=int, default=16384,
                    help="측정에 쓸 스텝 수 (기본 16384)")
    args = ap.parse_args()

    from sim.env_simple import ChessArmEnvSimple
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import sim.train_correction as tc

    n_envs = args.n_envs or tc.N_ENVS
    print(f"[benchmark] device={args.device}  n_envs={n_envs}  "
          f"n_steps={tc.N_STEPS}  batch={tc.BATCH_SIZE}")

    venv = DummyVecEnv([lambda: ChessArmEnvSimple() for _ in range(n_envs)])
    model = PPO("MlpPolicy", venv, n_steps=tc.N_STEPS,
                batch_size=tc.BATCH_SIZE, verbose=0, device=args.device)

    # 워밍업 — 초기화·컴파일 오버헤드를 측정에서 제외
    warm = tc.N_STEPS * n_envs
    print(f"  워밍업 {warm:,} 스텝...")
    model.learn(total_timesteps=warm)

    print(f"  측정 {args.measure_steps:,} 스텝...")
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.measure_steps, reset_num_timesteps=False)
    dt = time.perf_counter() - t0
    venv.close()

    fps = args.measure_steps / dt
    print(f"\n  ▶ 처리량: {fps:,.0f} steps/sec\n")

    print(f"  {'목표 스텝':>12}  {'예상 시간':>10}   비고")
    plans = [
        (300_000,  "최소 — 보정 효과가 보이기 시작"),
        (500_000,  "권장 하한 — 안정적인 개선"),
        (1_000_000, "충분"),
        (tc.STAGE1_STEPS, "1단계 기본 설정"),
        (tc.STAGE1_STEPS + tc.STAGE2_STEPS, "1+2단계 전체 (기본값)"),
    ]
    for steps, note in plans:
        h = steps / fps / 3600
        t = f"{h*60:.0f}분" if h < 1 else f"{h:.1f}시간"
        print(f"  {steps:>12,}  {t:>10}   {note}")

    print("\n  ※ 50,000 스텝마다 체크포인트가 저장되므로 중간에 끊고")
    print("    나중에 이어서 학습할 수 있다 (같은 명령 재실행).")


if __name__ == "__main__":
    main()
