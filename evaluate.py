import argparse
from pathlib import Path

import pandas as pd

from rl.metrics import evaluate, relative_to, summarize
from rl.registry import METHODS, build_agent, ckpt_ext, make_env
from rl.scenarios import build_scenarios
from rl.utils import LOG, get_device, load_config, set_seed, setup_logging


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--methods", nargs="*", default=["fixed_order", *METHODS])
    args = p.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, args.set)
    device = get_device(cfg["device"])
    root = Path(cfg["out_dir"])
    _, test_sc = build_scenarios(cfg)
    frames = []
    for method in args.methods:
        for n in cfg["platoon_sizes"]:
            for seed in cfg["seeds"]:
                set_seed(seed)
                agent = build_agent(method, n, cfg, seed, device)
                if method != "fixed_order":
                    path = root / "agents" / f"{method}_N{n}_s{seed}{ckpt_ext(method)}"
                    if not path.exists():
                        LOG.warning("missing %s", path)
                        continue
                    agent.load(path)
                df = evaluate(agent, make_env(cfg, n, method), test_sc)
                df["method"], df["N"], df["seed"] = method, n, seed
                frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(root / "eval_raw.csv", index=False)
    summ = relative_to(summarize(raw, ["method", "N"]), "fixed_order")
    summ.to_csv(root / "eval_summary.csv", index=False)
    cols = ["method", "N", "reward_mean", "distance_mean", "final_sigma_mean", "frequency_mean",
            "invalid_rate_mean", "over_rate_mean", "runtime_s_mean"]
    print(summ[cols].round(4).to_string())


if __name__ == "__main__":
    main()
