import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rl.registry import METHODS, build_agent, ckpt_ext, make_env, train_config
from rl.scenarios import build_scenarios
from rl.utils import LOG, get_device, load_config, set_seed, setup_logging


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--methods", nargs="*", default=list(METHODS))
    p.add_argument("--sizes", nargs="*", type=int)
    p.add_argument("--seeds", nargs="*", type=int)
    args = p.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, args.set)
    device = get_device(cfg["device"])
    out = Path(cfg["out_dir"]) / "agents"
    out.mkdir(parents=True, exist_ok=True)
    train_sc, _ = build_scenarios(cfg)
    for method in args.methods:
        for n in args.sizes or cfg["platoon_sizes"]:
            for seed in args.seeds or cfg["seeds"]:
                set_seed(seed)
                env = make_env(cfg, n, method)
                agent = build_agent(method, n, cfg, seed, device)
                log = lambda d: LOG.info("[%s N=%d s=%d] %s", method, n, seed,
                                         {k: round(float(v), 3) for k, v in d.items()})
                hist = agent.train(env, train_sc, train_config(method, cfg), np.random.default_rng(seed), log)
                name = f"{method}_N{n}_s{seed}"
                agent.save(out / f"{name}{ckpt_ext(method)}")
                pd.DataFrame(hist).to_csv(out / f"{name}_history.csv", index=False)


if __name__ == "__main__":
    main()
