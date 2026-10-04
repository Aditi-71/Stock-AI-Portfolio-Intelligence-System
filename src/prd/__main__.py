"""Run with python -m src.prd from the project root."""

from __future__ import annotations

import argparse
import json
import sys

from .config import REQUIRED_MODELS, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PRD stock-return regression and portfolio research"
    )
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "rebuild", "vader"):
        sub.add_parser(command)
    for command in ("train", "all"):
        p = sub.add_parser(command)
        p.add_argument("--models", nargs="+", choices=["naive", *REQUIRED_MODELS])
        p.add_argument("--tickers", nargs="+")
        p.add_argument("--horizons", nargs="+", type=int)
        p.add_argument("--run-name")
    for command in ("eda", "portfolio", "sentiment", "report"):
        p = sub.add_parser(command)
        p.add_argument("--run-dir")
        if command == "portfolio":
            p.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        if args.command in ("prepare", "rebuild"):
            from .prepare import prepare, rebuild

            result = prepare(config) if args.command == "prepare" else rebuild(config)
            print(json.dumps(result, indent=2, default=str))
        elif args.command in ("train", "all"):
            from .experiment import run_experiment

            result = run_experiment(config, args.models, args.tickers, args.horizons, args.run_name)
            print(f"Model results: {result}")
            if args.command == "all":
                from .eda import run_eda
                from .portfolio import run_portfolio
                from .report import write_report
                from .sentiment import run_sentiment

                run_eda(config, result)
                run_portfolio(config, result)
                run_sentiment(config, result)
                print(write_report(config, result))
        elif args.command == "eda":
            from .eda import run_eda

            print(run_eda(config, args.run_dir))
        elif args.command == "portfolio":
            from .portfolio import run_portfolio

            print(run_portfolio(config, args.run_dir, args.horizon))
        elif args.command == "vader":
            from .sentiment import score_vader

            print(score_vader(config))
        elif args.command == "sentiment":
            from .sentiment import run_sentiment

            print(run_sentiment(config, args.run_dir))
        elif args.command == "report":
            from .report import write_report

            print(write_report(config, args.run_dir))
    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        FileExistsError,
        ModuleNotFoundError,
    ) as error:
        print(f"PRD command could not finish: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
