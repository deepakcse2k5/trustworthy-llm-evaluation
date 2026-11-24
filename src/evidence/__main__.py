"""Command line.

python -m evidence compare RESULTS_A RESULTS_B --items ITEMS.jsonl
    Results files are JSON objects mapping question id to 1 (pass) or 0 (fail).
python -m evidence size rate --rate 0.8 --half-width 0.03
python -m evidence size paired --discordant 0.08 --effect 0.03
python -m evidence size detectable --questions 1000 --discordant 0.08
"""

import argparse
import json
from pathlib import Path

from evidence.compare import compare, format_report
from evidence.dataset import load
from evidence.power import detectable_effect, n_for_precision, n_paired


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evidence")
    sub = parser.add_subparsers(dest="command", required=True)
    cmp = sub.add_parser("compare", help="compare two versions run on the same questions")
    cmp.add_argument("results_a")
    cmp.add_argument("results_b")
    cmp.add_argument("--items", required=True, help="the questions both versions answered")
    cmp.add_argument("--labels", nargs=2, default=["A", "B"])
    size = sub.add_parser("size", help="plan how many questions an evaluation needs")
    size.add_argument("kind", choices=["rate", "paired", "detectable"])
    size.add_argument("--rate", type=float, help="expected pass rate (rate)")
    size.add_argument("--half-width", type=float, help="interval half-width wanted (rate)")
    size.add_argument("--discordant", type=float, help="share of questions that differ")
    size.add_argument("--effect", type=float, help="difference to detect (paired)")
    size.add_argument("--questions", type=int, help="questions available (detectable)")
    size.add_argument("--power", type=float, default=0.8)
    args = parser.parse_args()

    if args.command == "size":
        if args.kind == "rate":
            n = n_for_precision(args.rate, args.half_width)
            print(f"{n:,} questions for +/- {args.half_width:.1%} around {args.rate:.0%}")
        elif args.kind == "paired":
            n = n_paired(args.discordant, args.effect, power=args.power)
            print(f"{n:,} questions to detect {args.effect:+.1%} with {args.power:.0%} power")
        else:
            d = detectable_effect(args.questions, args.discordant, power=args.power)
            print(f"{args.questions:,} questions detect a difference of {d:.1%} or more")
        return

    a = json.loads(Path(args.results_a).read_text())
    b = json.loads(Path(args.results_b).read_text())
    items = load(args.items)
    missing = [i.id for i in items if i.id not in a or i.id not in b]
    if missing:
        raise SystemExit(f"{len(missing)} questions lack a result in one of the files")
    print(format_report(compare(a, b, items), *args.labels))


if __name__ == "__main__":
    main()
