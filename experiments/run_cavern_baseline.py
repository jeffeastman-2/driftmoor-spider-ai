"""Run in a separate process rooted in the chosen Driftmoor checkout."""
import argparse
import json
from pathlib import Path
import sys

if __package__:
    from .game_runtime import game_root, run_game
else:
    from game_runtime import game_root, run_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driftmoor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "resolved", "encounters"), default="baseline")
    parser.add_argument("--seeds", type=int, default=5, help="Encounter seeds starting at zero")
    args = parser.parse_args()
    if not 1 <= args.seeds <= 1000:
        parser.error("--seeds must be between 1 and 1000")
    root = args.driftmoor_root.expanduser().resolve()
    if not (root / "engine/md_loader.py").is_file():
        parser.error("Checkout must contain engine/md_loader.py")
    # Run against the selected game's authored data in an isolated interpreter.
    function = "run_resolved_approach" if args.mode == "resolved" else "run_approach"
    if args.mode == "encounters":
        code = (
            "import json; from spider_sim.spider_experiment import run_encounters; "
            f"print(json.dumps([r for seed in range({args.seeds}) for r in run_encounters(seed)]))"
        )
    else:
        code = (
            f"import json; from spider_sim.spider_experiment import {function}; "
            f"print(json.dumps({function}()))"
        )
    result = run_game(root, code)
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    rows = json.loads(result.stdout)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if args.mode == "encounters":
        from collections import Counter
        counts = Counter((r["episode"]["name"], r["outcome"]["status"]) for r in rows)
        for (name, status), count in sorted(counts.items()):
            print(f"{name:24} {status:26} {count}")
    for row in ([] if args.mode == "encounters" else rows):
        if args.mode == "resolved":
            print(f'{row["elapsed_seconds"]:3}s {row["action"]["actor_id"]:12} '
                  f'{row["action"]["type"]:4} '
                  f'moved={row["outcome"]["distance_moved_ft"]:4.1f}ft '
                  f'contacts={len(row["observer_after"]["contacts"])} '
                  f'active={row["baseline"]["spider_active"]}')
        else:
            print(f'{row["elapsed_seconds"]:3}s {row["label"]:26} '
                  f'active={row["baseline"]["spider_active"]} '
                  f'observations={bool(row["observer_input"]["perception"])}')
    print(f"Saved {len(rows)} records to {output}")


if __name__ == "__main__":
    main()
