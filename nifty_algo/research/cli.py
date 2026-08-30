"""
The research briefings, headless.

    python -m nifty_algo.research macro --market india
    python -m nifty_algo.research risk  --market india --json

WHY THE CLI IS THE PRODUCT AND NOT AN AFTERTHOUGHT. These packs are consumed
by a Claude Code skill, which runs this command and writes the prose from what
comes back. So `--json` is the real interface and the human-readable form is
the courtesy: the JSON carries `unavailable` and `judgment_required` as
top-level lists precisely so the skill can check its two hard rules - cite no
number that is not in the pack, and name every fact that is missing - without
walking the tree.

UNBUFFERED, `flush=True` on every progress line. A long-running command that
writes nothing for two minutes is one you cannot tell apart from a hung one -
this repo has already lost 101 CPU-minutes to exactly that.
"""
from __future__ import annotations

import argparse
import sys

from ..config import DEFAULT
from ..swing import markets as markets_mod
from . import macro, risk_report

#: Registry, for the same reason every other registry in this repo exists: a
#: report that the CLI can run and the UI cannot list is the bug this prevents.
REPORTS = {
    macro.REPORT: (macro.TITLE, macro.build),
    risk_report.REPORT: (risk_report.TITLE, risk_report.build),
}


def main(argv=None) -> int:
    cfg = DEFAULT
    parser = argparse.ArgumentParser(
        description="Institutional-style research briefings, as fact packs.")
    parser.add_argument("report", choices=sorted(REPORTS),
                        help="which briefing to build")
    parser.add_argument("--market", default=cfg.swing.default_market,
                        choices=markets_mod.keys(cfg))
    parser.add_argument("--json", action="store_true",
                        help="emit the fact pack as JSON on stdout - the form "
                             "a skill consumes")
    parser.add_argument("--refresh", action="store_true",
                        help="force a fresh download of the macro series")
    parser.add_argument("--out", default="",
                        help="write the JSON to this path as well as stdout")
    args = parser.parse_args(argv)

    title, build = REPORTS[args.report]
    quiet = args.json
    pack = build(cfg, market_key=args.market, force_refresh=args.refresh,
                 progress=None if quiet else _progress)

    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(pack.to_json(), encoding="utf-8")
        if not quiet:
            print(f"written to {args.out}", flush=True)

    if args.json:
        print(pack.to_json())
        return 0

    _render(pack, title)
    return 0


def _progress(message: str) -> None:
    print(f"  ... {message}", flush=True)


def _render(pack, title: str) -> None:
    """
    The human form. Deliberately plain text - a terminal report that needs a
    wide window is one you stop reading.
    """
    print()
    print(title.upper())
    print(f"generated {pack.generated_at} | market "
          f"{pack.inputs.get('market_label', '?')}")

    if pack.stood_down:
        print()
        print("STOOD DOWN: " + pack.stood_down)

    if pack.caveats:
        print()
        print("READ THESE FIRST")
        for c in pack.caveats:
            print(_wrap(f"  - {c}"))

    for section in pack.sections:
        print()
        print(f"== {section.name}")
        for fact in section.facts:
            print(f"   {fact.label}: {fact.display()}")
            # An unavailable fact already prints its reason inside display();
            # printing `note` again would say the same sentence twice.
            if fact.note and fact.available:
                print(_wrap(f"       {fact.note}", indent=7))
        if section.rows:
            _table(section.rows)
        if section.note:
            print(_wrap(f"   {section.note}", indent=3))
        for question in section.judgment:
            print(_wrap(f"   [JUDGMENT] {question}", indent=3))

    missing = pack.unavailable()
    print()
    print(f"== {len(missing)} fact(s) this run could NOT establish")
    for m in missing:
        print(_wrap(f"   {m['section']} / {m['label']}: {m['note']}", indent=3))
    print()
    print("A briefing written from this pack must name every line above as "
          "unavailable rather than skip it, and must cite no figure that is "
          "not in the pack.")


def _table(rows: list[dict], limit: int = 12) -> None:
    columns = list(rows[0])
    widths = {c: max(len(str(c)), *(len(_cell(r.get(c))) for r in rows))
              for c in columns}
    print("   " + " | ".join(str(c).ljust(widths[c]) for c in columns))
    print("   " + "-+-".join("-" * widths[c] for c in columns))
    for row in rows[:limit]:
        print("   " + " | ".join(_cell(row.get(c)).ljust(widths[c])
                                 for c in columns))
    if len(rows) > limit:
        print(f"   ... {len(rows) - limit} more row(s); use --json for all")


def _cell(value) -> str:
    """`None` prints as `n/a`, never as a blank - a blank reads as a zero."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.4g}" if abs(value) < 1000 else f"{value:,.2f}"
    text = str(value)
    return text if len(text) <= 40 else text[:37] + "..."


def _wrap(text: str, indent: int = 0, width: int = 88) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width,
                                   subsequent_indent=" " * indent)) or text


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(main())
