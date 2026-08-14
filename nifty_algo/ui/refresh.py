"""
Auto-refresh, defined once, for every page that needs it.

This used to be `streamlit_autorefresh`, imported inside a try/except that
degraded to a caption - and the package was never in requirements.txt. So on
any fresh install the toggle was a no-op: it looked switched on and nothing
polled. A silent failure in the one mechanism whose entire job is to tell you
something happened.

Streamlit's own `st.fragment(run_every=...)` replaces it with no dependency,
and does it better. `st_autorefresh` re-executed the whole script every tick -
sidebar, chart, tables, scroll position, the lot. A fragment reruns only the
block that wraps the changing content, so the page stops flickering every
15 seconds and your scroll position survives.

The controls live in the sidebar rather than on the Live page because the old
placement had a real consequence: polling only existed on one page, so sitting
on the Daily brief meant the engine was never evaluated at all.
"""
from __future__ import annotations
from datetime import datetime

import streamlit as st

# Fast enough to be useful intraday, slow enough not to hammer a broker API.
INTERVAL_CHOICES = (5, 10, 15, 30, 60)

_TICK_SLOT = "refresh_tick_slot"


def sidebar_controls(cfg) -> None:
    """The toggle, the interval, and a slot for the last-tick stamp."""
    on = st.toggle(
        "Auto-refresh", key="auto_refresh",
        help="Re-evaluates the engine on a timer. Only the live panels "
             "repaint, so the page does not jump.",
    )

    default = min(INTERVAL_CHOICES,
                  key=lambda s: abs(s - cfg.data.poll_seconds))
    st.segmented_control(
        "Interval", INTERVAL_CHOICES, key="refresh_seconds",
        default=default, format_func=lambda s: f"{s}s",
        disabled=not on, label_visibility="collapsed",
    )

    # The fragment writes the stamp here on every tick. Per the st.fragment
    # contract an outside container must receive one write during the initial
    # full app run before a fragment can target it, hence this placeholder.
    slot = st.empty()
    slot.caption("Auto-refresh off — use **Evaluate now**." if not on
                 else "Waiting for the first tick…")
    st.session_state[_TICK_SLOT] = slot


def interval() -> float | None:
    """Seconds between ticks, or None when auto-refresh is off."""
    if not st.session_state.get("auto_refresh"):
        return None
    return float(st.session_state.get("refresh_seconds") or INTERVAL_CHOICES[2])


def live(fn, *args, **kwargs):
    """
    Run `fn` inside a fragment that reruns itself every `interval()` seconds.

    Decorating on each script run is safe: Streamlit derives the fragment id
    from the function's module, name and position in the element tree, not
    from the wrapper object, so the same fragment is re-registered and only
    the auto-rerun interval changes when you move the slider.
    """
    return st.fragment(run_every=interval())(fn)(*args, **kwargs)


def mark_tick() -> None:
    """Stamp the sidebar slot. Called from inside the fragment."""
    slot = st.session_state.get(_TICK_SLOT)
    if slot is None:
        return
    every = interval()
    slot.caption(f"Last refresh {datetime.now():%H:%M:%S}"
                 + (f" · every {every:.0f}s" if every else ""))
