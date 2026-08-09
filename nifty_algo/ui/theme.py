"""
Chart palette and shared CSS.

Colours are taken from the validated reference palette and were re-checked
with the palette validator for the subset actually used here (blue + orange
categorical, both modes): all six checks pass.

Role assignment:

  candles       status good/critical - up and down are genuinely a state, not
                an identity, so status tokens are correct rather than
                categorical slots.
  VWAP          categorical slot 1 (blue)
  trendline     categorical slot 2 (orange)
  levels        muted ink - they are chrome, not data series
  entry/target/stop  status tokens, each direct-labelled so colour never
                carries the meaning alone.
"""
from __future__ import annotations

# ---- categorical (identity) ----
SERIES_1_LIGHT = "#2a78d6"      # blue   - VWAP
SERIES_1_DARK = "#3987e5"
SERIES_2_LIGHT = "#eb6834"      # orange - trendline
SERIES_2_DARK = "#d95926"

# ---- status (state) - fixed, never themed ----
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

# ---- chrome & ink ----
SURFACE_LIGHT = "#fcfcfb"
SURFACE_DARK = "#1a1a19"
INK_PRIMARY_LIGHT = "#0b0b0b"
INK_PRIMARY_DARK = "#ffffff"
INK_SECONDARY_LIGHT = "#52514e"
INK_SECONDARY_DARK = "#c3c2b7"
MUTED = "#898781"
GRID_LIGHT = "#e1e0d9"
GRID_DARK = "#2c2c2a"
AXIS_LIGHT = "#c3c2b7"
AXIS_DARK = "#383835"


class Palette:
    """Resolved colours for one theme mode."""

    def __init__(self, dark: bool):
        self.dark = dark
        self.surface = SURFACE_DARK if dark else SURFACE_LIGHT
        self.ink = INK_PRIMARY_DARK if dark else INK_PRIMARY_LIGHT
        self.ink_secondary = INK_SECONDARY_DARK if dark else INK_SECONDARY_LIGHT
        self.muted = MUTED
        self.grid = GRID_DARK if dark else GRID_LIGHT
        self.axis = AXIS_DARK if dark else AXIS_LIGHT
        self.series_1 = SERIES_1_DARK if dark else SERIES_1_LIGHT
        self.series_2 = SERIES_2_DARK if dark else SERIES_2_LIGHT
        self.good = GOOD
        self.warning = WARNING
        self.serious = SERIOUS
        self.critical = CRITICAL


def get_palette() -> Palette:
    """Follow the Streamlit theme so charts match the app chrome."""
    try:
        import streamlit as st
        base = st.get_option("theme.base")
        return Palette(dark=(base == "dark"))
    except Exception:
        return Palette(dark=False)


def layout_defaults(p: Palette, height: int = 520) -> dict:
    """
    Recessive chrome: hairline solid gridlines one shade off the surface, no
    dashes, generous padding, transparent plot so the Streamlit card shows
    through.
    """
    axis = dict(
        showgrid=True, gridcolor=p.grid, gridwidth=1,
        zeroline=False, linecolor=p.axis, linewidth=1,
        tickfont=dict(color=p.muted, size=11),
        title_font=dict(color=p.ink_secondary, size=12),
    )
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  color=p.ink_secondary, size=12),
        margin=dict(l=8, r=8, t=36, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=p.surface, bordercolor=p.axis,
                        font=dict(color=p.ink, size=12)),
        xaxis=axis, yaxis=axis,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0,
                    font=dict(color=p.ink_secondary, size=11),
                    bgcolor="rgba(0,0,0,0)"),
    )


CSS = """
<style>
  .alert-card {
    border-left: 4px solid var(--alert-accent, #0ca30c);
    background: rgba(127,127,127,0.06);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.8rem;
  }
  .alert-card h4 { margin: 0 0 .5rem 0; font-size: 1.05rem; }
  .alert-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(105px, 1fr));
    gap: .55rem 1.1rem;
    margin: .6rem 0;
  }
  .alert-grid div { display: flex; flex-direction: column; }
  .alert-grid .lbl {
    font-size: .68rem; letter-spacing: .06em; text-transform: uppercase;
    color: #898781;
  }
  .alert-grid .val { font-size: 1.02rem; font-weight: 600; }
  .alert-why {
    font-size: .84rem; color: #898781; margin-top: .35rem; line-height: 1.45;
  }
  .banner {
    border-radius: 8px; padding: .6rem .9rem; margin-bottom: .7rem;
    font-size: .87rem; line-height: 1.45;
    border-left: 4px solid var(--banner-accent, #fab219);
    background: rgba(127,127,127,0.06);
  }
  .pill {
    display: inline-block; padding: .1rem .55rem; border-radius: 999px;
    font-size: .72rem; font-weight: 600; letter-spacing: .03em;
    border: 1px solid rgba(127,127,127,.35);
  }
  /* Tabular figures for columns that must align; proportional elsewhere. */
  .alert-grid .val, .stDataFrame { font-variant-numeric: tabular-nums; }
</style>
"""
