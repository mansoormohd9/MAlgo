
"""
Plotly figures.

Design rules applied throughout (see the dataviz method):
  - One y-axis per plot. Never two scales - volume gets its own subplot row
    rather than a secondary axis, because overlaying price and volume on two
    scales invents a correlation the data does not contain.
  - Solid hairline gridlines, no dashes on chrome. Dashes are reserved for
    marks that genuinely mean "threshold" - the target and stop lines.
  - Legend present whenever two or more series are drawn; the entry, target,
    and stop lines are additionally direct-labelled, so colour never carries
    meaning alone.
  - Every figure has a table-view twin in the page that renders it.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import Palette, layout_defaults
from .. import signals as sig


def price_chart(bars: pd.DataFrame, palette: Palette,
                levels=None, vwap_series: pd.Series | None = None,
                trendlines=None, alert=None, title: str = "") -> go.Figure:
    """Candlesticks with the structures the strategies actually used."""
    p = palette
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.04,
    )

    # Up/down are a STATE, so status tokens are correct here rather than
    # categorical identity slots.
    fig.add_trace(go.Candlestick(
        x=bars.index, open=bars["open"], high=bars["high"],
        low=bars["low"], close=bars["close"],
        name="Price",
        increasing=dict(line=dict(color=p.good, width=1), fillcolor=p.good),
        decreasing=dict(line=dict(color=p.critical, width=1), fillcolor=p.critical),
        showlegend=False,
    ), row=1, col=1)

    if vwap_series is not None and not vwap_series.dropna().empty:
        fig.add_trace(go.Scatter(
            x=vwap_series.index, y=vwap_series, name="VWAP",
            line=dict(color=p.series_1, width=2), mode="lines",
        ), row=1, col=1)

    # Levels are chrome, not a data series - muted ink, no legend entry.
    for lv in (levels or [])[:10]:
        fig.add_hline(
            y=lv.price, line=dict(color=p.muted, width=1),
            annotation_text=f"{lv.kind[:3]} {lv.price:.0f} ({lv.touches})",
            annotation_position="right",
            annotation_font=dict(size=9, color=p.muted),
            row=1, col=1,
        )

    for tl in (trendlines or []):
        xs = list(bars.index)
        ys = [tl.value_at(i) for i in range(len(bars))]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, name=f"{tl.kind.replace('_', ' ')} (R² {tl.r_squared:.2f})",
            line=dict(color=p.series_2, width=2), mode="lines",
        ), row=1, col=1)

    if alert is not None and alert.underlying_price:
        stop_pts = alert.underlying_stop_points
        long_side = alert.direction == "long"
        entry = alert.underlying_price
        stop = entry - stop_pts if long_side else entry + stop_pts
        rr = 2.0
        target = entry + stop_pts * rr if long_side else entry - stop_pts * rr

        for y, label, colour in (
            (entry, "ENTRY", p.series_1),
            (target, "TARGET", p.good),
            (stop, "STOP", p.critical),
        ):
            fig.add_hline(
                y=y, line=dict(color=colour, width=2, dash="dash"),
                annotation_text=f"{label} {y:,.0f}",
                annotation_position="left",
                annotation_font=dict(size=11, color=colour),
                row=1, col=1,
            )

    fig.add_trace(go.Bar(
        x=bars.index, y=bars["volume"], name="Volume",
        marker=dict(color=p.muted, line=dict(width=0)),
        opacity=0.55, showlegend=False,
    ), row=2, col=1)

    layout = layout_defaults(p, height=560)
    legend = layout.pop("legend")
    xaxis = layout.pop("xaxis")
    yaxis = layout.pop("yaxis")
    fig.update_layout(**layout, legend=legend, title=None,
                      xaxis_rangeslider_visible=False, bargap=0.15)
    fig.update_xaxes(**xaxis, row=1, col=1)
    fig.update_xaxes(**xaxis, row=2, col=1)
    fig.update_yaxes(**yaxis, title_text="Price", row=1, col=1)
    fig.update_yaxes(**yaxis, title_text="Volume", row=2, col=1)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color=p.ink, size=14)))
    return fig


def equity_curve(cum_r: pd.Series, palette: Palette) -> go.Figure:
    """
    Cumulative R. One series, so no legend box - the title names it.
    The zero line is the reference that matters, drawn as a threshold.
    """
    p = palette
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cum_r) + 1)), y=cum_r.values,
        mode="lines", name="Cumulative R",
        line=dict(color=p.series_1, width=2),
        hovertemplate="Trade %{x}<br>Cumulative %{y:+.2f} R<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=p.axis, width=1))

    if len(cum_r):
        final = float(cum_r.iloc[-1])
        fig.add_annotation(
            x=len(cum_r), y=final, text=f"{final:+.2f} R",
            showarrow=False, xanchor="left", xshift=8,
            font=dict(color=p.good if final >= 0 else p.critical, size=12),
        )

    layout = layout_defaults(p, height=340)
    layout["hovermode"] = "x"
    fig.update_layout(**layout, showlegend=False,
                      title=dict(text="Equity curve — cumulative R, net of friction",
                                 font=dict(color=p.ink, size=14)))
    fig.update_xaxes(title_text="Trade number")
    fig.update_yaxes(title_text="Cumulative R")
    return fig


def expectancy_by_strategy(by_strategy: dict, palette: Palette,
                           labels: dict[str, str] | None = None) -> go.Figure:
    """
    Expectancy per strategy. Polarity is the story - above or below zero -
    so this is a diverging encoding: blue/red poles about a neutral zero.
    """
    p = palette
    labels = labels or {}
    rows = sorted(by_strategy.items(), key=lambda kv: kv[1].expectancy_r)
    names = [labels.get(k, k) for k, _ in rows]
    vals = [m.expectancy_r for _, m in rows]
    counts = [m.trades for _, m in rows]

    colours = [p.series_1 if v >= 0 else p.critical for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        text=[f"{v:+.3f} R  ({n} trades)" for v, n in zip(vals, counts)],
        textposition="outside",
        textfont=dict(color=p.ink_secondary, size=11),
        hovertemplate="%{y}<br>Expectancy %{x:+.3f} R<extra></extra>",
        showlegend=False,
    ))
    fig.add_vline(x=0, line=dict(color=p.axis, width=1))

    layout = layout_defaults(p, height=max(240, 52 * len(rows) + 90))
    layout["hovermode"] = "closest"
    fig.update_layout(**layout,
                      title=dict(text="Expectancy per strategy (R, net of friction)",
                                 font=dict(color=p.ink, size=14)))
    fig.update_xaxes(title_text="Expectancy (R)")
    fig.update_yaxes(title_text=None)
    return fig


def build_overlays(bars: pd.DataFrame, cfg):
    """Recompute the structures the strategies saw, for drawing."""
    levels, trendlines, vwap_series = [], [], None
    try:
        levels = sig.build_levels(
            bars, lookback=cfg.signal.pivot_lookback,
            cluster_atr_frac=cfg.signal.level_cluster_atr_frac,
            min_touches=cfg.signal.min_level_touches)
        levels = sorted(levels, key=lambda lv: lv.touches, reverse=True)
    except Exception:
        pass
    try:
        vwap_series = sig.vwap(bars)
    except Exception:
        pass
    for kind in ("rising_support", "falling_resistance"):
        try:
            tl = sig.fit_trendline(
                bars, lookback=cfg.signal.pivot_lookback, kind=kind,
                min_points=cfg.strategy.trendline_min_points)
            if tl and tl.r_squared >= cfg.strategy.trendline_min_r2:
                trendlines.append(tl)
        except Exception:
            pass
    return levels, trendlines, vwap_series
