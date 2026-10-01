"""AeroMill-Adaptive AI dashboard: a thin view over aeromill.session (no business logic here)."""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from aeromill.session import RunSession, compare, list_modes, list_scenarios

TICKS_PER_REFRESH = 10  # 1 s of simulation per 1 s refresh = real time; one RF tick is ~50 ms, so a frame always finishes
SURFACE, INK, INK_2, INK_3, GRID = "#191b1d", "#e8e6e1", "#b5b2aa", "#7d7b75", "#26292c"
AXES = {
    "X": "#4f86c6",
    "Y": "#c0703f",
    "Z": "#3a9a78",
}  # muted steel/copper/sage; dataviz validator: all checks pass (dark, all pairs)
STATUS = {
    "good": "#5c9e68",
    "warning": "#c9a24a",
    "serious": "#c07a52",
    "critical": "#b95454",
}  # never color alone: always with a label
LOOP = [
    ("WARMUP", "Calibrate"),
    ("MONITOR", "Observe"),
    ("PLAN", "Plan"),
    ("APPLY", "Act"),
    ("SETTLE", "Settle"),
    ("VERIFY", "Verify"),
]
STATE_STATUS = {
    "WARMUP": ("warning", "Calibrating"),
    "MONITOR": ("good", "Monitoring"),
    "PLAN": ("serious", "Planning"),
    "APPLY": ("serious", "Applying"),
    "SETTLE": ("serious", "Settling"),
    "VERIFY": ("warning", "Verifying"),
    "HOLD": ("critical", "Stopped · HOLD"),
    "COMPLETED": ("good", "Completed"),
}
MODE_LABELS = {
    "ml_agent": "ML agent",
    "threshold_search": "Threshold search (ablation)",
    "baseline": "Threshold baseline",
    "no_adaptation": "No adaptation",
}
CHART_CFG = {"displayModeBar": False}

st.set_page_config(page_title="AeroMill-Adaptive AI", layout="wide")
st.markdown(
    f"<style>{(Path(__file__).parent / 'assets' / 'style.css').read_text()}</style>",
    unsafe_allow_html=True,
)

LOGO = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8fb3de" stroke-width="1.8" stroke-linecap="round">'
    '<circle cx="12" cy="12" r="3.2"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1 7 17M17 7l2.1-2.1"/></svg>'
)


def base_layout(
    fig: go.Figure, title: str, height: int = 230, y_title: str = "", x_title: str = ""
) -> go.Figure:
    fig.update_layout(
        title=dict(
            text=title, font=dict(size=12.5, color=INK_2), x=0.01, xanchor="left"
        ),
        height=height,
        margin=dict(l=50, r=14, t=34, b=38),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family="Segoe UI, system-ui, Arial, sans-serif", color=INK_3, size=11
        ),
        hovermode="x unified",
        showlegend=False,
        uirevision="keep",
        hoverlabel=dict(
            bgcolor="#23262a",
            bordercolor=GRID,
            font=dict(
                color=INK, family="Consolas, DejaVu Sans Mono, monospace", size=11
            ),
        ),
    )
    fig.update_xaxes(
        title=x_title,
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        title_font=dict(size=11),
    )
    fig.update_yaxes(
        title=y_title,
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        title_font=dict(size=11),
    )
    return fig


def status_html(state: str) -> str:
    role, label = STATE_STATUS.get(state, ("warning", state.title()))
    return f'<span class="am-dot" style="background:{STATUS[role]}"></span>{label}'


def tile(key: str, value: str, sub: str = "") -> str:
    return f'<div class="am-tile"><div class="am-k">{key}</div><div class="am-v">{value}</div><div class="am-s">{sub or "&nbsp;"}</div></div>'


def loop_html(state: str, changed: bool) -> str:
    """Agent loop with the current step highlighted; the entry animation plays only when the state just changed."""
    names = [name for name, _ in LOOP]
    idx = (
        names.index(state)
        if state in names
        else (len(names) if state == "COMPLETED" else -1)
    )
    extra = " halt" if state == "HOLD" else " finish" if state == "COMPLETED" else ""
    cells = []
    for i, (name, verb) in enumerate(LOOP):
        cls = "active" if i == idx else "done" if i < idx else ""
        if cls == "active" and changed:
            cls += " enter"
        cells.append(
            f'<div class="am-step {cls}"><div class="n">{i + 1:02d}</div><div class="l">{verb}</div><div class="s">{name}</div></div>'
        )
    return f'<div class="am-loop{extra}">{"".join(cells)}</div>'


def fmt_pair(pair) -> str:
    return f"{pair[0]:.0f} RPM · {pair[1]:.0f} mm/min"


EVENT_TAG = {
    "planning_record": "serious",
    "set_cutting_parameters": "serious",
    "recovery_claim": "good",
    "record_outcome": "warning",
    "hold": "critical",
    "feed_hold": "critical",
}


def describe(event: dict) -> str:
    kind = event.get("event", "?")
    if kind == "planning_record":
        if not event.get("chosen"):
            return "<b>Plan</b> no admissible candidate left"
        return f"<b>Plan</b> {len(event.get('ranked', []))} ranked, {len(event.get('excluded', []))} excluded → {fmt_pair(event['chosen'])}"
    skip = {"event", "run_id", "time_s"}
    fields = ", ".join(
        f"{k}={str(v)[:8] + '…' if k.endswith('_id') and len(str(v)) > 10 else v}"
        for k, v in event.items()
        if k not in skip and v not in ("", None) and not isinstance(v, (list, dict))
    )
    return f"<b>{kind.replace('_', ' ')}</b> {fields}"


def vibration_chart(s) -> go.Figure:
    fig = go.Figure()
    tail = np.asarray(s.vibration_tail)[
        -820:
    ]  # last 0.1 s: individual waves stay readable
    if len(tail):
        t = s.time_s - (len(tail) - 1 - np.arange(len(tail))) / 8192
        for i, (name, color) in reversed(
            list(enumerate(AXES.items()))
        ):  # X drawn last, on top
            fig.add_scatter(
                x=t,
                y=tail[:, i],
                name=f"{name} axis",
                mode="lines",
                line=dict(width=1.3, color=color),
            )
    fig = base_layout(
        fig,
        "Vibration · last 0.1 s · normalized units",
        y_title="amplitude",
        x_title="time, s",
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            y=1.16,
            x=1,
            xanchor="right",
            font=dict(color=INK_2),
            bgcolor="rgba(0,0,0,0)",
            traceorder="reversed",
        ),
    )
    return fig


def spectrum_chart(s) -> go.Figure:
    freqs, psd = (np.asarray(a) for a in s.spectrum)
    fig = go.Figure()
    if len(freqs):
        for k in range(1, 7):
            if k * s.tooth_hz < freqs[-1]:
                fig.add_vline(
                    x=k * s.tooth_hz, line=dict(color="#3a3d41", width=1, dash="dot")
                )
        fig.add_scatter(
            x=freqs,
            y=psd,
            mode="lines",
            line=dict(width=1.6, color=AXES["X"]),
            fill="tozeroy",
            fillcolor="rgba(79,134,198,0.10)",
            name="PSD, X axis",
        )
        fig.add_annotation(
            x=s.tooth_hz,
            y=1,
            yref="paper",
            text="tooth-pass harmonics",
            showarrow=False,
            font=dict(color=INK_3, size=10),
            xanchor="left",
            yanchor="bottom",
        )
    fig = base_layout(
        fig,
        "Spectrum · latest window · X axis",
        y_title="PSD (log)",
        x_title="frequency, Hz",
    )
    fig.update_yaxes(type="log")
    return fig


def score_chart(s) -> go.Figure:
    t, score = s.series["t"], s.series["score"]
    fig = go.Figure()
    fig.add_hrect(
        y0=0.8, y1=1.02, fillcolor=STATUS["critical"], opacity=0.07, line_width=0
    )
    fig.add_hrect(
        y0=-0.02, y1=0.3, fillcolor=STATUS["good"], opacity=0.05, line_width=0
    )
    fig.add_hline(
        y=0.8,
        line=dict(color=STATUS["critical"], width=1, dash="dash"),
        annotation_text="detect ≥ 0.8",
        annotation_font=dict(color=INK_2, size=10),
        annotation_position="top left",
    )
    fig.add_hline(
        y=0.3,
        line=dict(color=STATUS["good"], width=1, dash="dash"),
        annotation_text="verify ≤ 0.3",
        annotation_font=dict(color=INK_2, size=10),
        annotation_position="bottom left",
    )
    fig.add_scatter(
        x=t,
        y=score,
        mode="lines",
        line=dict(width=2, color=INK, shape="spline", smoothing=0.4),
        fill="tozeroy",
        fillcolor="rgba(232,230,225,0.06)",
        name="score",
    )
    fig = base_layout(
        fig,
        "Detector score · model output, not a failure probability",
        y_title="score",
        x_title="time, s",
    )
    fig.update_yaxes(range=[-0.02, 1.02])
    return fig


def setpoint_chart(
    s, key: str, target: float, title: str, unit: str, color: str
) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(
        y=target,
        line=dict(color=INK_3, width=1, dash="dot"),
        annotation_text="target",
        annotation_font=dict(color=INK_3, size=10),
        annotation_position="top right",
    )
    fig.add_scatter(
        x=s.series["t"],
        y=s.series[key],
        mode="lines",
        line=dict(width=2, color=color),
        name=unit,
    )
    return base_layout(fig, title, height=170, y_title=unit, x_title="time, s")


def toolpath_chart(s) -> go.Figure:
    fig = go.Figure()
    for start, end, name in s.zones:  # hatched bands: texture, not color alone
        fig.add_bar(
            x=[(start + end) / 2],
            y=[2],
            width=[end - start],
            base=[-1],
            hoverinfo="text",
            hovertext=f"{name}: {start:.0f}–{end:.0f} mm",
            marker=dict(
                color="rgba(192,122,82,0.10)",
                line=dict(width=0),
                pattern=dict(
                    shape="/", fgcolor="rgba(192,122,82,0.45)", size=7, solidity=0.15
                ),
            ),
        )
        fig.add_annotation(
            x=start,
            y=0.95,
            text=name,
            showarrow=False,
            xanchor="left",
            font=dict(color=INK_2, size=10),
        )
    fig.add_scatter(
        x=[0, s.path_length_mm],
        y=[0, 0],
        mode="lines",
        line=dict(width=4, color="#2c3035"),
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=[0, s.x_mm],
        y=[0, 0],
        mode="lines",
        line=dict(width=4, color=AXES["X"]),
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=[s.x_mm],
        y=[0],
        mode="markers",
        marker=dict(size=15, color=INK, line=dict(width=3, color=AXES["X"])),
        hovertemplate="tool at %{x:.1f} mm<extra></extra>",
    )
    fig = base_layout(fig, "Tool path · public zone map", height=132, x_title="x, mm")
    fig.update_yaxes(visible=False, range=[-1, 1])
    fig.update_xaxes(range=[0, s.path_length_mm], showgrid=False)
    fig.update_layout(
        hovermode="closest", bargap=0, margin=dict(l=50, r=14, t=30, b=44)
    )
    return fig


# ---- session state ------------------------------------------------------------------
for key, default in {
    "session": None,
    "running": False,
    "last_state": None,
    "seen_events": 0,
}.items():
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.markdown("### Run setup")
    scenarios = list_scenarios()
    scenario = st.selectbox(
        "Scenario",
        scenarios,
        index=scenarios.index("recover_a") if "recover_a" in scenarios else 0,
    )
    modes = list_modes()
    mode = st.selectbox(
        "Control mode",
        modes,
        index=modes.index("ml_agent") if "ml_agent" in modes else 0,
        format_func=lambda m: MODE_LABELS.get(m, m),
    )
    seed = st.number_input("Seed", min_value=0, value=42, step=1)
    c1, c2 = st.columns(2)
    if c1.button(
        "Start", type="primary", width="stretch", disabled=st.session_state.running
    ):
        if (
            st.session_state.session is None
            or st.session_state.session.snapshot().terminal
        ):
            st.session_state.session, st.session_state.seen_events = (
                RunSession(scenario, int(seed), mode),
                0,
            )
        st.session_state.running = True
    if c2.button("Stop", width="stretch", disabled=not st.session_state.running):
        st.session_state.session.request_stop()
    if st.button("New run", width="stretch"):
        st.session_state.session, st.session_state.seen_events = (
            RunSession(scenario, int(seed), mode),
            0,
        )
        st.session_state.running = False
    st.caption(
        "Stop is a priority feed hold: feed drops to zero and no late acknowledgment can restart motion."
    )

st.markdown(
    f'<div class="am-head"><div class="am-logo">{LOGO}</div><span class="am-title">AeroMill-Adaptive AI</span>'
    '<span class="am-sub">Autonomous chatter detection and cutting-parameter correction</span>'
    '<span class="am-badge">Synthetic simulation · not validated on real CNC</span></div>',
    unsafe_allow_html=True,
)

live_tab, compare_tab = st.tabs(["Live run", "Mode comparison"])


@st.fragment(run_every=1.0)
def live_view():
    session = st.session_state.session
    if session is None:
        st.markdown(loop_html("", False), unsafe_allow_html=True)
        st.markdown(
            '<div class="am-card am-empty">Choose a scenario and a control mode in the sidebar, then press '
            "<b>Start</b>. The agent calibrates, observes, plans, acts, and verifies on its own.</div>",
            unsafe_allow_html=True,
        )
        return
    s = (
        session.advance(TICKS_PER_REFRESH)
        if st.session_state.running
        else session.snapshot()
    )
    if s.terminal:
        st.session_state.running = False
    changed = s.state != st.session_state.last_state
    st.session_state.last_state = s.state
    fresh_from = st.session_state.seen_events
    st.session_state.seen_events = len(s.events)

    st.markdown(loop_html(s.state, changed), unsafe_allow_html=True)
    score = s.series["score"]
    last_score = (
        next((v for v in score[::-1] if np.isfinite(v)), float("nan"))
        if len(score)
        else float("nan")
    )
    commands = sum(1 for e in s.events if e.get("event") == "set_cutting_parameters")
    st.markdown(
        '<div class="am-tiles">'
        + "".join(
            [
                tile("Agent state", status_html(s.state), s.reason),
                tile(
                    "Detector score",
                    "—" if not np.isfinite(last_score) else f"{last_score:.2f}",
                    "detect ≥ 0.8 · verify ≤ 0.3",
                ),
                tile(
                    "Spindle speed", f"{s.rpm:.0f}", f"RPM · target {s.target_rpm:.0f}"
                ),
                tile(
                    "Feed",
                    f"{s.feed_mm_min:.0f}",
                    f"mm/min · target {s.target_feed_mm_min:.0f}",
                ),
                tile(
                    "Tool position",
                    f"{s.x_mm:.0f}",
                    f"mm of {s.path_length_mm:.0f} · t = {s.time_s:.1f} s",
                ),
                tile("Commands sent", f"{commands}", MODE_LABELS.get(mode, mode)),
            ]
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(toolpath_chart(s), width="stretch", config=CHART_CFG)

    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.plotly_chart(vibration_chart(s), width="stretch", config=CHART_CFG)
        st.plotly_chart(score_chart(s), width="stretch", config=CHART_CFG)
        a, b = st.columns(2)
        a.plotly_chart(
            setpoint_chart(s, "rpm", s.target_rpm, "Spindle speed", "RPM", AXES["X"]),
            width="stretch",
            config=CHART_CFG,
        )
        b.plotly_chart(
            setpoint_chart(
                s, "feed", s.target_feed_mm_min, "Feed", "mm/min", AXES["Z"]
            ),
            width="stretch",
            config=CHART_CFG,
        )
    with right:
        st.plotly_chart(spectrum_chart(s), width="stretch", config=CHART_CFG)
        plans = [e for e in s.events if e.get("event") == "planning_record"]
        st.markdown(
            '<div class="am-h">Agent plan <span>· latest decision</span></div>',
            unsafe_allow_html=True,
        )
        if plans:
            p = plans[-1]
            rows = "".join(
                f'<tr class="{"chosen" if c["pair"] == p.get("chosen") else ""}"><td>{fmt_pair(c["pair"])}</td>'
                f"<td>{'chosen' if c['pair'] == p.get('chosen') else f'ranked #{i}'}</td></tr>"
                for i, c in enumerate(p.get("ranked", []), 1)
            )
            rows += "".join(
                f"<tr><td>{fmt_pair(c['pair'])}</td><td>excluded · {c.get('reason', '')}</td></tr>"
                for c in p.get("excluded", [])
            )
            st.markdown(
                f'<div class="am-card"><table class="am-plan"><tr><th>candidate</th><th>decision</th></tr>{rows}</table></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="am-card am-empty">No incident yet. The agent is observing.</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="am-h">Decision log <span>· observation → reason → command → outcome</span></div>',
            unsafe_allow_html=True,
        )
        shown = [(i, e) for i, e in enumerate(s.events) if e.get("event") != "tick"][
            -14:
        ][::-1]  # ticks are telemetry, not decisions
        lines = "".join(
            f'<div class="am-ev{" new" if i >= fresh_from else ""}"><span class="tag" style="background:{STATUS.get(EVENT_TAG.get(e.get("event")), GRID)}"></span>'
            f'<span class="t">{e.get("time_s", 0):.1f}s</span><span>{describe(e)}</span></div>'
            for i, e in shown
        )
        st.markdown(
            f'<div class="am-card" style="max-height:330px;overflow:auto">{lines or "<span class=am-empty>No events yet.</span>"}</div>',
            unsafe_allow_html=True,
        )

    if s.terminal and s.evaluation:
        ev = s.evaluation
        ok = bool(ev.get("run_success"))
        role = "good" if ok else "critical"
        verdict = "Run succeeded" if ok else "Run did not succeed"
        why = (
            f"{ev.get('confirmed_recoveries', 0)} of {ev.get('incidents', 0)} incident(s) confirmed recovered, "
            f"{ev.get('missed_incidents', 0)} missed, {ev.get('false_recoveries', 0)} false claims"
        )
        st.markdown(
            f'<div class="am-h">Result <span>· independent evaluator on hidden ground truth, never seen by the agent</span></div>'
            f'<div class="am-verdict {role}{" enter" if changed else ""}"><span class="am-dot" style="background:{STATUS[role]}"></span>'
            f"<b>{verdict}</b><span>{why}</span></div>",
            unsafe_allow_html=True,
        )
        latency = [
            d.get("detection_latency_s")
            for d in ev.get("incident_details", [])
            if d.get("detection_latency_s") is not None
        ]
        rms = ev.get("rms_reduction")
        st.markdown(
            '<div class="am-tiles">'
            + "".join(
                [
                    tile(
                        "Detection latency",
                        f"{max(latency):.2f} s" if latency else "—",
                        "from chatter onset · target ≤ 1 s",
                    ),
                    tile(
                        "Time in chatter",
                        f"{ev.get('unstable_time_s', 0):.2f} s",
                        "hidden envelope A ≥ 0.75",
                    ),
                    tile(
                        "Vibration reduction",
                        f"{rms * 100:.0f}%" if isinstance(rms, (int, float)) else "—",
                        "RMS, before → after command",
                    ),
                    tile(
                        "Traversal time",
                        f"{ev.get('traversal_time_s', 0):.1f} s",
                        f"{s.path_length_mm:.0f} mm path",
                    ),
                    tile(
                        "False interventions",
                        f"{ev.get('false_interventions', 0)}",
                        "commands without real chatter",
                    ),
                    tile("Stops", f"{ev.get('stops', 0)}", ev.get("state", "")),
                ]
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        with st.expander("All evaluator metrics"):
            keys = [k for k, v in ev.items() if not isinstance(v, (list, dict))]
            st.dataframe(
                {"metric": keys, "value": [str(ev[k]) for k in keys]},
                hide_index=True,
                width="stretch",
            )
        csv_bytes, json_bytes = session.export()
        d1, d2, _ = st.columns([1, 1, 4])
        d1.download_button(
            "Export CSV", csv_bytes, file_name="aeromill_run.csv", mime="text/csv"
        )
        d2.download_button(
            "Export JSON",
            json_bytes,
            file_name="aeromill_run.json",
            mime="application/json",
        )


with live_tab:
    live_view()

with compare_tab:
    st.markdown(
        '<div class="am-h">Mode comparison <span>· same scenario, seed and disturbances; metrics from the independent evaluator</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The threshold baseline stays inside the resonance zone by construction. The ML contribution is judged against threshold search, which uses the same candidates."
    )
    if st.button("Run comparison", type="primary"):
        with st.spinner("Running all modes…"):
            st.session_state.comparison = compare(scenario, int(seed))
    rows = st.session_state.get("comparison")
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
        st.download_button(
            "Export comparison JSON",
            json.dumps(rows, default=str).encode(),
            file_name="aeromill_comparison.json",
            mime="application/json",
        )
