"""AeroMill-Adaptive AI dashboard: a thin view over aeromill.session (no business logic here)."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from aeromill.parts import DEMO_SEED_MIN, Part, random_part
from aeromill.scenarios import EngagementSegment, Region, Scenario
from aeromill.session import RunSession, compare, list_modes, list_scenarios
from aeromill.viewer import catalog, stability_map, trajectory

TICKS_PER_REFRESH = (
    10  # 1 s of sim per 1 s refresh at 1x; a tick is ~13 ms now, so 4x still fits
)
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
SAFE_STOP = {
    "ATTEMPTS_EXHAUSTED": "Every allowed spindle speed still chattered on this part, so the agent stopped instead of cutting a bad surface.",
    "NO_CANDIDATE": "No allowed spindle speed was left to try, so the agent stopped instead of cutting a bad surface.",
    "DATA_QUALITY": "The sensor data became invalid, so the agent refused to act on it and stopped.",
    "CALIBRATION_FAILED": "The cut started already unstable, so calibration was impossible and the agent stopped.",
    "COMMAND_REJECTED": "The controller rejected a command, so the agent did not assume it worked and stopped.",
    "ACK_TIMEOUT": "The controller never confirmed a command, so the agent did not assume it worked and stopped.",
    "MANUAL_STOP": "The operator pressed Stop; feed went to zero and nothing could restart it.",
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


def status_html_text(role: str, label: str) -> str:
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
        if k not in skip and v not in ("", None) and not isinstance(v, (list, tuple, dict))
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
    fig = base_layout(fig, title, height=170, y_title=unit, x_title="time, s")
    values = np.append(np.asarray(s.series[key], dtype=float), target)
    values = values[np.isfinite(values)]
    if len(values) and np.ptp(values) < 20:
        # a flat line otherwise gets an absurd axis like 3200.5 / 3201
        mid = float(values.mean())
        fig.update_yaxes(range=[mid - 60, mid + 60])
    return fig


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


def map_chart(m, traj, feed) -> go.Figure:
    fig = go.Figure()
    # one hue, dark to light copper: more copper = stronger chatter
    scale = [[0, SURFACE], [0.18, "#3a2a20"], [0.6, "#8a5230"], [1, "#d99a6c"]]
    fig.add_heatmap(
        x=m["x_mm"],
        y=m["rpm"],
        z=m["A_target"],
        zmin=0,
        zmax=4.1,
        colorscale=scale,
        colorbar=dict(
            title=dict(text="A", font=dict(color=INK_3, size=10)),
            thickness=8,
            len=0.8,
            x=1.07,
            tickfont=dict(color=INK_3, size=9),
            outlinewidth=0,
        ),
        hovertemplate="x %{x:.0f} mm · %{y:.0f} RPM<br>chatter envelope ≈ %{z:.2f}<extra></extra>",
    )
    fig.add_contour(
        x=m["x_mm"],
        y=m["rpm"],
        z=m["A_target"],
        showscale=False,
        hoverinfo="skip",
        contours=dict(start=0.75, end=0.75, size=1, coloring="none"),
        line=dict(color=INK_2, width=1, dash="dash"),
    )
    for label, rpm in CANDIDATES.items():
        fig.add_hline(
            y=rpm,
            line=dict(color="#4a4d52", width=1, dash="dot"),
            annotation_text=label,
            annotation_position="right",
            annotation_font=dict(color=INK_3, size=10),
        )
    if len(traj["x_mm"]):
        fig.add_scatter(
            x=traj["x_mm"],
            y=traj["rpm"],
            mode="lines",
            line=dict(width=2.5, color=AXES["X"]),
            name="agent path",
            hovertemplate="agent at %{x:.0f} mm, %{y:.0f} RPM<extra></extra>",
        )
        fig.add_scatter(
            x=traj["x_mm"][-1:],
            y=traj["rpm"][-1:],
            mode="markers",
            hoverinfo="skip",
            marker=dict(size=11, color=INK, line=dict(width=2, color=AXES["X"])),
        )
    fig = base_layout(
        fig,
        f"Hidden stability map · copper = chatter at {feed:.0f} mm/min · dashed = chatter boundary · blue = agent path",
        height=300,
        y_title="spindle speed, RPM",
        x_title="x, mm",
    )
    fig.update_layout(hovermode="closest", margin=dict(l=60, r=70, t=34, b=40))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def envelope_chart(r, without_agent=None, events=()) -> go.Figure:
    def thin(res):
        step = max(1, len(res["envelope"]) // 1500)
        return res["time_s"][::step], res["envelope"][::step]

    t, a = thin(r)
    top = max(4.2, float(a.max()) + 0.3)
    fig = go.Figure()
    fig.add_hrect(y0=0.75, y1=top, fillcolor=STATUS["critical"], opacity=0.06, line_width=0)
    fig.add_hline(y=0.75, line=dict(color=STATUS["critical"], width=1, dash="dash"),
                  annotation_text="chatter zone", annotation_font=dict(color=INK_3, size=10),
                  annotation_position="top right")

    if without_agent is not None:
        tw, aw = thin(without_agent)
        fig.add_scatter(x=tw, y=aw, mode="lines", name="same part, no agent",
                        line=dict(width=1.6, color=INK_3, dash="dot"))
    fig.add_scatter(x=t, y=a, mode="lines", name="with the agent", line=dict(width=2.2, color=AXES["Y"]),
                    fill="tozeroy", fillcolor="rgba(192,112,63,0.10)")

    # mark when chatter began and when the agent acted, so the spike tells a story
    details = r["evaluation"].get("incident_details", []) if r.get("evaluation") else []
    commands = [e["time_s"] for e in events if e.get("event") == "set_cutting_parameters"]
    for i, d in enumerate(details):
        fig.add_vline(x=d["start_s"], line=dict(color=STATUS["critical"], width=1, dash="dash"))
        if i == 0:
            fig.add_annotation(x=d["start_s"], y=top * 0.92, text="chatter starts", showarrow=False,
                               xanchor="right", xshift=-4, font=dict(color=INK_2, size=10))
    for i, c in enumerate(commands):
        fig.add_vline(x=c, line=dict(color=AXES["X"], width=1.2))
        if i == 0:
            fig.add_annotation(x=c, y=top * 0.78, text="agent changes speed", showarrow=False,
                               xanchor="left", xshift=4, font=dict(color=INK_2, size=10))

    fig = base_layout(fig, "Hidden chatter intensity A(t) · agent vs no agent",
                      height=300, y_title="chatter intensity A", x_title="time, s")
    fig.update_layout(showlegend=True, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.28, x=0, font=dict(color=INK_2), bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=50, r=14, t=34, b=64))
    fig.update_yaxes(range=[0, top])
    fig.update_xaxes(rangemode="nonnegative")
    return fig


def replay_without_agent(part):
    # same hidden part, no adaptation: the honest "before" for the reveal chart
    other = RunSession.from_part(part, "no_adaptation")
    while not other.snapshot().terminal:
        other.advance(200)
    return other.reveal()


@st.cache_resource
def background_pool():
    return ThreadPoolExecutor(max_workers=1)


def describe_part(r) -> str:
    bits = []
    for z in r["zones"]:
        peaks = ", ".join(f"{c:.0f}±{w:.0f}" for c, w in zip(z.centers, z.widths))
        bits.append(f"resonance at {peaks} RPM over {z.start_mm:.0f}–{z.end_mm:.0f} mm")
    for e in r["engagement"]:
        bits.append(
            f"engagement ×{e.level:.2f} over {e.start_mm:.0f}–{e.end_mm:.0f} mm"
        )
    return "; ".join(bits) or "no resonance zone at all"


CATALOG = {c["name"]: c for c in catalog()}
CANDIDATES = {"start": 3200, "A": 3520, "B": 2880, "C": 3840}

for key, default in {
    "session": None,
    "running": False,
    "last_state": None,
    "seen_events": 0,
    "preset": "recover_a",
    "source": "Preset",
    "unknown_part": None,
}.items():
    st.session_state.setdefault(key, default)


def reset_run():
    st.session_state.session = None
    st.session_state.running = False
    st.session_state.seen_events = 0


def load_preset(name):
    # runs as a button callback, i.e. before the sidebar widgets get rebuilt
    st.session_state.preset = name
    st.session_state.source = "Preset"
    reset_run()


def custom_part_form():
    center = st.slider("Resonance centre, RPM", 2880, 3840, 3200, 40)
    width = st.slider("Resonance width, RPM", 100, 180, 120, 5)
    start = st.slider("Zone starts at, mm", 60, 100, 80, 5)
    centers, widths = [float(center)], [float(width)]
    if st.checkbox("Second resonance in the same zone"):
        centers.append(float(st.slider("Second centre, RPM", 2880, 3840, 3520, 40)))
        widths.append(float(width))

    segments = ()
    if st.checkbox("Engagement rise (louder, not chatter)"):
        e_start = st.slider("Rise starts at, mm", 60, 280, 150, 10)
        e_len = st.slider("Rise length, mm", 40, 120, 100, 10)
        level = st.slider("Rise level, × nominal", 1.3, 2.0, 2.0, 0.1)
        segments = (
            EngagementSegment(float(e_start), float(e_start + e_len), float(level)),
        )

    chatter = st.slider("Chatter frequency, Hz", 900, 1800, 1200, 50)
    noise = st.slider("Sensor noise", 0.02, 0.10, 0.05, 0.01)

    scenario = Scenario(
        "custom",
        (Region(float(start), 400.0, tuple(centers), tuple(widths)),),
        engagement=segments,
    )
    return Part(
        DEMO_SEED_MIN, scenario, noise_std=float(noise), chatter_hz=float(chatter)
    )


def make_session(source, preset, seed, mode, part):
    st.session_state.run_meta = {"source": source, "preset": preset, "part": part}
    st.session_state.no_agent = None
    if source == "Preset":
        return RunSession(preset, int(seed), mode)
    return RunSession.from_part(part, mode)


with st.sidebar:
    st.markdown("### Run setup")
    source = st.radio(
        "Part",
        ["Preset", "Unknown part", "Custom part"],
        key="source",
        on_change=reset_run,
    )
    part, seed = None, 42

    if source == "Preset":
        preset = st.selectbox(
            "Scenario",
            list_scenarios(),
            key="preset",
            format_func=lambda n: CATALOG.get(n, {}).get("title", n),
            on_change=reset_run,
        )
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
    elif source == "Unknown part":
        preset = None
        if (
            st.button("Draw a new part", width="stretch")
            or st.session_state.unknown_part is None
        ):
            st.session_state.unknown_part = random_part()
            reset_run()
        part = st.session_state.unknown_part
        st.caption(
            f"Part #{part.seed}. Nobody picked it. Its zones, engagement and chatter "
            "frequency stay hidden from you and the agent until the run ends."
        )
    else:
        preset = None
        with st.expander("Part parameters", expanded=True):
            try:
                part = custom_part_form()
            except ValueError as err:
                st.error(f"This part is outside what the model was trained on: {err}")

    modes = list_modes()
    mode = st.selectbox(
        "Control mode",
        modes,
        index=modes.index("ml_agent") if "ml_agent" in modes else 0,
        format_func=lambda m: MODE_LABELS.get(m, m),
    )
    speed = st.select_slider(
        "Playback speed", options=[1, 2, 4], value=1, format_func=lambda v: f"{v}×"
    )

    can_start = source == "Preset" or part is not None
    c1, c2 = st.columns(2)
    if c1.button(
        "Start",
        type="primary",
        width="stretch",
        disabled=st.session_state.running or not can_start,
    ):
        if (
            st.session_state.session is None
            or st.session_state.session.snapshot().terminal
        ):
            st.session_state.session = make_session(source, preset, seed, mode, part)
            st.session_state.seen_events = 0
        st.session_state.running = True
    if c2.button("Stop", width="stretch", disabled=not st.session_state.running):
        st.session_state.session.request_stop()
    if st.button("New run", width="stretch", disabled=not can_start):
        reset_run()
        st.session_state.session = make_session(source, preset, seed, mode, part)

    st.caption(
        "Stop is a priority feed hold: feed drops to zero and no late acknowledgment can restart motion."
    )

st.markdown(
    f'<div class="am-head"><div class="am-logo">{LOGO}</div><span class="am-title">AeroMill-Adaptive AI</span>'
    '<span class="am-sub">Autonomous chatter detection and cutting-parameter correction</span>'
    '<span class="am-badge">Synthetic simulation · not validated on real CNC</span></div>',
    unsafe_allow_html=True,
)

live_tab, catalog_tab, compare_tab = st.tabs(
    ["Live run", "Scenario catalog", "Mode comparison"]
)


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
        session.advance(TICKS_PER_REFRESH * speed)
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
    meta = st.session_state.get("run_meta", {})
    if meta.get("source") == "Preset" and meta.get("preset") in CATALOG:
        c = CATALOG[meta["preset"]]
        st.markdown(
            f'<div class="am-brief"><b>{c["title"]}</b><span>Viewer briefing, hidden from the agent: {c["hides"]} '
            f"It should: {c['expect']}</span></div>",
            unsafe_allow_html=True,
        )
    elif meta.get("source"):
        st.markdown(
            '<div class="am-brief"><b>Unknown part</b><span>Nobody here knows what is inside it, including us. '
            "The truth is revealed when the run ends.</span></div>",
            unsafe_allow_html=True,
        )
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

    # the map always sits right under the tool path: same x axis, easy to compare
    feed = next((f for f in s.series["feed"][::-1] if f > 0), 1200.0)
    rpm_grid = np.linspace(2400, 4200, 91)
    x_grid = np.linspace(0, s.path_length_mm, 121)

    if meta.get("source") == "Preset":
        if st.toggle(
            "Show the hidden stability map (viewer only, the agent never sees it)",
            key="show_map",
        ):
            m = stability_map(meta["preset"], feed, rpm_grid, x_grid)
            st.plotly_chart(map_chart(m, trajectory(s), feed), width="stretch", config=CHART_CFG)

    elif meta.get("source") and s.terminal:
        r = session.reveal()
        st.markdown(
            f'<div class="am-h">Reveal <span>· the hidden part: {describe_part(r)}</span></div>',
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns([3, 2], gap="medium")
        m1.plotly_chart(
            map_chart(session.stability_map(feed, rpm_grid, x_grid), trajectory(s), feed),
            width="stretch",
            config=CHART_CFG,
        )
        # the no-agent replay starts only after the run: during it, it would steal the GIL
        job = st.session_state.get("no_agent")
        if job is None:
            job = st.session_state.no_agent = background_pool().submit(replay_without_agent, meta["part"])
        baseline = job.result() if job.done() else None
        if baseline is None:
            m2.caption("Replaying the same part without the agent for comparison…")
        m2.plotly_chart(envelope_chart(r, baseline, s.events), width="stretch", config=CHART_CFG)
        m2.caption("The spike is the chatter episode. Its width is how long the agent needed to get out of it. "
                   "The dotted line is the same part with nobody in control.")

    elif meta.get("source"):
        st.markdown(
            '<div class="am-card am-locked">Hidden stability map · locked until the run ends. '
            "Right now you see exactly what the agent sees.</div>",
            unsafe_allow_html=True,
        )

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
        why = (
            f"{ev.get('confirmed_recoveries', 0)} of {ev.get('incidents', 0)} incident(s) confirmed recovered, "
            f"{ev.get('missed_incidents', 0)} missed, {ev.get('false_recoveries', 0)} false claims"
        )
        # a deliberate HOLD with nothing missed or faked is the right answer, not a failure
        # starting in chatter is the spec's bounded calibration failure, so it counts as an honest stop too
        honest_stop = s.reason == "CALIBRATION_FAILED" or (
            s.state == "HOLD"
            and s.reason != "RUN_TIMEOUT"
            and not ev.get("missed_incidents")
            and not ev.get("false_recoveries")
            and not ev.get("unverified_recoveries")
        )
        if ok:
            role, verdict = "good", "Run succeeded"
        elif honest_stop:
            role = "warning"
            verdict = "Stopped safely"
            why = SAFE_STOP.get(s.reason, f"The agent stopped the cut on purpose ({s.reason}).") + " " + why
        else:
            role, verdict = "critical", "Run did not succeed"
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
                        f"{ev.get('unstable_time_s') or 0:.2f} s",
                        "hidden envelope A ≥ 0.75",
                    ),
                    tile(
                        "Vibration reduction",
                        f"{rms * 100:.0f}%" if isinstance(rms, (int, float)) else "—",
                        "RMS, before → after command",
                    ),
                    tile(
                        "Traversal time",
                        "—" if ev.get("traversal_time_s") is None else f"{ev['traversal_time_s']:.1f} s",
                        f"{s.path_length_mm:.0f} mm path"
                        if ev.get("traversal_time_s") is not None
                        else "not completed: the cut was stopped",
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

with catalog_tab:
    st.markdown(
        '<div class="am-h">Scenario catalog <span>· what every preset hides and what the agent has to do about it</span></div>',
        unsafe_allow_html=True,
    )
    items = list(CATALOG.values())
    for row in range(0, len(items), 3):
        cols = st.columns(3, gap="medium")
        for col, c in zip(cols, items[row : row + 3]):
            with col:
                st.markdown(
                    f'<div class="am-cat"><div class="am-cat-t">{c["title"]}</div>'
                    f'<div class="am-cat-k">Hidden in the part</div><div>{c["hides"]}</div>'
                    f'<div class="am-cat-k">The agent should</div><div>{c["expect"]}</div>'
                    f'<div class="am-cat-k">Evaluator checks</div><div class="am-cat-s">{c["evaluator_checks"]}</div></div>',
                    unsafe_allow_html=True,
                )
                st.button(
                    "Load in live run",
                    key=f"load_{c['name']}",
                    on_click=load_preset,
                    args=(c["name"],),
                    width="stretch",
                )
    if st.session_state.source == "Preset" and st.session_state.session is None:
        st.caption(
            f"Loaded: {CATALOG.get(st.session_state.preset, {}).get('title', st.session_state.preset)}. Press Start in the sidebar."
        )

MODE_STORY = {
    "no_adaptation": ("Nobody in control", "Never changes speed or feed. This is the 'before' picture."),
    "baseline": ("Simple loudness rule", "If the vibration gets loud, slow the feed once. It keeps the same spindle speed."),
    "threshold_search": ("Loudness rule + candidates", "Same A/B/C candidates and checks as the agent, but it decides by loudness, not ML."),
    "ml_agent": ("ML agent", "Our agent: an ML detector decides, then plan, act, settle and verify."),
}
MODE_COLOR = {"no_adaptation": "#5a5d62", "baseline": "#7d7b75", "threshold_search": "#c0703f", "ml_agent": "#4f86c6"}


def outcome(row):
    """Same three verdicts as the live run, so the two screens never disagree."""
    if row.get("run_success"):
        return "good", "Succeeded"
    honest = row.get("state") == "HOLD" and not row.get("missed_incidents") and not row.get("false_recoveries")
    if honest:
        return "warning", "Stopped safely"
    if row.get("state") == "COMPLETED" and row.get("incidents") and row.get("missed_incidents"):
        return "critical", "Chatter not fixed"
    return "critical", "Did not succeed"


def latency(row):
    vals = [d.get("detection_latency_s") for d in row.get("incident_details") or [] if d.get("detection_latency_s") is not None]
    return max(vals) if vals else None


def compare_story(rows):
    by = {r["mode"]: r for r in rows}
    ml, ts, none = by.get("ml_agent"), by.get("threshold_search"), by.get("no_adaptation")
    lines = []
    if none and none.get("incidents"):
        lines.append(f"Without any control this part spends <b>{none['unstable_time_s']:.1f} s</b> in chatter.")
    elif none:
        lines.append("This part never chatters, so the best possible result is to do nothing.")
    if ml and none and none.get("incidents"):
        lines.append(f"The ML agent cuts that to <b>{ml['unstable_time_s']:.1f} s</b>.")
    if ml and ts:
        if ts.get("false_interventions", 0) > ml.get("false_interventions", 0):
            lines.append(f"The loudness rule raised <b>{ts['false_interventions']}</b> false alarm(s) here; the ML agent raised "
                         f"{ml['false_interventions']}. Loud is not the same as chatter, and that is where the ML detector pays off.")
        elif abs(ts["unstable_time_s"] - ml["unstable_time_s"]) < 0.05 and ts.get("commands") == ml.get("commands"):
            lines.append("On this part the loudness rule and the ML agent behave the same. "
                         "Try <b>Louder, still stable</b> to see where they differ.")
    return " ".join(lines)


with compare_tab:
    st.markdown(
        '<div class="am-h">Mode comparison <span>· one part, four ways to control it</span></div>'
        '<div class="am-brief"><b>What is compared</b><span>The same part, seed, noise and disturbances run four times. '
        "Only the controller changes. Every result comes from the independent evaluator, which sees the hidden truth.</span></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4, gap="small")
    for col, (mode, (name, what)) in zip(cols, MODE_STORY.items()):
        col.markdown(
            f'<div class="am-cat" style="min-height:96px"><div class="am-cat-t">'
            f'<span class="am-dot" style="background:{MODE_COLOR[mode]}"></span>{name}</div>'
            f'<div class="am-cat-s">{what}</div></div>',
            unsafe_allow_html=True,
        )

    if source != "Preset":
        st.info("Comparison runs on presets. Pick a preset in the sidebar.")
    elif st.button("Run comparison", type="primary"):
        with st.spinner("Running all four modes on the same part…"):
            st.session_state.comparison = compare(preset, int(seed))
            st.session_state.comparison_of = CATALOG.get(preset, {}).get("title", preset)

    rows = st.session_state.get("comparison")
    if rows:
        st.markdown(
            f'<div class="am-h">Result <span>· {st.session_state.get("comparison_of", "")}</span></div>'
            f'<div class="am-brief"><b>What this shows</b><span>{compare_story(rows)}</span></div>',
            unsafe_allow_html=True,
        )

        cards = []
        for r in rows:
            role, verdict = outcome(r)
            name = MODE_STORY.get(r["mode"], (r["mode"], ""))[0]
            fixed = f"{r.get('confirmed_recoveries', 0)} of {r.get('incidents', 0)} fixed" if r.get("incidents") else "no chatter in this part"
            cut = "—" if r.get("traversal_time_s") is None else f"{r['traversal_time_s']:.1f} s"
            cards.append(
                f'<div class="am-tile"><div class="am-k"><span class="am-dot" style="background:{MODE_COLOR.get(r["mode"], GRID)}"></span>{name}</div>'
                f'<div class="am-v" style="font-size:1.05rem">{status_html_text(role, verdict)}</div>'
                f'<div class="am-s">{fixed} · {r.get("commands", 0)} command(s)</div>'
                f'<div class="am-s">in chatter {r.get("unstable_time_s", 0):.1f} s · cut {cut}</div></div>'
            )
        st.markdown(f'<div class="am-tiles" style="grid-template-columns:repeat(4,minmax(0,1fr))">{"".join(cards)}</div>',
                    unsafe_allow_html=True)

        # a part that never chatters makes every chatter bar zero, so ask the question that matters there
        chatters = any(r.get("unstable_time_s", 0) > 0.05 for r in rows)
        if chatters:
            values = [r.get("unstable_time_s", 0) for r in rows]
            labels = [f"{v:.1f} s" for v in values]
            title, axis, hover = "Time spent in chatter · shorter is better", "seconds of chatter (hidden truth)", "%{y}: %{x:.2f} s in chatter<extra></extra>"
        else:
            values = [r.get("false_interventions", 0) for r in rows]
            labels = [str(int(v)) for v in values]
            title = "False alarms · fewer is better (this part never chatters)"
            axis, hover = "commands sent without real chatter", "%{y}: %{x} false alarm(s)<extra></extra>"

        fig = go.Figure()
        names = [MODE_STORY.get(r["mode"], (r["mode"],))[0] for r in rows]
        fig.add_bar(
            y=names, x=values, orientation="h",
            marker=dict(color=[MODE_COLOR.get(r["mode"], GRID) for r in rows], line=dict(width=0), cornerradius=4),
            text=labels, textposition="outside", cliponaxis=False,
            textfont=dict(color=INK_2, size=11), hovertemplate=hover,
        )
        fig = base_layout(fig, title, height=230, x_title=axis)
        fig.update_layout(hovermode="closest", bargap=0.35, margin=dict(l=170, r=40, t=34, b=40))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        longest = max(values + [1.0])
        fig.update_xaxes(range=[0, longest * 1.2])  # room for the value labels at the bar ends
        if not chatters:
            fig.update_xaxes(dtick=1)
        if not chatters and not any(values):
            # nothing to compare: a row of zero-length bars says less than one plain sentence
            st.markdown(
                f'<div class="am-verdict good"><span class="am-dot" style="background:{STATUS["good"]}"></span>'
                "<b>No false alarms</b><span>This part never chatters, and all four modes left it alone: "
                "0 commands sent for nothing.</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.plotly_chart(fig, width="stretch", config=CHART_CFG)

        table = []
        for r in rows:
            lat = latency(r)
            table.append({
                "Mode": MODE_STORY.get(r["mode"], (r["mode"],))[0],
                "Outcome": outcome(r)[1],
                "Episodes": r.get("incidents", 0),
                "Fixed": r.get("confirmed_recoveries", 0),
                "Missed": r.get("missed_incidents", 0),
                "In chatter, s": f"{r.get('unstable_time_s', 0):.2f}",
                "Detected, s": "—" if lat is None else f"{lat:.2f}",
                "Commands": r.get("commands", 0),
                "False alarms": r.get("false_interventions", 0),
                "Cut, s": "—" if r.get("traversal_time_s") is None else f"{r['traversal_time_s']:.1f}",
                "Extra time": "—" if r.get("productivity_loss") is None else f"{r['productivity_loss'] * 100:.0f}%",
            })
        st.dataframe(table, hide_index=True, width="stretch")

        with st.expander("All evaluator metrics"):
            flat = [{k: v for k, v in r.items() if not isinstance(v, (list, tuple, dict))} for r in rows]
            st.dataframe(flat, hide_index=True, width="stretch")
        st.download_button(
            "Export comparison JSON",
            json.dumps(rows, default=str).encode(),
            file_name="aeromill_comparison.json",
            mime="application/json",
        )


def install_guide():
    # The tour runs in the page itself, not in the component's iframe, so it can draw over
    # the dashboard and survive reruns. It never touches Streamlit state.
    assets = Path(__file__).parent / "assets"
    css = json.dumps((assets / "guide.css").read_text())
    js = json.dumps((assets / "guide.js").read_text())
    components.html(
        f"""<script>
        const doc = window.parent.document;
        if (!doc.getElementById("am-guide-style")) {{
          const style = doc.createElement("style");
          style.id = "am-guide-style";
          style.textContent = {css};
          doc.head.appendChild(style);
        }}
        if (!doc.getElementById("am-guide-script")) {{
          const script = doc.createElement("script");
          script.id = "am-guide-script";
          script.textContent = {js};
          doc.head.appendChild(script);
        }}
        </script>""",
        height=0,
    )


install_guide()
