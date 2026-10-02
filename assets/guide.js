// Guided tour for the dashboard. It only draws on top of the page: no Streamlit state,
// no reruns, nothing in the app is changed. Closing it removes every node it added.
(function () {
  if (window.__amGuide) return;

  const NS = "http://www.w3.org/2000/svg";
  const reduce = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const EASE_OUT = "cubic-bezier(0.16, 1, 0.3, 1)";
  const EASE_IN = "cubic-bezier(0.7, 0, 0.84, 0)";
  const COLORS = { x: "#4f86c6", y: "#c0703f", z: "#3a9a78", good: "#5c9e68", warn: "#c9a24a", bad: "#b95454", ink: "#e8e6e1", mute: "#7d7b75" };

  // ---- finding things on the page --------------------------------------------------

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const box = (el) => el && el.closest('[data-testid="stElementContainer"]') || el;

  const visible = (el) => el && el.isConnected && el.getClientRects().length > 0 && el.getBoundingClientRect().height > 4;

  // both tabs stay in the DOM, so always prefer the copy that is actually on screen
  function chart(prefix) {
    const charts = $$(".gtitle")
      .filter((t) => t.textContent.trim().startsWith(prefix))
      .map((t) => t.closest('[data-testid="stPlotlyChart"]'));
    return charts.find(visible) || null;
  }

  // live-run steps only look inside the Live run tab, so they never pick up a look-alike block
  // from another tab (the comparison tab has its own briefs and tiles)
  function liveOnly(sel) {
    const panel = $$('[role="tabpanel"]').find((p) => $(".am-loop", p));
    return panel ? $$(sel, panel).find(visible) || null : null;
  }

  // and the comparison steps only look inside the Mode comparison tab
  function comparePanel() {
    return $$('[role="tabpanel"]').find((p) => $$(".am-h", p).some((h) => h.textContent.trim().startsWith("Mode comparison")));
  }

  function inCompare(el) {
    const panel = comparePanel();
    return panel && el && panel.contains(el) ? el : null;
  }

  function headerAndNext(text) {
    const h = $$(".am-h").filter((el) => el.textContent.trim().startsWith(text)).find(visible);
    if (!h) return null;
    const head = box(h);
    return [head, head.nextElementSibling].filter(Boolean);
  }

  // ---- what the tour says ---------------------------------------------------------

  const chip = (color, kind = "") => `<span class="amg-chip ${kind}" style="background:${color};color:${color}"></span>`;

  const STEPS = [
    {
      kicker: "Welcome",
      title: "AeroMill-Adaptive AI in two minutes",
      body: `<svg class="amg-logo" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3.2"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1 7 17M17 7l2.1-2.1"/></svg>
        <p>When a milling cutter hits a resonance, it starts to <b>chatter</b>: a self-excited vibration that ruins the surface and can break the tool.</p>
        <p>This agent listens to the vibration, decides when chatter starts, changes spindle speed and feed, and checks that the fix really worked.</p>
        <p class="amg-why">This tour shows what every block does, why it is there and how to read it.</p>`,
      noRunHint: `<p class="amg-why"><b>Tip:</b> start a run first (sidebar → Start). The tour then explains live data.</p>`,
    },
    {
      target: () => $('section[data-testid="stSidebar"]'),
      kicker: "Controls",
      title: "Run setup",
      body: `<p>Pick a <b>part</b>, a <b>control mode</b> and the playback speed, then press Start.</p>
        <ul class="amg-read">
          <li>${chip(COLORS.x)}<span><b>Preset</b>: a known scenario, explained in the Scenario catalog.</span></li>
          <li>${chip(COLORS.y)}<span><b>Unknown part</b>: drawn at random, hidden from you and the agent. It proves the runs are not scripted.</span></li>
          <li>${chip(COLORS.z)}<span><b>Custom part</b>: build your own part with sliders.</span></li>
        </ul>
        <p class="amg-why"><b>Stop</b> is a priority feed hold: feed drops to zero and nothing can restart it.</p>`,
    },
    {
      target: () => liveOnly(".am-loop"),
      kicker: "How the agent thinks",
      title: "The agent loop",
      body: `<p>Every run goes through the same six steps: <b>calibrate</b> on a quiet cut, <b>observe</b>, <b>plan</b> a fix, <b>act</b>, let the machine <b>settle</b>, and <b>verify</b> the result.</p>
        <ul class="amg-read">
          <li>${chip(COLORS.x)}<span>Blue border: the step the agent is in right now.</span></li>
          <li>${chip(COLORS.good)}<span>Tick: steps already passed.</span></li>
          <li>${chip(COLORS.bad)}<span>Red: the agent stopped on purpose (HOLD).</span></li>
        </ul>
        <p class="amg-why"><b>Why it matters:</b> this closed loop (see, decide, act, check) is what makes it an agent, not a dashboard.</p>`,
    },
    {
      target: () => liveOnly(".am-brief"),
      kicker: "For the viewer",
      title: "Briefing",
      body: `<p>One line on what this part hides and what the agent should do about it.</p>
        <p class="amg-why">The agent never sees this text. For an Unknown part nobody knows what is inside until the run ends.</p>`,
    },
    {
      target: () => liveOnly(".am-tiles"),
      kicker: "Live readouts",
      title: "What is happening now",
      body: `<ul class="amg-read">
          <li>${chip(COLORS.good)}<span><b>Agent state</b>: the current loop step, in words.</span></li>
          <li>${chip(COLORS.ink)}<span><b>Detector score</b>: 0 means calm, 1 means chatter.</span></li>
          <li>${chip(COLORS.x)}<span><b>Spindle speed</b> and <b>feed</b>: the actual values and the target the agent asked for.</span></li>
          <li>${chip(COLORS.mute)}<span><b>Tool position</b> and how many commands were sent.</span></li>
        </ul>`,
    },
    {
      target: () => chart("Tool path"),
      kicker: "Where the tool is",
      title: "Tool path",
      body: `<p>The cutter moves left to right along the part. The blue line is the distance already cut.</p>
        <ul class="amg-read"><li>${chip("rgba(192,122,82,.55)")}<span>Hatched band: the material zone where trouble can start. Its position is public; whether and where it resonates is not.</span></li></ul>`,
    },
    {
      target: () => chart("Hidden stability map") || $(".am-locked"),
      kicker: "The hidden truth",
      title: "Stability map",
      body: `<p>For every spindle speed (up) and position along the cut (across), how strongly this part would chatter.</p>
        <ul class="amg-read">
          <li>${chip("#d99a6c")}<span>Copper: chatter. The darker the background, the calmer.</span></li>
          <li>${chip(COLORS.ink2 || "#b5b2aa", "dash")}<span>Dashed line: the chatter boundary.</span></li>
          <li>${chip(COLORS.x, "line")}<span>Blue line: the speed the agent actually used. A good run leaves the copper band.</span></li>
          <li>${chip(COLORS.mute, "dot")}<span>Dotted lines: the start speed and the allowed candidates A, B and C.</span></li>
        </ul>
        <p class="amg-why">Built from the simulator's hidden parameters, never shown to the agent. For Unknown parts it stays locked until the end.</p>`,
    },
    {
      target: () => chart("Vibration"),
      kicker: "What the sensors hear",
      title: "Vibration",
      sweep: true,
      body: `<p>The raw accelerometer signal on three axes, for the last tenth of a second.</p>
        <ul class="amg-read">
          <li>${chip(COLORS.x, "line")}${chip(COLORS.y, "line")}${chip(COLORS.z, "line")}<span>X, Y and Z axes.</span></li>
        </ul>
        <p><b>How to read it:</b> even, repeating waves are the teeth hitting the metal, which is normal. Chatter adds a faster, bigger wave on top, and the signal gets rough and loud.</p>`,
    },
    {
      target: () => chart("Detector score"),
      kicker: "The ML detector",
      title: "Detector score",
      sweep: true,
      body: `<p>Every 0.1 s a RandomForest model reads 23 features of the last 250 ms of signal and scores it from 0 (calm) to 1 (chatter).</p>
        <ul class="amg-read">
          <li>${chip(COLORS.bad, "dash")}<span>Above 0.8 three times in a row: the agent opens an incident.</span></li>
          <li>${chip(COLORS.good, "dash")}<span>Below 0.3 during Verify: calm again.</span></li>
        </ul>
        <p class="amg-why">It is a model score, not the probability of a failure. A spike that drops back quickly is chatter caught and fixed.</p>`,
    },
    {
      target: () => [chart("Spindle speed"), chart("Feed")].filter(Boolean).map(box),
      kicker: "What the agent changes",
      title: "Spindle speed and feed",
      sweep: true,
      body: `<p>These are the agent's actions. Each step up or down is a command the controller accepted.</p>
        <ul class="amg-read">
          <li>${chip(COLORS.x, "line")}<span>Actual spindle speed. The slope is the machine's speed limit, not a delay.</span></li>
          <li>${chip(COLORS.z, "line")}<span>Actual feed. Slower feed means calmer cutting, but a longer cut.</span></li>
          <li>${chip(COLORS.mute, "dot")}<span>Dotted line: the target the agent asked for.</span></li>
        </ul>`,
    },
    {
      target: () => chart("Spectrum"),
      kicker: "What the detector looks at",
      title: "Spectrum",
      body: `<p>How much energy the vibration has at each frequency, from the latest window.</p>
        <ul class="amg-read">
          <li>${chip(COLORS.mute, "dot")}<span>Dotted lines: tooth-pass harmonics. Peaks here are the normal sound of cutting.</span></li>
          <li>${chip(COLORS.x, "line")}<span>A strong peak <b>between</b> them is chatter.</span></li>
        </ul>
        <p class="amg-why">That is why louder cutting alone does not fool the agent: it looks at where the energy is, not only how much there is.</p>`,
    },
    {
      target: () => headerAndNext("Agent plan"),
      kicker: "Planning",
      title: "Agent plan",
      body: `<p>When an incident opens, the agent ranks the allowed speed and feed pairs and picks one.</p>
        <p>A pair is skipped when it was already tried, is the current one, or is outside the machine limits. A fix remembered from earlier on the same zone comes first.</p>
        <p class="amg-why">Only pre-approved pairs are allowed, so the agent can never send an arbitrary speed to the machine.</p>`,
    },
    {
      target: () => headerAndNext("Decision log"),
      kicker: "Every step, on record",
      title: "Decision log",
      body: `<p>Everything the agent did, newest first: what it saw, why it decided, which command it sent and what came back.</p>
        <ul class="amg-read">
          <li>${chip("#c07a52")}<span>Plans and commands.</span></li>
          <li>${chip(COLORS.warn)}<span>Recorded outcomes.</span></li>
          <li>${chip(COLORS.good)}<span>Recovery claims (still checked by the evaluator below).</span></li>
          <li>${chip(COLORS.bad)}<span>Stops.</span></li>
        </ul>`,
    },
    {
      target: () => {
        const v = liveOnly(".am-verdict");
        if (!v) return null;
        const head = box(v);
        return [head, head.nextElementSibling].filter(Boolean);
      },
      kicker: "Who grades the agent",
      title: "Independent verdict",
      body: `<p>When a run ends, a separate evaluator checks the simulator's hidden state, which the agent never sees, and grades every claim the agent made.</p>
        <ul class="amg-read">
          <li>${chip(COLORS.good)}<span><b>Run succeeded</b>: every chatter episode was really fixed.</span></li>
          <li>${chip(COLORS.warn)}<span><b>Stopped safely</b>: there was no safe speed or the data was bad, so stopping was the right call.</span></li>
          <li>${chip(COLORS.bad)}<span><b>Did not succeed</b>: something was missed or a claim was false.</span></li>
        </ul>
        <p class="amg-why">The agent is never its own judge.</p>`,
    },
    {
      target: () => chart("Hidden chatter intensity"),
      kicker: "What really happened",
      title: "Hidden chatter intensity",
      sweep: true,
      body: `<p>The true strength of chatter inside the part over time, revealed after the run.</p>
        <ul class="amg-read">
          <li>${chip(COLORS.y, "line")}<span>With the agent. The spike is the chatter episode; its width is how long the agent needed.</span></li>
          <li>${chip(COLORS.mute, "dot")}<span>The same part with nobody in control.</span></li>
          <li>${chip(COLORS.bad, "dash")}<span>Chatter starts.</span></li>
          <li>${chip(COLORS.x, "line")}<span>Each speed change the agent made.</span></li>
        </ul>
        <p class="amg-why">The gap between the two lines is what the agent saved.</p>`,
    },
    {
      target: () => headerAndNext("Mode comparison"),
      kicker: "Mode comparison",
      title: "One part, four ways to control it",
      body: `<p>The same part, seed, noise and disturbances are run four times. Only the controller changes, so any difference in the result comes from the controller alone.</p>
        <ul class="amg-read">
          <li>${chip("#5a5d62")}<span><b>Nobody in control</b>: the 'before' picture.</span></li>
          <li>${chip("#7d7b75")}<span><b>Simple loudness rule</b>: slows the feed once when it gets loud.</span></li>
          <li>${chip("#c0703f")}<span><b>Loudness rule + candidates</b>: the agent's own plan and checks, but decided by loudness. This is the fair rival.</span></li>
          <li>${chip("#4f86c6")}<span><b>ML agent</b>: ours.</span></li>
        </ul>
        <p class="amg-why">The part is the preset and seed chosen in the sidebar. Same choice, same result; change either to compare on another part.</p>`,
    },
    {
      target: () => (headerAndNext("Result") || []).filter((el) => inCompare(el)),
      kicker: "Mode comparison",
      title: "What this shows",
      body: `<p>The first line is written from the numbers: how long the part chatters with nobody in control, how much the agent saves, and whether the loudness rule raised false alarms.</p>
        <p>Below it, one card per mode with the same verdict as a live run: <b>Succeeded</b>, <b>Stopped safely</b> or <b>Chatter not fixed</b>.</p>
        <p class="amg-why"><b>Tip:</b> on <b>Louder, still stable</b> the loudness rule reacts to plain loud cutting and the ML agent does not. That is the detector earning its place.</p>`,
    },
    {
      target: () => inCompare(chart("Time spent in chatter") || chart("False alarms")) || (comparePanel() ? $$(".am-verdict", comparePanel()).find(visible) || null : null),
      kicker: "Mode comparison",
      title: "The comparison chart",
      body: `<p>One bar per mode; the colours match the mode cards above.</p>
        <ul class="amg-read">
          <li>${chip(COLORS.y)}<span>If the part chatters: <b>time spent in chatter</b>, measured from the hidden truth. Shorter is better.</span></li>
          <li>${chip(COLORS.x)}<span>If it never chatters: <b>false alarms</b>, commands sent for nothing. Fewer is better, and zero is the goal.</span></li>
        </ul>
        <p class="amg-why">Equal bars mean the two controllers did equally well. When every mode scores zero there is nothing to compare, so a single line says so instead of an empty chart.</p>`,
    },
    {
      target: () => $$('[data-testid="stDataFrame"]', comparePanel() || document.createElement("div")).find(visible),
      kicker: "Mode comparison",
      title: "The results table",
      body: `<p>One row per mode. Read it left to right:</p>
        <ul class="amg-read">
          <li>${chip(COLORS.mute)}<span><b>Episodes</b>: chatter episodes the evaluator found in the part. <b>Fixed</b> were really recovered; <b>Missed</b> never were.</span></li>
          <li>${chip(COLORS.y)}<span><b>In chatter, s</b>: total time in chatter. <b>Detected, s</b>: from chatter starting to the controller noticing it.</span></li>
          <li>${chip(COLORS.x)}<span><b>Commands</b> sent, and <b>False alarms</b>: commands with no real chatter behind them.</span></li>
          <li>${chip(COLORS.z)}<span><b>Cut, s</b>: time to finish the path (— if the cut was stopped). <b>Extra time</b>: how much longer than with nobody in control, the price of slowing the feed.</span></li>
        </ul>
        <p class="amg-why"><b>The best row</b> has 0 missed, 0 false alarms and the least time in chatter, at a small extra time.</p>`,
    },
    {
      target: () => $('[role="tablist"]'),
      kicker: "Go further",
      title: "Catalog and comparison",
      body: `<p><b>Scenario catalog</b> explains every preset in plain words, including the failure tests: broken sensor data, a rejected command, a lost acknowledgment, a manual stop.</p>
        <p><b>Mode comparison</b> runs every control mode on the same part, so you can compare the ML agent with simple threshold rules.</p>`,
    },
    {
      kicker: "Done",
      title: "That's the whole screen",
      body: `<p>Everything is back exactly as it was. Press <b>Guide</b> any time to see this again.</p>
        <p class="amg-why">Synthetic simulation; not validated on real CNC.</p>`,
    },
  ];

  // ---- the overlay ----------------------------------------------------------------

  let root, svg, shade, cut, ring, sweep, card, progress, count, backBtn, nextBtn;
  let steps = [], index = 0, raf = 0, last = 0, savedScroll = [], lastFocus = null;
  const rect = { x: 0, y: 0, w: 0, h: 0, r: 14, shown: 0 };
  const goal = { x: 0, y: 0, w: 0, h: 0, shown: 0 };
  let ringDrawn = 1, sweepAt = -1;

  function svgEl(tag, attrs) {
    const el = document.createElementNS(NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }

  function build() {
    root = document.createElement("div");
    root.className = "amg-root";
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
    root.setAttribute("aria-label", "Screen guide");

    svg = svgEl("svg", { class: "amg-veil" });
    const defs = svgEl("defs", {});
    const mask = svgEl("mask", { id: "amg-mask" });
    mask.appendChild(svgEl("rect", { x: 0, y: 0, width: "100%", height: "100%", fill: "white" }));
    cut = svgEl("rect", { fill: "black", rx: 14 });
    mask.appendChild(cut);
    const ringGrad = svgEl("linearGradient", { id: "amg-ring-grad", x1: "0", y1: "0", x2: "1", y2: "1" });
    ringGrad.appendChild(svgEl("stop", { offset: "0", "stop-color": "#7fa8d8" }));
    ringGrad.appendChild(svgEl("stop", { offset: "0.5", "stop-color": "#4f86c6", "stop-opacity": "0.55" }));
    ringGrad.appendChild(svgEl("stop", { offset: "1", "stop-color": "#c0703f", "stop-opacity": "0.75" }));
    const sweepGrad = svgEl("linearGradient", { id: "amg-sweep-grad", x1: "0", y1: "0", x2: "1", y2: "0" });
    sweepGrad.appendChild(svgEl("stop", { offset: "0", "stop-color": "#7fa8d8", "stop-opacity": "0" }));
    sweepGrad.appendChild(svgEl("stop", { offset: "0.7", "stop-color": "#7fa8d8", "stop-opacity": "0.14" }));
    sweepGrad.appendChild(svgEl("stop", { offset: "1", "stop-color": "#cfe0f5", "stop-opacity": "0.45" }));
    const clip = svgEl("clipPath", { id: "amg-clip" });
    const clipRect = svgEl("rect", { rx: 14 });
    clip.appendChild(clipRect);
    defs.append(mask, ringGrad, sweepGrad, clip);
    shade = svgEl("rect", { x: 0, y: 0, width: "100%", height: "100%", class: "amg-shade", mask: "url(#amg-mask)" });
    sweep = svgEl("rect", { class: "amg-sweep", "clip-path": "url(#amg-clip)", opacity: 0 });
    ring = svgEl("rect", { class: "amg-ring", rx: 14 });
    svg.append(defs, shade, sweep, ring);
    svg._clipRect = clipRect;

    card = document.createElement("div");
    card.className = "amg-card";
    card.innerHTML = `
      <button class="amg-close" aria-label="Close guide">×</button>
      <div class="amg-content"></div>
      <div class="amg-foot">
        <button class="amg-btn amg-back">Back</button>
        <div class="amg-progress"><i></i></div>
        <span class="amg-count"></span>
        <button class="amg-btn primary amg-next">Next</button>
      </div>
      <div class="amg-keys"><kbd>←</kbd> <kbd>→</kbd> to move · <kbd>Esc</kbd> to close</div>`;
    progress = $(".amg-progress i", card);
    count = $(".amg-count", card);
    backBtn = $(".amg-back", card);
    nextBtn = $(".amg-next", card);
    backBtn.onclick = () => go(index - 1);
    nextBtn.onclick = () => (index >= steps.length - 1 ? close() : go(index + 1));
    $(".amg-close", card).onclick = close;

    root.append(svg, card);
    document.body.appendChild(root);
  }

  // the steps that make sense on the screen as it is right now
  function available() {
    return STEPS.filter((s) => {
      if (!s.target) return true;
      const t = s.target();
      const list = Array.isArray(t) ? t : [t];
      return list.some(visible);
    });
  }

  function targetsOf(step) {
    if (!step.target) return [];
    const t = step.target();
    return (Array.isArray(t) ? t : [t]).filter(visible);
  }

  function unionRect(els) {
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    for (const el of els) {
      const r = el.getBoundingClientRect();
      x1 = Math.min(x1, r.left); y1 = Math.min(y1, r.top);
      x2 = Math.max(x2, r.right); y2 = Math.max(y2, r.bottom);
    }
    const pad = 8;
    return { x: x1 - pad, y: y1 - pad, w: x2 - x1 + 2 * pad, h: y2 - y1 + 2 * pad };
  }

  function scrollers() {
    return $$("*").filter((el) => {
      const s = getComputedStyle(el);
      return /(auto|scroll)/.test(s.overflowY) && el.scrollHeight > el.clientHeight + 20;
    });
  }

  // ---- motion ---------------------------------------------------------------------

  // critically damped follow: the spotlight glides like a physical object, and it also
  // keeps up when the page scrolls or a chart redraws under it
  function tick(now) {
    const dt = Math.min(64, now - (last || now));
    last = now;
    const k = reduce() ? 1 : 1 - Math.exp(-dt / 95);

    const step = steps[index];
    const els = targetsOf(step);
    if (els.length) {
      const u = unionRect(els);
      Object.assign(goal, u, { shown: 1 });
    } else {
      // no target: close the spotlight into the centre of the screen
      Object.assign(goal, { x: innerWidth / 2, y: innerHeight / 2, w: 0, h: 0, shown: 0 });
    }
    for (const key of ["x", "y", "w", "h", "shown"]) rect[key] += (goal[key] - rect[key]) * k;

    for (const el of [cut, ring, svg._clipRect]) {
      el.setAttribute("x", rect.x); el.setAttribute("y", rect.y);
      el.setAttribute("width", Math.max(0, rect.w)); el.setAttribute("height", Math.max(0, rect.h));
    }
    ring.setAttribute("opacity", rect.shown.toFixed(3));

    // the outline draws itself in when it arrives
    const perimeter = 2 * (Math.max(0, rect.w) + Math.max(0, rect.h));
    ringDrawn += (1 - ringDrawn) * (reduce() ? 1 : 1 - Math.exp(-dt / 220));
    ring.setAttribute("stroke-dasharray", perimeter);
    ring.setAttribute("stroke-dashoffset", (perimeter * (1 - ringDrawn)).toFixed(1));

    // one slow light sweep across time-based charts: "read it from left to right"
    if (sweepAt >= 0 && !reduce()) {
      const p = Math.min(1, (now - sweepAt) / 1400);
      const e = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
      const w = Math.max(40, rect.w * 0.18);
      sweep.setAttribute("x", rect.x - w + (rect.w + w) * e);
      sweep.setAttribute("y", rect.y);
      sweep.setAttribute("width", w);
      sweep.setAttribute("height", Math.max(0, rect.h));
      sweep.setAttribute("opacity", Math.sin(Math.PI * p).toFixed(3));
      if (p >= 1) { sweepAt = -1; sweep.setAttribute("opacity", 0); }
    }

    placeCard();
    raf = requestAnimationFrame(tick);
  }

  let cardPos = null;
  function placeCard() {
    const cw = card.offsetWidth, ch = card.offsetHeight, m = 18, vw = innerWidth, vh = innerHeight;
    let x, y;
    if (goal.shown < 0.5) {
      x = (vw - cw) / 2; y = (vh - ch) / 2;
    } else if (goal.x + goal.w + m + cw < vw - 8) {
      x = goal.x + goal.w + m; y = goal.y;
    } else if (goal.x - m - cw > 8) {
      x = goal.x - m - cw; y = goal.y;
    } else if (goal.y + goal.h + m + ch < vh - 8) {
      x = goal.x + (goal.w - cw) / 2; y = goal.y + goal.h + m;
    } else if (goal.y - m - ch > 8) {
      x = goal.x + (goal.w - cw) / 2; y = goal.y - m - ch;
    } else {
      x = vw - cw - 24; y = vh - ch - 24;
    }
    x = Math.max(12, Math.min(vw - cw - 12, x));
    y = Math.max(12, Math.min(vh - ch - 12, y));
    if (!cardPos || reduce()) cardPos = { x, y };
    const k = 0.16;
    cardPos.x += (x - cardPos.x) * k;
    cardPos.y += (y - cardPos.y) * k;
    card.style.transform = `translate3d(${cardPos.x.toFixed(1)}px, ${cardPos.y.toFixed(1)}px, 0)`;
  }

  function renderContent(step, direction) {
    const content = $(".amg-content", card);
    const html = `<div class="amg-kicker">${step.kicker}</div><div class="amg-title">${step.title}</div>
      <div class="amg-body">${step.body}${step.noRunHint && !$(".am-tiles") ? step.noRunHint : ""}</div>`;
    const swap = () => {
      content.innerHTML = html;
      if (reduce()) return;
      // staggered entrance: title first, then each paragraph and bullet, 60 ms apart
      const parts = [$(".amg-kicker", content), $(".amg-title", content), ...$$(".amg-body > *, .amg-read li", content)];
      parts.forEach((el, i) => {
        el.animate(
          [{ opacity: 0, transform: `translateY(${8 * direction}px)`, filter: "blur(3px)" }, { opacity: 1, transform: "none", filter: "blur(0)" }],
          { duration: 460, delay: Math.min(i, 8) * 55, easing: EASE_OUT, fill: "backwards" }
        );
      });
      const logo = $(".amg-logo", content);
      if (logo) {
        $$("circle, path", logo).forEach((p, i) => {
          const len = p.getTotalLength ? p.getTotalLength() : 40;
          p.style.strokeDasharray = len;
          p.animate([{ strokeDashoffset: len }, { strokeDashoffset: 0 }], { duration: 900, delay: 120 + i * 90, easing: EASE_OUT, fill: "backwards" });
        });
      }
    };
    if (content.childElementCount && !reduce()) {
      content.animate([{ opacity: 1, transform: "none" }, { opacity: 0, transform: `translateY(${-6 * direction}px)` }], { duration: 150, easing: EASE_IN }).onfinish = swap;
    } else {
      swap();
    }
  }

  function go(i) {
    if (i < 0 || i >= steps.length) return;
    const direction = i >= index ? 1 : -1;
    index = i;
    const step = steps[index];
    const els = targetsOf(step);
    if (els.length) els[0].scrollIntoView({ behavior: reduce() ? "auto" : "smooth", block: "center", inline: "nearest" });
    ringDrawn = 0;
    sweepAt = step.sweep ? performance.now() + 450 : -1;
    renderContent(step, direction);
    backBtn.disabled = index === 0;
    nextBtn.textContent = index === steps.length - 1 ? "Finish" : "Next";
    count.textContent = `${index + 1} / ${steps.length}`;
    progress.style.transform = `scaleX(${(index + 1) / steps.length})`;
    nextBtn.focus({ preventScroll: true });
  }

  function onKey(e) {
    if (!root) return;
    if (e.key === "Escape") { e.preventDefault(); close(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); index >= steps.length - 1 ? close() : go(index + 1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); go(index - 1); }
    else if (e.key === "Tab") {
      // keep keyboard focus inside the card while it is open
      const f = $$("button", card).filter((b) => !b.disabled);
      if (!f.length) return;
      const at = f.indexOf(document.activeElement);
      if (e.shiftKey && at <= 0) { e.preventDefault(); f[f.length - 1].focus(); }
      else if (!e.shiftKey && at === f.length - 1) { e.preventDefault(); f[0].focus(); }
    }
  }

  function open() {
    if (root) return;
    lastFocus = document.activeElement;
    savedScroll = scrollers().map((el) => [el, el.scrollTop]);
    steps = available();
    index = 0;
    build();
    Object.assign(rect, { x: innerWidth / 2, y: innerHeight / 2, w: 0, h: 0, shown: 0 });
    cardPos = null;
    if (!reduce()) {
      shade.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 420, easing: EASE_OUT });
      card.animate([{ opacity: 0, scale: "0.96" }, { opacity: 1, scale: "1" }], { duration: 520, easing: EASE_OUT });
    }
    document.addEventListener("keydown", onKey, true);
    last = 0;
    raf = requestAnimationFrame(tick);
    go(0);
  }

  function close() {
    if (!root) return;
    const node = root;
    document.removeEventListener("keydown", onKey, true);
    // exit mirrors the entrance, then the page is put back exactly where it was
    const done = () => {
      cancelAnimationFrame(raf);
      node.remove();
      root = null;
      for (const [el, top] of savedScroll) if (el.isConnected) el.scrollTo({ top, behavior: reduce() ? "auto" : "smooth" });
      savedScroll = [];
      // hand focus back to the Guide button (the header may have been redrawn meanwhile)
      const again = $(".am-guide-btn");
      if (again) again.focus({ preventScroll: true });
      else if (document.activeElement) document.activeElement.blur();
    };
    if (reduce()) return done();
    card.animate([{ opacity: 1, scale: "1" }, { opacity: 0, scale: "0.97" }], { duration: 220, easing: EASE_IN, fill: "forwards" });
    shade.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 320, easing: EASE_IN, fill: "forwards" });
    ring.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 220, easing: EASE_IN, fill: "forwards" });
    setTimeout(done, 330);
  }

  // ---- the Guide button in the header ----------------------------------------------

  const ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M9.6 9.3a2.5 2.5 0 1 1 3.4 2.3c-.6.3-1 .8-1 1.5v.4M12 16.8h.01"/></svg>';

  function ensureButton() {
    const head = $(".am-head");
    if (!head || $(".am-guide-btn", head)) return;
    const b = document.createElement("button");
    b.className = "am-guide-btn";
    b.type = "button";
    b.innerHTML = ICON + "<span>Guide</span>";
    b.title = "A short tour of the screen";
    b.onclick = open;
    const badge = $(".am-badge", head);
    badge ? head.insertBefore(b, badge.nextSibling) : head.appendChild(b);
  }

  // Streamlit redraws the header on reruns, so put the button back whenever it disappears
  new MutationObserver(ensureButton).observe(document.body, { childList: true, subtree: true });
  ensureButton();

  window.__amGuide = { open, close };
})();
