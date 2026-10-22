"use strict";

// ---------- state ----------
const me = { code: null, seat: null, token: null, host: false };
let ws = null;
let room = null;      // latest "room" message
let game = null;      // latest "state" message
let presets = [];
let selection = [];   // indices into game.hand
let autoStart = false;

const $ = (id) => document.getElementById(id);
const TRICK_TYPES = { set: "singles or sets", run: "singles or runs" };
const ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th"];

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function card(rank, small) {
  return el("div", `card r${rank}${small ? " small" : ""}`, String(rank));
}

function cardRow(ranks, small) {
  const row = el("div", "cards");
  ranks.forEach((r) => row.append(card(r, small)));
  return row;
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg.charAt(0).toUpperCase() + msg.slice(1);
  t.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.add("hidden"), 3500);
}

function playerName() {
  const name = $("name").value.trim() || "Player";
  localStorage.setItem("rainbow.name", name);
  return name;
}

function seatName(seat) {
  const s = room && room.seats[seat];
  if (!s) return `Seat ${seat + 1}`;
  if (s.kind === "open") return "Open seat";
  return s.name;
}

// ---------- sound and tab title ----------
let soundOn = false;
try { soundOn = localStorage.getItem("rainbow.sound") === "on"; } catch { /* default off */ }
let audio = null;

/** A short sine tone sequence: [frequency, start offset] pairs, gently enveloped. */
function tones(notes, length = 0.12, volume = 0.06) {
  if (!soundOn) return;
  try {
    audio = audio || new AudioContext();
    const now = audio.currentTime;
    for (const [freq, at] of notes) {
      const osc = audio.createOscillator();
      const gain = audio.createGain();
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0, now + at);
      gain.gain.linearRampToValueAtTime(volume, now + at + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + at + length);
      osc.connect(gain).connect(audio.destination);
      osc.start(now + at);
      osc.stop(now + at + length + 0.02);
    }
  } catch { /* audio unavailable */ }
}
const SOUNDS = {
  turn: () => tones([[660, 0], [880, 0.1]]),
  trick: () => tones([[523, 0], [659, 0.07], [784, 0.14]], 0.18, 0.04),
  chat: () => tones([[990, 0]], 0.08, 0.03),
};


function updateTitle() {
  const myTurn = game && !game.over && game.seat !== null && game.current === game.seat;
  document.title = myTurn && document.hidden ? "● Your turn · Rainbow" : "Rainbow";
}
document.addEventListener("visibilitychange", updateTitle);

// ---------- chat ----------
function addChat(msg) {
  const log = $("chat-log");
  const li = el("li");
  li.append(el("b", null, msg.seat === me.seat && msg.seat !== null ? "You" : msg.from), document.createTextNode(` ${msg.text}`));
  log.append(li);
  while (log.children.length > 100) log.firstChild.remove();
  log.scrollTop = log.scrollHeight;
}

// ---------- connection ----------
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => {
    $("status").textContent = "";
    if (me.code) send({ t: "join", code: me.code, name: playerName(), token: me.token });
    // then anything the player did while the socket was connecting
    while (outbox.length) send(outbox.shift());
  };
  ws.onmessage = (ev) => receive(JSON.parse(ev.data));
  ws.onclose = () => {
    $("status").textContent = "Reconnecting…";
    setTimeout(connect, 1500);
  };
}

// Messages sent while (re)connecting wait here and go out once the socket opens.
const outbox = [];

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  else outbox.push(msg);
}

function receive(msg) {
  switch (msg.t) {
    case "joined":
      me.code = msg.code;
      me.seat = msg.seat;
      me.token = msg.token || null;
      me.host = msg.host;
      if (me.token) localStorage.setItem(`rainbow.token.${me.code}`, me.token);
      history.replaceState(null, "", `/?room=${me.code}`);
      break;
    case "room":
      room = msg;
      if (room.phase === "lobby") game = null;
      if (autoStart && me.host && room.phase === "lobby" && !room.seats.some((s) => s.kind === "open")) {
        autoStart = false;
        send({ t: "start" });
      }
      break;
    case "state": {
      const old = game;
      const oldHand = old && old.hand ? old.hand.join() : "";
      game = msg; // set first: announcements name seats relative to this state
      announceChanges(old && old.tricks_played <= msg.tricks_played ? old : null, msg);
      if (!game.hand || game.hand.join() !== oldHand) selection = [];
      break;
    }
    case "chat":
      addChat(msg);
      if (msg.seat !== me.seat || msg.seat === null) SOUNDS.chat();
      return;
    case "hint": {
      const l = game && game.legal.find((x) => x.action === msg.action);
      if (l) {
        selectCombo(l);
        renderHand();
      }
      return;
    }
    case "error":
      toast(msg.msg);
      if (msg.msg === "no room with that code" || msg.msg === "the room has closed") leave();
      break;
  }
  render();
}

function leave() {
  me.code = null;
  me.seat = null;
  me.token = null;
  me.host = false;
  room = null;
  game = null;
  $("chat-log").innerHTML = "";
  history.replaceState(null, "", "/");
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ t: "leave" }));
  render();
}

// ---------- home ----------
function presetOptions(select, withOpen, value) {
  select.innerHTML = "";
  if (withOpen) select.append(new Option("Open for a friend", "open"));
  presets.forEach((p) => {
    const o = new Option(p.label, p.id);
    o.title = p.description;
    select.append(o);
  });
  if (value) select.value = value;
}

function seatPickers(container, count, fallback, saved) {
  const old = [...container.querySelectorAll("select")].map((s) => s.value);
  container.innerHTML = "";
  for (let i = 1; i < count; i++) {
    const label = el("label", null, `Seat ${i + 1}`);
    const select = el("select");
    presetOptions(select, false, old[i - 1] || (saved && saved[i - 1]) || fallback);
    if (!select.value) select.value = fallback;
    select.onchange = saveSetup;
    label.append(select);
    container.append(label);
  }
}

function saveSetup() {
  try {
    localStorage.setItem("rainbow.setup", JSON.stringify({
      solo: +$("solo-players").value,
      opponent: $("solo-opponent").value,
      custom: $("solo-custom").open,
      soloSeats: [...$("solo-seats").querySelectorAll("select")].map((s) => s.value),
      online: +$("online-players").value,
    }));
  } catch { /* storage unavailable: nothing to remember */ }
}

function loadSetup() {
  try { return JSON.parse(localStorage.getItem("rainbow.setup")) || {}; } catch { return {}; }
}

function setupHome() {
  for (const id of ["solo-players", "online-players"]) {
    const s = $(id);
    for (let n = 2; n <= 6; n++) s.append(new Option(`${n} players`, n));
    s.value = 4;
  }
  const saved = loadSetup();
  if (saved.solo) $("solo-players").value = saved.solo;
  if (saved.online) $("online-players").value = saved.online;
  const defaultAgent = presets.some((p) => p.id === "dmc") ? "dmc" : presets[0] && presets[0].id;
  presetOptions($("solo-opponent"), false, saved.opponent || defaultAgent);
  if (!$("solo-opponent").value) $("solo-opponent").value = defaultAgent;
  $("solo-custom").open = !!saved.custom;
  const refresh = () => {
    seatPickers($("solo-seats"), +$("solo-players").value, $("solo-opponent").value, saved.soloSeats);
    saveSetup();
  };
  $("solo-players").onchange = refresh;
  $("solo-opponent").onchange = () => {
    // a new opponent choice applies to every seat
    $("solo-seats").querySelectorAll("select").forEach((s) => { s.value = $("solo-opponent").value; });
    saveSetup();
  };
  $("solo-custom").ontoggle = saveSetup;
  $("online-players").onchange = saveSetup;
  refresh();

  $("solo-start").onclick = () => {
    autoStart = true;
    const n = +$("solo-players").value;
    const seats = $("solo-custom").open
      ? [...$("solo-seats").querySelectorAll("select")].map((s) => s.value)
      : Array(n - 1).fill($("solo-opponent").value);
    send({ t: "create", name: playerName(), seats });
  };
  // online rooms start with every other seat open; the host can fill seats in the lobby
  $("online-create").onclick = () =>
    send({ t: "create", name: playerName(), seats: Array(+$("online-players").value - 1).fill("open") });
  const join = () => {
    const code = $("join-code").value.trim().toUpperCase();
    if (code.length !== 4) return toast("Room codes have four letters");
    me.code = code;
    me.token = localStorage.getItem(`rainbow.token.${code}`);
    send({ t: "join", code, name: playerName(), token: me.token });
  };
  $("join-btn").onclick = join;
  $("join-code").onkeydown = (e) => { if (e.key === "Enter") join(); };
}

// ---------- lobby ----------
function renderLobby() {
  $("lobby-code").textContent = room.code;
  $("share-link").value = `${location.origin}/?room=${room.code}`;
  const list = $("lobby-seats");
  list.innerHTML = "";
  room.seats.forEach((s, i) => {
    const li = el("li");
    const line = el("div", "seat-line");
    const who = el("span");
    if (s.kind === "human") {
      who.append(document.createTextNode(s.name + " "));
      if (i === me.seat) who.append(el("span", "tag you", "you"));
      if (s.host) who.append(el("span", "tag", "host"));
      if (!s.connected) who.append(el("span", "tag off", "away"));
    } else if (s.kind === "agent") {
      who.append(document.createTextNode(s.name + " "), el("span", "tag", "agent"));
    } else {
      who.append(el("span", "hint", "Waiting for a player…"));
    }
    line.append(who);
    const editable = me.host && i !== me.seat && !(s.kind === "human" && s.connected);
    if (editable) {
      const select = el("select");
      presetOptions(select, true, s.kind === "agent" ? s.preset : "open");
      select.setAttribute("aria-label", `Player for seat ${i + 1}`);
      select.onchange = () => send({ t: "seat", seat: i, agent: select.value === "open" ? null : select.value });
      line.append(select);
    }
    li.append(line);
    list.append(li);
  });
  const open = room.seats.filter((s) => s.kind === "open").length;
  $("lobby-start").classList.toggle("hidden", !me.host);
  $("lobby-start").disabled = open > 0;
  $("lobby-hint").textContent = me.host
    ? open > 0 ? "Share the link, or let the computer take the open seats." : "Everyone is here."
    : me.seat === null ? "The room is full; you are watching." : "Waiting for the host to start the game.";
}

// ---------- table ----------
let freshFrom = Infinity; // plays in the current trick from this index on are new: animate them
let gainFlash = null;     // { at, taken }: points each seat just took, for a brief highlight
const GAIN_MS = 900;

/** Whether a seat's points changed a moment ago (they get a short highlight). */
function justScored(seat) {
  return !!gainFlash && gainFlash.taken[seat] > 0 && Date.now() - gainFlash.at < GAIN_MS;
}

function who(seat) {
  return seat === game.seat ? "You" : seatName(seat);
}

function cardsText(cards) {
  return cards.join(" ");
}

/** Screen-reader (and log) announcements for what changed between two states. */
function announceChanges(old, next) {
  const live = $("live");
  const said = [];
  const oldLog = old ? old.log.length : 0;
  const oldTrick = old && old.log.length === next.log.length ? old.trick.length : 0;
  if (next.log.length > oldLog) {
    const done = next.log[next.log.length - 1];
    // plays of the finished trick not yet announced
    done.plays.slice(oldLog === next.log.length - 1 && old ? old.trick.length : 0)
      .forEach((pl) => said.push(`${who(pl.seat)} played ${cardsText(pl.cards)}.`));
    const gains = [...new Set(done.plays.map((pl) => pl.seat))]
      .map((seat) => `${who(seat)} +${done.taken[seat]}`);
    said.push(`Trick over: ${gains.join(", ")}.`);
  }
  next.trick.slice(oldTrick).forEach((pl) => said.push(`${who(pl.seat)} played ${cardsText(pl.cards)}.`));
  if (!next.over && next.current === next.seat && (!old || old.current !== next.seat)) said.push("Your turn.");
  if (next.over && (!old || !old.over)) said.push("Game over.");
  if (said.length) live.textContent = said.join(" ");
  if (next.log.length > oldLog && old) {
    SOUNDS.trick();
    gainFlash = { at: Date.now(), taken: next.log[next.log.length - 1].taken };
  }
  if (!next.over && next.current === next.seat && (!old || old.current !== next.seat)) SOUNDS.turn();
  freshFrom = oldTrick;
}

function matchingLegal() {
  if (!game || !game.legal.length || !selection.length) return null;
  const ranks = selection.map((i) => game.hand[i]).sort((a, b) => a - b).join();
  return game.legal.find((l) => l.cards.join() === ranks) || null;
}

/** Why the selected cards can't be played, in a few words for the Play button. */
function whyNot() {
  const ranks = selection.map((i) => game.hand[i]).sort((a, b) => a - b);
  const set = ranks.every((r) => r === ranks[0]);
  const run = ranks.every((r, i) => i === 0 || r === ranks[i - 1] + 1);
  if (!set && !run) return "Pick equal cards or a run";
  if (game.trick_type !== "open") return `Only ${TRICK_TYPES[game.trick_type]} now`;
  return "Can't play that";
}

/** Selects the hand cards of a legal play. */
function selectCombo(l) {
  const used = new Set();
  selection = l.cards.map((r) => {
    const i = game.hand.findIndex((x, j) => x === r && !used.has(j));
    used.add(i);
    return i;
  });
}

function play(action) {
  selection = [];
  send({ t: "play", action });
}

function renderOpponents(spectator, base, n) {
  const opps = $("opponents");
  opps.innerHTML = "";
  for (let k = spectator ? 0 : 1; k < n; k++) {
    const seat = (base + k) % n;
    const p = game.players[seat];
    const s = room.seats[seat] || {};
    const turn = game.current === seat;
    const box = el("div", "opp" + (turn ? " turn" : ""));
    const head = el("div", "who");
    head.append(
      el("span", turn && s.kind === "agent" ? "thinking" : "", seatName(seat)),
      el("span", "kind offline", s.connected === false ? "away" : ""),
    );
    const meta = el("div", "meta");
    const count = el("span", "count");
    const back = el("span", "back");
    back.setAttribute("aria-hidden", "true");
    count.append(back, document.createTextNode(
      `${p.hand_size}${p.reserve_size ? ` + ${p.reserve_size}` : ""}`));
    count.setAttribute("aria-label", `${p.hand_size} cards${p.reserve_size ? `, ${p.reserve_size} in reserve` : ""}`);
    meta.append(count, el("span", "points" + (justScored(seat) ? " bump" : ""), `${p.points} pts`));
    box.append(head, meta);
    if (me.host && s.kind === "human" && !s.connected && !game.over) {
      const select = el("select");
      select.setAttribute("aria-label", `Replace ${seatName(seat)} with an agent`);
      presetOptions(select, false);
      select.prepend(new Option("Replace with an agent…", ""));
      select.value = "";
      select.onchange = () => select.value && send({ t: "seat", seat, agent: select.value });
      box.append(select);
    }
    opps.append(box);
  }
}

function renderMiddle() {
  const piles = $("piles");
  piles.innerHTML = "";
  game.piles.forEach((v, i) => {
    const pile = el("div", "pile");
    pile.setAttribute("aria-label", `${ORDINALS[i]} pile: ${v} points`);
    pile.append(el("b", null, String(v)), el("small", null, ORDINALS[i]));
    piles.append(pile);
  });
  if (!game.piles.length) piles.append(el("span", "hint", "No piles left"));

  // the current trick, or the last finished one until the next trick starts
  const trick = $("trick");
  trick.innerHTML = "";
  const current = !game.last_trick || (game.trick.length > 0 && !game.over);
  const plays = current ? game.trick : game.last_trick.plays;
  trick.classList.toggle("past", !current);
  $("trick-title").textContent = current ? `Trick ${game.tricks_played + 1}` : "Last trick";
  const restricted = current && game.trick_type !== "open";
  $("trick-type").textContent = restricted ? `Only ${TRICK_TYPES[game.trick_type]}` : "";
  $("trick-type").classList.toggle("hidden", !restricted);
  plays.forEach((pl, i) => {
    const leading = current && i === game.trick_leader;
    const box = el("div", "play" + (leading ? " leading" : "") + (current && i >= freshFrom ? " fresh" : ""));
    const by = el("div", "by", who(pl.seat));
    if (leading && plays.length > 1) by.append(el("span", "lead-tag", "leading"));
    const lastOfSeat = !plays.slice(i + 1).some((q) => q.seat === pl.seat);
    if (!current && lastOfSeat) by.append(el("span", "gain", `+${game.last_trick.taken[pl.seat]}`));
    box.append(by, cardRow(pl.cards, true));
    trick.append(box);
  });
  if (!plays.length) trick.append(el("span", "hint", `${who(game.current)} to lead`));
}

/** Runs a re-render that rebuilds buttons, keeping keyboard focus on the same logical control. */
function keepingFocus(fn) {
  const key = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.key : null;
  fn();
  if (key) {
    const again = document.querySelector(`[data-key="${key}"]`);
    if (again) again.focus();
  }
}

function renderHand() {
  keepingFocus(renderHandNow);
}

function renderHandNow() {
  const mine = game.players[game.seat];
  const myTurn = game.current === game.seat;
  $("me").classList.toggle("turn", myTurn);
  $("me-name").textContent = seatName(game.seat);
  $("me-points").textContent = `${mine.points} pts`;
  $("me-points").classList.toggle("bump", justScored(game.seat));
  $("me-reserve").textContent = mine.reserve_size ? `${mine.reserve_size} cards in reserve` : "";
  $("turn-hint").textContent = game.over ? "" : myTurn ? "Your turn" : `Waiting for ${seatName(game.current)}…`;

  const hand = $("hand");
  hand.innerHTML = "";
  let group = null;
  let lastRank = null;
  game.hand.forEach((rank, i) => {
    if (rank !== lastRank) {
      group = el("div", "group");
      hand.append(group);
      lastRank = rank;
    }
    const c = el("button", `card r${rank}`, String(rank));
    const selected = selection.includes(i);
    c.classList.toggle("sel", selected);
    c.setAttribute("aria-pressed", String(selected));
    c.setAttribute("aria-label", `Card ${rank}`);
    c.dataset.key = `card-${i}`;
    c.onclick = () => {
      selection = selected ? selection.filter((j) => j !== i) : [...selection, i];
      renderHand();
    };
    group.append(c);
  });
  if (!game.hand.length) hand.append(el("span", "hint", "No cards left"));

  const match = matchingLegal();
  const btn = $("play-selected");
  btn.disabled = !match;
  btn.textContent = match ? `Play ${cardsText(match.cards)}`
    : myTurn && selection.length ? whyNot()
    : myTurn ? "Select cards" : "Play";
  btn.classList.toggle("reason", !match && myTurn && selection.length > 0);
  btn.onclick = () => match && play(match.action);
  $("hint-btn").classList.toggle("hidden", !(room && room.hints && myTurn));
}

function renderSide() {
  // cards I haven't seen, per rank
  const unseen = $("unseen");
  unseen.innerHTML = "";
  $("unseen-box").classList.toggle("hidden", !game.unseen);
  (game.unseen || []).forEach((count, i) => {
    const cell = el("div", "unseen-cell");
    cell.append(card(i + 1, true), el("span", null, `×${count}`));
    cell.setAttribute("aria-label", `${count} unseen ${i + 1}s`);
    unseen.append(cell);
  });

  // trick log, newest first
  const log = $("log");
  log.innerHTML = "";
  game.log.slice().reverse().forEach((t, k) => {
    const li = el("li");
    li.append(el("div", "log-title", `Trick ${game.log.length - k}`));
    t.plays.forEach((pl, i) => {
      const line = el("div", "log-line");
      line.append(el("span", null, who(pl.seat)), cardRow(pl.cards, true));
      const lastOfSeat = !t.plays.slice(i + 1).some((q) => q.seat === pl.seat);
      if (lastOfSeat) line.append(el("span", "gain", `+${t.taken[pl.seat]}`));
      li.append(line);
    });
    log.append(li);
  });
  $("log-box").classList.toggle("hidden", !game.log.length);
}

// Player colours for charts: a validated categorical palette per theme, assigned by seat.
const SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"];
const SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"];
let SERIES = SERIES_DARK;

function lightTheme() {
  const t = document.documentElement.dataset.theme;
  return t ? t === "light" : matchMedia("(prefers-color-scheme: light)").matches;
}
const SVG_NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}

/** Cumulative points per seat after each trick, as a line chart with a hover crosshair. */
function renderChart() {
  SERIES = lightTheme() ? SERIES_LIGHT : SERIES_DARK;
  const box = $("chart");
  box.innerHTML = "";
  const n = game.players.length;
  const series = [...Array(n).keys()].map((seat) => {
    let total = 0;
    return [0, ...game.log.map((t) => (total += t.taken[seat]))];
  });
  // end-of-game bonus points (2 players) aren't in the trick log
  series.forEach((pts, seat) => { pts[pts.length - 1] = Math.max(pts[pts.length - 1], game.players[seat].points); });
  const W = 440, H = 190, L = 34, R = 78, T = 10, B = 24;
  const tricks = game.log.length;
  const maxY = Math.max(10, ...series.flat());
  const x = (i) => L + (i / Math.max(1, tricks)) * (W - L - R);
  const y = (v) => T + (1 - v / maxY) * (H - T - B);
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img",
    "aria-label": "Points over the game: " + series.map((p, s) => `${who(s)} ${p[p.length - 1]}`).join(", ") });
  // recessive grid: a few horizontal lines
  const step = maxY > 60 ? 20 : 10;
  for (let v = 0; v <= maxY; v += step) {
    chart.append(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid" }));
    const t = svg("text", { x: L - 6, y: y(v) + 4, class: "tick", "text-anchor": "end" });
    t.textContent = v;
    chart.append(t);
  }
  const xl = svg("text", { x: (L + W - R) / 2, y: H - 4, class: "tick", "text-anchor": "middle" });
  xl.textContent = "Trick";
  chart.append(xl);
  // lines, with direct labels at the end (nudged apart so they don't collide)
  const ends = series.map((p, seat) => ({ seat, y: y(p[p.length - 1]), v: p[p.length - 1] })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 13);
  series.forEach((pts, seat) => {
    chart.append(svg("polyline", { points: pts.map((v, i) => `${x(i)},${y(v)}`).join(" "), class: "series", stroke: SERIES[seat] }));
  });
  ends.forEach(({ seat, y: ly, v }) => {
    chart.append(svg("circle", { cx: x(tricks), cy: y(v), r: 4, fill: SERIES[seat], class: "end" }));
    const t = svg("text", { x: x(tricks) + 8, y: ly + 4, class: "end-label" });
    t.textContent = `${who(seat)} ${v}`;
    chart.append(t);
  });
  // hover: crosshair at the nearest trick, values in a tooltip
  const cross = svg("line", { y1: T, y2: H - B, class: "cross", visibility: "hidden" });
  chart.append(cross);
  const tip = el("div", "chart-tip hidden");
  chart.addEventListener("pointermove", (e) => {
    const r = chart.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(tricks, Math.round(((px - L) / (W - L - R)) * tricks)));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    tip.innerHTML = "";
    tip.append(el("div", "tip-title", i === 0 ? "Start" : `After trick ${i}`));
    [...series.keys()].sort((a, b) => series[b][i] - series[a][i]).forEach((seat) => {
      const row = el("div", "tip-row");
      const dot = el("span", "dot"); dot.style.background = SERIES[seat];
      row.append(dot, el("span", null, who(seat)), el("b", null, String(series[seat][i])));
      tip.append(row);
    });
    tip.classList.remove("hidden");
    tip.style.left = `${Math.min(r.width - 150, (x(i) / W) * r.width + 12)}px`;
  });
  chart.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.classList.add("hidden"); });
  // legend, always present for two or more series
  const legend = el("div", "legend");
  series.forEach((_, seat) => {
    const item = el("span", "legend-item");
    const dot = el("span", "dot"); dot.style.background = SERIES[seat];
    item.append(dot, document.createTextNode(who(seat)));
    legend.append(item);
  });
  box.append(legend, chart, tip);
}

function renderTable() {
  const n = game.players.length;
  const spectator = game.seat === null;
  const base = spectator ? 0 : game.seat;
  renderOpponents(spectator, base, n);
  renderMiddle();
  $("me").classList.toggle("hidden", spectator);
  if (!spectator) renderHand();
  renderSide();

  // game over
  $("game-over").classList.toggle("hidden", !game.over);
  if (game.over) {
    const winners = game.winners || [];
    $("over-title").textContent = winners.includes(game.seat)
      ? winners.length > 1 ? "You tied for the win!" : "You win!"
      : `${winners.map(seatName).join(" and ")} win${winners.length > 1 ? "" : "s"}`;
    const standings = $("standings");
    standings.innerHTML = "";
    [...game.players.keys()]
      .sort((a, b) => game.players[b].points - game.players[a].points)
      .forEach((seat) => {
        const li = el("li", winners.includes(seat) ? "win" : "");
        li.append(el("span", null, who(seat)), el("span", "points", `${game.players[seat].points} pts`));
        standings.append(li);
      });
    renderChart();
    $("next-game").classList.toggle("hidden", !me.host);
    $("over-hint").textContent = me.host ? "" : "Waiting for the host to start the next game.";
    // move focus into the dialog once, when the game ends
    if (!$("game-over").contains(document.activeElement)) (me.host ? $("next-game") : $("over-title")).focus();
  }
}

// ---------- views ----------
function render() {
  const inRoom = me.code && room && room.code === me.code;
  const view = !inRoom ? "home" : room.phase === "lobby" || !game ? "lobby" : "table";
  for (const v of ["home", "lobby", "table"]) $(v).classList.toggle("hidden", v !== view);
  $("room-badge").classList.toggle("hidden", !inRoom);
  if (inRoom) {
    $("room-badge").innerHTML = "";
    $("room-badge").append(el("span", "badge-label", "Room "), el("b", null, room.code));
    $("room-badge").setAttribute("aria-label", `Room ${room.code}`);
    if (room.spectators) $("room-badge").append(document.createTextNode(` · ${room.spectators} watching`));
  }
  if (view === "lobby") renderLobby();
  if (view === "table") keepingFocus(renderTable);
  // chat only matters with other people around
  const people = inRoom ? room.seats.filter((s) => s.kind === "human").length + room.spectators : 0;
  $("chat-box").classList.toggle("hidden", people < 2);
  const slot = $(view === "lobby" ? "lobby-chat-slot" : "side-chat-slot");
  if ($("chat-box").parentElement !== slot) slot.append($("chat-box"));
  updateTitle();
}

// Keyboard: 1-6 add a card of that rank to the selection, Enter plays it, Escape clears it.
document.addEventListener("keydown", (e) => {
  if (!game || game.seat === null || $("table").classList.contains("hidden")) return;
  const tag = document.activeElement.tagName;
  if (tag === "INPUT" || tag === "SELECT" || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "Enter" && tag !== "BUTTON") {
    const match = matchingLegal();
    if (match) { e.preventDefault(); play(match.action); }
  } else if (e.key === "h" || e.key === "H") {
    if (room && room.hints && game.current === game.seat) send({ t: "hint" });
  } else if (e.key === "Escape") {
    selection = [];
    renderHand();
  } else if (/^[1-6]$/.test(e.key)) {
    const rank = +e.key;
    const i = game.hand.findIndex((r, j) => r === rank && !selection.includes(j));
    if (i >= 0) { selection = [...selection, i]; renderHand(); }
  }
});

async function main() {
  $("name").value = localStorage.getItem("rainbow.name") || "";
  try {
    presets = await (await fetch("/api/agents")).json();
  } catch {
    toast("Couldn't load the agent list");
  }
  setupHome();
  const howto = () => {
    $("howto").showModal();
    $("howto").scrollTop = 0;
    $("howto-title").focus();
  };
  $("howto-btn").onclick = howto;
  $("howto-link").onclick = howto;
  $("hint-btn").onclick = () => send({ t: "hint" });
  $("sound-toggle").checked = soundOn;
  $("sound-toggle").onchange = () => {
    soundOn = $("sound-toggle").checked;
    try { localStorage.setItem("rainbow.sound", soundOn ? "on" : "off"); } catch { /* not remembered */ }
    SOUNDS.turn();
  };
  $("theme-select").value = document.documentElement.dataset.theme || "auto";
  $("theme-select").onchange = () => {
    const t = $("theme-select").value;
    if (t === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = t;
    try { localStorage.setItem("rainbow.theme", t); } catch { /* not remembered */ }
    if (game && game.over) renderChart(); // the chart palette depends on the theme
  };
  // close the settings menu when clicking elsewhere
  document.addEventListener("click", (e) => {
    if (!$("settings").contains(e.target)) $("settings").open = false;
  });
  $("chat-form").onsubmit = (e) => {
    e.preventDefault();
    const text = $("chat-input").value.trim();
    if (text) send({ t: "chat", text });
    $("chat-input").value = "";
  };
  $("lobby-start").onclick = () => send({ t: "start" });
  $("next-game").onclick = () => send({ t: "start" });
  $("lobby-leave").onclick = leave;
  $("over-leave").onclick = leave;
  $("copy-link").onclick = () => {
    navigator.clipboard.writeText($("share-link").value).then(() => toast("Link copied"), () => $("share-link").select());
  };
  const code = new URLSearchParams(location.search).get("room");
  if (code) {
    $("join-code").value = code.toUpperCase();
    // join straight away only if we know the player's name; otherwise let them enter it
    if (localStorage.getItem("rainbow.name")) {
      me.code = code.toUpperCase();
      me.token = localStorage.getItem(`rainbow.token.${me.code}`);
    } else {
      $("name").focus();
    }
  }
  connect();
  render();
}

main();
