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

/** A brief message: red for errors (announced at once); `info` ones are neutral and polite. */
function toast(msg, kind = "error") {
  const t = $("toast");
  t.textContent = msg.charAt(0).toUpperCase() + msg.slice(1);
  t.classList.toggle("info", kind === "info");
  t.setAttribute("role", kind === "info" ? "status" : "alert");
  t.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.add("hidden"), 3500);
}

// Buttons waiting on the server (start, create, join): disabled and relabelled until the
// server answers, so a slow reply shows and a double click doesn't make two rooms.
const pendingButtons = new Map();
function setPending(btn, label) {
  if (pendingButtons.has(btn)) return false;
  pendingButtons.set(btn, btn.textContent);
  btn.disabled = true;
  btn.textContent = label;
  return true;
}
function clearPending() {
  for (const [btn, label] of pendingButtons) {
    btn.disabled = false;
    btn.textContent = label;
  }
  pendingButtons.clear();
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
  win: () => tones([[523, 0], [659, 0.1], [784, 0.2], [1047, 0.3]], 0.3, 0.05),
  end: () => tones([[587, 0], [440, 0.16]], 0.3, 0.04),
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
      clearPending();
      me.code = msg.code;
      me.seat = msg.seat;
      me.token = msg.token || null;
      me.host = msg.host;
      if (me.token) localStorage.setItem(`rainbow.token.${me.code}`, me.token);
      if (computerSpeed() !== "normal") send({ t: "speed", pace: PACES[computerSpeed()] }); // solo games only; the server checks
      history.replaceState(null, "", `/?room=${me.code}`);
      break;
    case "room":
      room = msg;
      // hosting passes to another player while the room's creator is away
      if (me.seat !== null && room.seats[me.seat]) me.host = !!room.seats[me.seat].host;
      if (room.phase === "lobby") game = null;
      if (autoStart && me.host && room.phase === "lobby" && !room.seats.some((s) => s.kind === "open")) {
        autoStart = false;
        send({ t: "start" });
      }
      break;
    case "state": {
      const old = game;
      if (msg.over && old && !old.over) {
        game = msg;
        recordResult();
      }
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
      hintPending(false);
      const l = game && game.legal.find((x) => x.action === msg.action);
      if (l) {
        selectCombo(l);
        renderHand();
      }
      return;
    }
    case "error":
      clearPending();
      hintPending(false);
      toast(sentence(msg.msg));
      if (msg.msg === "that room doesn't exist or has closed" || msg.msg === "the room has closed") leave();
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
  renderRecord();
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ t: "leave" }));
  render();
}

// ---------- home ----------
function presetOptions(select, withOpen, value) {
  select.innerHTML = "";
  if (withOpen) select.append(new Option("Open for a friend", "open"));
  // grouped by difficulty, easiest first (the server sends them in that order)
  const groups = new Map();
  presets.forEach((p) => {
    if (!groups.has(p.level)) {
      const g = document.createElement("optgroup");
      g.label = p.level;
      groups.set(p.level, g);
      select.append(g);
    }
    const o = new Option(`${p.label} (${p.rating.toFixed(2)})`, p.id);
    o.title = p.description;
    groups.get(p.level).append(o);
  });
  if (value) select.value = value;
}

/** The Watch a Game dialog: the player count and an agent for every seat. */
function openWatch() {
  const select = $("watch-players");
  if (!select.options.length) for (let n = 2; n <= 6; n++) select.append(new Option(`${n} players`, n));
  let saved = [];
  try { saved = JSON.parse(localStorage.getItem("rainbow.watch")) || []; } catch { /* none */ }
  select.value = String(saved.length >= 2 ? saved.length : 4);
  watchSeats(+select.value, saved);
  $("watch-dialog").showModal();
  $("watch-title").focus();
}

/** One agent menu per seat; by default a varied table, from rule-based players to search. */
function watchSeats(count, saved = []) {
  const box = $("watch-seats");
  const old = [...box.querySelectorAll("select")].map((x) => x.value);
  const variety = ["economist", "counter", "dmc", "ismcts", "ppo", "guided"].filter((id) => presets.some((p) => p.id === id));
  box.innerHTML = "";
  for (let i = 0; i < count; i++) {
    const label = el("label", null, `Seat ${i + 1}`);
    const select = el("select");
    presetOptions(select, false, old[i] || saved[i] || variety[i % variety.length]);
    if (!select.value && presets[0]) select.value = presets[0].id;
    label.append(select);
    box.append(label);
  }
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

// ---------- personal record against the computer (kept on this device) ----------
function loadRecord() {
  try { return JSON.parse(localStorage.getItem("rainbow.record")) || {}; } catch { return {}; }
}

/** Records a finished solo game against a single kind of opponent. */
function recordResult() {
  const opponent = soloOpponent();
  if (!opponent) return;
  const record = loadRecord();
  const r = record[opponent] || { games: 0, wins: 0 };
  r.games += 1;
  if ((game.winners || []).includes(game.seat)) r.wins += 1;
  record[opponent] = r;
  try { localStorage.setItem("rainbow.record", JSON.stringify(record)); } catch { /* not kept */ }
}

/** "Your record against X: ..." for a preset id, or "" without a record. */
function recordText(id) {
  const r = loadRecord()[id];
  const p = presets.find((x) => x.id === id);
  if (!r || !p) return "";
  return `Your record against ${p.label}: ${r.wins} win${r.wins === 1 ? "" : "s"} in ${r.games} game${r.games === 1 ? "" : "s"}`;
}

/** Under the opponent menu: what the opponent is, and the player's record against it. */
function renderRecord() {
  const id = $("solo-opponent").value;
  const p = presets.find((x) => x.id === id);
  const text = recordText(id);
  const box = $("solo-record");
  box.innerHTML = "";
  if (p) box.append(el("span", null, `${p.level} · rating ${p.rating.toFixed(2)}. ${p.description}`));
  if (text) box.append(el("span", null, `${text}.`));
  box.classList.toggle("hidden", !box.childElementCount);
}

/** The next stronger opponent, once the player wins at least half their games (2 or more)
 * against `id`; presets come ordered by strength. */
function nextChallenge(id) {
  const r = loadRecord()[id];
  const i = presets.findIndex((p) => p.id === id);
  if (!r || i < 0 || i + 1 >= presets.length || r.wins < 2 || r.wins * 2 < r.games) return null;
  return presets[i + 1];
}

/** The single kind of computer opponent in this solo game, or null. */
function soloOpponent() {
  if (!room || !room.hints || game.seat === null) return null;
  const opponents = room.seats.filter((s, i) => i !== game.seat).map((s) => s.preset);
  return opponents.length && opponents.every((p) => p === opponents[0]) ? opponents[0] : null;
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
  // a first game against a Casual opponent; later visits remember the player's choice
  const defaultAgent = presets.some((p) => p.id === "economist") ? "economist" : presets[0] && presets[0].id;
  presetOptions($("solo-opponent"), false, saved.opponent || defaultAgent);
  if (!$("solo-opponent").value) $("solo-opponent").value = defaultAgent;
  $("solo-custom").open = !!saved.custom;
  const refresh = () => {
    seatPickers($("solo-seats"), +$("solo-players").value, $("solo-opponent").value, saved.soloSeats);
    saveSetup();
  };
  $("solo-players").onchange = refresh;
  $("solo-opponent").addEventListener("change", renderRecord);
  $("solo-opponent").onchange = () => {
    // a new opponent choice applies to every seat
    $("solo-seats").querySelectorAll("select").forEach((s) => { s.value = $("solo-opponent").value; });
    saveSetup();
  };
  $("solo-custom").ontoggle = saveSetup;
  renderRecord();
  $("online-players").onchange = saveSetup;
  refresh();

  $("solo-start").onclick = () => {
    if (!setPending($("solo-start"), "Starting…")) return;
    autoStart = true;
    const n = +$("solo-players").value;
    const seats = $("solo-custom").open
      ? [...$("solo-seats").querySelectorAll("select")].map((s) => s.value)
      : Array(n - 1).fill($("solo-opponent").value);
    send({ t: "create", name: playerName(), seats });
  };
  // online rooms start with every other seat open; the host can fill seats in the lobby
  $("online-create").onclick = () => {
    if (!setPending($("online-create"), "Creating…")) return;
    send({ t: "create", name: playerName(), seats: Array(+$("online-players").value - 1).fill("open") });
  };
  const join = () => {
    // accept a pasted invite link as well as a bare code
    const raw = $("join-code").value.trim();
    const fromLink = raw.match(/[?&]room=([a-z]{4})/i);
    const code = (fromLink ? fromLink[1] : raw).toUpperCase();
    if (code.length !== 4) return toast("Room codes have four letters");
    const btn = $("home").classList.contains("invited") ? $("invite-join") : $("join-btn");
    if (!setPending(btn, "Joining…")) return;
    me.code = code;
    me.token = localStorage.getItem(`rainbow.token.${code}`);
    send({ t: "join", code, name: playerName(), token: me.token });
  };
  $("join-btn").onclick = join;
  $("join-code").onkeydown = (e) => { if (e.key === "Enter") join(); };
  // an invite link opens a focused view: a name and one button
  $("invite-join").onclick = join;
  $("name").onkeydown = (e) => { if (e.key === "Enter" && $("home").classList.contains("invited")) join(); };
  $("invite-dismiss").onclick = () => {
    $("home").classList.remove("invited");
    $("invite").classList.add("hidden");
    history.replaceState(null, "", "/");
  };
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
    const editable = me.host && i !== me.seat && !(s.kind === "human" && s.connected);
    if (editable) {
      // the host's dropdown says who sits here; no separate label needed
      const select = el("select");
      presetOptions(select, true, s.kind === "agent" ? s.preset : "open");
      if (s.kind === "human") select.prepend(new Option(`${s.name} (away)`, ""));
      select.value = s.kind === "agent" ? s.preset : s.kind === "human" ? "" : "open";
      select.setAttribute("aria-label", `Player for seat ${i + 1}`);
      select.onchange = () => select.value && send({ t: "seat", seat: i, agent: select.value === "open" ? null : select.value });
      line.append(select);
    } else if (s.kind === "human") {
      who.append(document.createTextNode(s.name));
      if (i === me.seat) who.append(el("span", "tag you", "you"));
      if (s.host) who.append(el("span", "tag", "host"));
      if (!s.connected) who.append(el("span", "tag off", "away"));
    } else if (s.kind === "agent") {
      who.append(document.createTextNode(s.name), el("span", "tag", "computer"));
    } else {
      who.append(el("span", "hint", "Waiting for a player…"));
    }
    if (!editable) line.append(who);
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

/** A play as spoken: "a 4", "three 5s", "a run 2 to 4". */
function cardsSpoken(cards) {
  const words = ["", "a", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
  if (cards.length === 1) return `a ${cards[0]}`;
  if (cards.every((r) => r === cards[0])) return `${words[cards.length] || cards.length} ${cards[0]}s`;
  return `a run ${cards[0]} to ${cards[cards.length - 1]}`;
}

/** Screen-reader announcements for what changed between two states. */
function announceChanges(old, next) {
  const live = $("live");
  const said = [];
  const oldLog = old ? old.log.length : 0;
  const oldTrick = old && old.log.length === next.log.length ? old.trick.length : 0;
  if (next.log.length > oldLog) {
    const done = next.log[next.log.length - 1];
    // plays of the finished trick not yet announced
    done.plays.slice(oldLog === next.log.length - 1 && old ? old.trick.length : 0)
      .forEach((pl) => said.push(`${who(pl.seat)} played ${cardsSpoken(pl.cards)}.`));
    const gains = [...new Set(done.plays.map((pl) => pl.seat))]
      .map((seat) => `${who(seat)} took ${done.taken[seat]}`);
    said.push(`Trick over. ${gains.join(", ")}.`);
  }
  next.trick.slice(oldTrick).forEach((pl) => said.push(`${who(pl.seat)} played ${cardsSpoken(pl.cards)}.`));
  if (!next.over && next.current === next.seat && (!old || old.current !== next.seat)) {
    said.push(next.trick_type !== "open" ? `Your turn. Only ${TRICK_TYPES[next.trick_type]} now.` : "Your turn.");
  }
  if (next.over && (!old || !old.over)) said.push("Game over.");
  if (said.length) live.textContent = said.join(" ");
  if (next.log.length > oldLog && old) {
    SOUNDS.trick();
    gainFlash = { at: Date.now(), taken: next.log[next.log.length - 1].taken };
  }
  if (!next.over && next.current === next.seat && (!old || old.current !== next.seat)) SOUNDS.turn();
  // the game's end, after the last trick's sound
  if (next.over && old && !old.over) {
    setTimeout((next.winners || []).includes(next.seat) ? SOUNDS.win : SOUNDS.end, 450);
  }
  freshFrom = oldTrick;
}

function matchingLegal() {
  if (!game || !game.legal.length || !selection.length) return null;
  const ranks = selection.map((i) => game.hand[i]).sort((a, b) => a - b).join();
  return game.legal.find((l) => l.cards.join() === ranks) || null;
}

/** For plays in order, the pile each would take if the trick ended now: plays rank by card
 * count, then top card, then order. */
function projectPiles(plays) {
  const order = plays.map((_, i) => i).sort((a, b) =>
    plays[b].length - plays[a].length || Math.max(...plays[b]) - Math.max(...plays[a]) || a - b);
  const piles = [];
  order.forEach((i, rank) => { piles[i] = game.piles[rank]; });
  return piles;
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

// Suggestions come from the strongest agent and take a moment; the button says so.
function hintPending(on) {
  $("hint-btn").disabled = on;
  $("hint-btn").textContent = on ? "Thinking…" : "Suggest";
}
/** Colours the browser's toolbar (on phones) like the page, following a chosen theme. */
function syncThemeColor() {
  const bg = getComputedStyle(document.body).backgroundColor;
  document.querySelectorAll('meta[name="theme-color"]').forEach((m) => {
    if (document.documentElement.dataset.theme) m.setAttribute("content", bg);
    else m.setAttribute("content", m.media.includes("dark") ? "#12161f" : "#f4f1ea");
  });
}

/** How quickly the computer moves in solo games: slow, normal or fast (the pause factor). */
const PACES = { slow: 1.8, normal: 1, fast: 0.35 };
function computerSpeed() {
  try {
    const s = localStorage.getItem("rainbow.speed");
    if (s in PACES) return s;
    return localStorage.getItem("rainbow.fast") === "on" ? "fast" : "normal"; // the old setting
  } catch { return "normal"; }
}

/** A server message as a sentence: capitalised, with a full stop. */
function sentence(text) {
  const t = text.charAt(0).toUpperCase() + text.slice(1);
  return /[.!?]$/.test(t) ? t : `${t}.`;
}

function askHint() {
  if ($("hint-btn").disabled) return;
  hintPending(true);
  send({ t: "hint" });
}

// A one-time tip for a first game, until the first play or a dismissal.
function tipSeen() {
  try { return localStorage.getItem("rainbow.tip") === "seen"; } catch { return true; }
}
function markTipSeen() {
  try { localStorage.setItem("rainbow.tip", "seen"); } catch { /* shown again next time */ }
  $("first-tip").classList.add("hidden");
}

// Where the cards just played sat in the hand, so they can glide from there into the trick.
let flight = null;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

function play(action) {
  markTipSeen();
  flight = {
    at: Date.now(),
    rects: [...document.querySelectorAll("#hand .card.sel")].map((c) => c.getBoundingClientRect()),
  };
  selection = [];
  send({ t: "play", action });
}

/** FLIP: start the new play's cards where they were in the hand and let them slide into place. */
function flyIn(row) {
  const reduce = reducedMotion.matches || document.documentElement.dataset.motion === "reduce";
  if (!flight || reduce || Date.now() - flight.at > 1500) return;
  const cards = [...row.children];
  cards.forEach((c, i) => {
    const from = flight.rects[Math.min(i, flight.rects.length - 1)];
    if (!from) return;
    const to = c.getBoundingClientRect();
    const dx = from.left + from.width / 2 - (to.left + to.width / 2);
    const dy = from.top + from.height / 2 - (to.top + to.height / 2);
    c.animate(
      [{ transform: `translate(${dx}px, ${dy}px) scale(${from.width / to.width})` }, { transform: "none" }],
      { duration: 260, easing: "cubic-bezier(.2,.8,.2,1)" },
    );
  });
  flight = null;
}

// When the current turn started, to show how long a person has been thinking.
let turnStart = { seat: null, at: 0 };
const SHOW_WAIT_AFTER = 15000;

function waitText(seat) {
  if (game.over || game.current !== seat) return "";
  const s = room.seats[seat] || {};
  if (s.kind !== "human") return "";
  const ms = Date.now() - turnStart.at;
  // the server plays an away player's turn after AWAY_GRACE (30 s) while others wait
  if (s.connected === false) {
    const left = Math.max(0, 30 - Math.floor(ms / 1000));
    return left > 0 ? `computer plays in ${left}s` : "computer playing…";
  }
  if (ms < SHOW_WAIT_AFTER) return "";
  const secs = Math.floor(ms / 1000);
  return `thinking… ${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
}

// refresh the waiting time once a second without re-rendering the table
setInterval(() => {
  document.querySelectorAll("[data-wait-seat]").forEach((e) => {
    e.textContent = waitText(+e.dataset.waitSeat);
  });
}, 1000);

function renderOpponents(spectator, base, n) {
  if (turnStart.seat !== game.current) turnStart = { seat: game.current, at: Date.now() };
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
    // a watched game shows every hand, to learn from
    let shown = null;
    if (p.hand) {
      shown = el("div", "open-hand");
      shown.setAttribute("aria-label", `Hand: ${p.hand.join(" ") || "empty"}`);
      for (const r of p.hand) shown.append(el("span", `card tiny r${r}`, String(r)));
    }
    const wait = el("span", "wait", waitText(seat));
    wait.dataset.waitSeat = seat;
    head.append(wait);
    box.append(head, meta);
    if (shown) box.append(shown);
    if (me.host && s.kind === "human" && !s.connected && !game.over) {
      const select = el("select");
      select.setAttribute("aria-label", `Give ${seatName(seat)}'s seat to the computer`);
      presetOptions(select, false);
      select.prepend(new Option("Hand seat to the computer…", ""));
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
  // in a finished trick, the play that took the most is the one that won it
  const taken = !current && game.last_trick.taken;
  const top = taken ? Math.max(...plays.map((pl) => taken[pl.seat])) : -1;
  // While the trick is open, the pile each play would take if it ended now: plays rank by
  // card count, then top card, then order.
  const projected = new Map();
  if (current && plays.length > 1) {
    projectPiles(plays.map((pl) => pl.cards)).forEach((pile, i) => { if (pile !== undefined) projected.set(i, pile); });
  }
  plays.forEach((pl, i) => {
    const lastOfSeat = !plays.slice(i + 1).some((q) => q.seat === pl.seat);
    const leading = current ? i === game.trick_leader : lastOfSeat && taken[pl.seat] === top;
    const box = el("div", "play" + (leading ? " leading" : "") + (current && i >= freshFrom ? " fresh" : ""));
    const by = el("div", "by", who(pl.seat));
    if (projected.has(i)) {
      const tag = el("span", "lead-tag", `→ ${projected.get(i)}`);
      tag.title = "The pile this play would take if the trick ended now";
      by.append(tag);
    }
    if (!current && lastOfSeat) by.append(el("span", "gain", `+${game.last_trick.taken[pl.seat]}`));
    const row = cardRow(pl.cards, true);
    box.append(by, row);
    trick.append(box);
  });
  if (!plays.length) trick.append(el("span", "hint", game.current === game.seat ? "Your lead" : `${who(game.current)} to lead`));
  // my play just landed: animate it from the hand (it is the last play of mine in this trick)
  const mine = current ? plays.map((pl, i) => [pl, i]).filter(([pl]) => pl.seat === game.seat).pop() : null;
  if (flight && mine && mine[1] >= freshFrom) {
    // no drop-in for this one; it flies in from the hand instead
    const box = trick.children[mine[1]];
    box.classList.remove("fresh");
    requestAnimationFrame(() => flyIn(box.querySelector(".cards")));
  }
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
    // double-tap: every card of this rank, the quick way to pick a set
    c.ondblclick = () => {
      selection = game.hand.flatMap((r, j) => (r === rank ? [j] : []));
      renderHand();
    };
    group.append(c);
  });
  if (!game.hand.length) hand.append(el("span", "hint", "No cards left"));

  const match = matchingLegal();
  const btn = $("play-selected");
  btn.disabled = !match;
  // with the pile it would take if the trick ended now, as in the trick area
  const would = match && game.trick && projectPiles([...game.trick.map((pl) => pl.cards), match.cards]).at(-1);
  btn.textContent = match ? `Play ${cardsText(match.cards)}${would !== undefined && would !== null ? ` → ${would}` : ""}`
    : myTurn && selection.length ? whyNot()
    : myTurn ? "Select cards" : "Play";
  btn.classList.toggle("reason", !match && myTurn && selection.length > 0);
  if (match && would !== undefined && would !== null) {
    btn.setAttribute("aria-label", `Play ${cardsSpoken(match.cards)}, which would take ${would} so far`);
  } else {
    btn.removeAttribute("aria-label");
  }
  btn.onclick = () => match && play(match.action);
  $("hint-btn").classList.toggle("hidden", !(room && room.hints && myTurn));
  $("first-tip").classList.toggle("hidden", !myTurn || tipSeen());
}

let showAllTricks = false;

function renderSide() {
  // cards I haven't seen, per rank
  const unseen = $("unseen");
  unseen.innerHTML = "";
  $("unseen-box").classList.toggle("hidden", !game.unseen);
  (game.unseen || []).forEach((count, i) => {
    const cell = el("div", "unseen-cell" + (count === 0 ? " gone" : ""));
    cell.append(card(i + 1, true), el("span", null, `×${count}`));
    cell.setAttribute("aria-label", `${count} unseen ${i + 1}s`);
    unseen.append(cell);
  });

  // trick log, newest first; the latest few unless the player asks for all
  const log = $("log");
  log.innerHTML = "";
  const shown = showAllTricks ? game.log.length : Math.min(3, game.log.length);
  game.log.slice().reverse().slice(0, shown).forEach((t, k) => {
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
  if (game.log.length > 3) {
    const more = el("button", "link", showAllTricks ? "Show fewer" : `Show all ${game.log.length} tricks`);
    more.onclick = () => { showAllTricks = !showAllTricks; renderSide(); };
    const li = el("li");
    li.append(more);
    log.append(li);
  }
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
  // the right margin fits the longest end label ("Economist 2 43"; about 6 units a character)
  const longest = Math.max(...series.map((p, seat) => `${who(seat)} ${p[p.length - 1]}`.length));
  const W = 440, H = 190, L = 34, R = Math.max(60, 16 + 6 * longest), T = 10, B = 24;
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
  $("watching").classList.toggle("hidden", !spectator);
  $("watching").textContent = room.watch
    ? "You're watching the computer play."
    : "You're watching this game; it's full.";
  const controls = room.watch && !game.over;
  $("watch-controls").classList.toggle("hidden", !controls);
  if (controls) {
    $("pause-btn").textContent = room.paused ? "Resume" : "Pause";
    $("step-btn").disabled = !room.paused;
    $("watch-speed").value = computerSpeed();
  }
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
    $("next-game").classList.toggle("hidden", !me.host && !room.watch);
    $("next-game").textContent = room.watch ? "Watch another" : "Play again";
    // against the computer there's no room to leave, just a setup to change
    $("over-leave").textContent = room.watch ? "Done" : room.hints ? "Change opponents" : "Leave room";
    $("over-hint").textContent = me.host
      ? (soloOpponent() ? recordText(soloOpponent()) : "")
      : room.watch ? "" : "Waiting for the host to start the next game.";
    const up = me.host && winners.includes(game.seat) && soloOpponent() && nextChallenge(soloOpponent());
    if (up) {
      const go = el("button", "link", `Try ${up.label}`);
      go.onclick = () => {
        const n = game.players.length;
        leave();
        $("solo-players").value = String(n);
        $("solo-custom").open = false;
        $("solo-opponent").value = up.id;
        $("solo-opponent").dispatchEvent(new Event("change"));
        $("solo-start").click();
      };
      $("over-hint").append(". ", go);
    }
    // move focus into the dialog once, when the game ends
    if (!$("game-over").contains(document.activeElement)) (me.host || room.watch ? $("next-game") : $("over-title")).focus();
  }
}

// ---------- views ----------
function render() {
  const inRoom = me.code && room && room.code === me.code;
  const view = !inRoom ? "home" : room.phase === "lobby" || !game ? "lobby" : "table";
  for (const v of ["home", "lobby", "table"]) $(v).classList.toggle("hidden", v !== view);
  // the lobby shows the code in its heading; the badge is for the table
  // a solo game's room code is no use to anyone, so only online tables show it
  const solo = inRoom && (room.watch
    || (room.hints && room.seats.filter((x) => x.kind === "human").length === 1 && !room.spectators));
  $("room-badge").classList.toggle("hidden", !inRoom || view === "lobby" || solo);
  if (inRoom) {
    $("room-badge").innerHTML = "";
    $("room-badge").append(el("span", "badge-label", "Room "), el("b", null, room.code));
    $("room-badge").setAttribute("aria-label", `Room ${room.code}`);
    if (room.spectators) $("room-badge").append(document.createTextNode(` · ${room.spectators} watching`));
  }
  $("leave-btn").classList.toggle("hidden", view !== "table");
  if (view === "lobby") renderLobby();
  if (view === "table") keepingFocus(renderTable);
  // chat only matters with other people around
  const people = inRoom ? room.seats.filter((s) => s.kind === "human").length + room.spectators : 0;
  $("chat-box").classList.toggle("hidden", people < 2);
  const slot = $(view === "lobby" ? "lobby-chat-slot" : "side-chat-slot");
  if ($("chat-box").parentElement !== slot) slot.append($("chat-box"));
  updateTitle();
}

// "?" opens How to play from anywhere (outside text fields), as on many sites.
document.addEventListener("keydown", (e) => {
  const tag = document.activeElement.tagName;
  if (e.key === "?" && tag !== "INPUT" && tag !== "TEXTAREA" && !$("howto").open) {
    e.preventDefault();
    $("howto-btn").click();
  }
});

// Watching: Space pauses or resumes, the right arrow plays the next move while paused.
document.addEventListener("keydown", (e) => {
  if (!game || !room || !room.watch || game.over || $("table").classList.contains("hidden")) return;
  const tag = document.activeElement.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "BUTTON" || tag === "TEXTAREA") return;
  if (e.ctrlKey || e.metaKey || e.altKey || document.querySelector("dialog[open]")) return;
  if (e.key === " ") { e.preventDefault(); togglePause(); }
  else if (e.key === "ArrowRight" && room.paused) { e.preventDefault(); send({ t: "step" }); }
});

function togglePause() {
  send({ t: "pause", paused: !room.paused });
}

function setSpeed(speed) {
  try { localStorage.setItem("rainbow.speed", speed); } catch { /* not remembered */ }
  $("speed-select").value = speed;
  if (me.code) send({ t: "speed", pace: PACES[speed] });
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
    if (room && room.hints && game.current === game.seat) askHint();
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
    toast("Couldn't load the opponents. Reload the page to try again.");
  }
  setupHome();
  const howto = () => {
    $("watch-btn").classList.toggle("hidden", !!me.code);
    $("howto").showModal();
    $("howto").scrollTop = 0;
    $("howto-title").focus();
  };
  $("howto-btn").onclick = howto;
  $("rating-info").onclick = () => {
    const open = $("rating-help").hidden;
    $("rating-help").hidden = !open;
    $("rating-info").setAttribute("aria-expanded", String(open));
  };
  // a game played by the computer alone, at a table the viewer chooses
  $("watch-btn").onclick = () => { $("howto").close(); openWatch(); };
  $("watch-link").onclick = openWatch;
  $("watch-players").onchange = () => watchSeats(+$("watch-players").value);
  $("watch-start").onclick = () => {
    const seats = [...$("watch-seats").querySelectorAll("select")].map((x) => x.value);
    try { localStorage.setItem("rainbow.watch", JSON.stringify(seats)); } catch { /* not remembered */ }
    $("watch-dialog").close();
    send({ t: "create", name: playerName(), seats, watch: true });
  };
  $("hint-btn").onclick = askHint;
  $("first-tip-close").onclick = markTipSeen;
  $("sound-toggle").checked = soundOn;
  $("sound-toggle").onchange = () => {
    soundOn = $("sound-toggle").checked;
    try { localStorage.setItem("rainbow.sound", soundOn ? "on" : "off"); } catch { /* not remembered */ }
    SOUNDS.turn();
  };
  $("pause-btn").onclick = togglePause;
  $("step-btn").onclick = () => send({ t: "step" });
  $("watch-speed").onchange = () => setSpeed($("watch-speed").value);
  $("speed-select").value = computerSpeed();
  $("speed-select").onchange = () => {
    const speed = $("speed-select").value;
    setSpeed(speed);
  };
  $("motion-toggle").checked = document.documentElement.dataset.motion === "reduce";
  $("motion-toggle").onchange = () => {
    const reduce = $("motion-toggle").checked;
    if (reduce) document.documentElement.dataset.motion = "reduce";
    else delete document.documentElement.dataset.motion;
    try { localStorage.setItem("rainbow.motion", reduce ? "reduce" : "full"); } catch { /* not remembered */ }
  };
  $("theme-select").value = document.documentElement.dataset.theme || "auto";
  $("theme-select").onchange = () => {
    const t = $("theme-select").value;
    if (t === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = t;
    try { localStorage.setItem("rainbow.theme", t); } catch { /* not remembered */ }
    if (game && game.over) renderChart(); // the chart palette depends on the theme
    syncThemeColor();
  };
  syncThemeColor();
  // close the settings menu when clicking elsewhere
  document.addEventListener("click", (e) => {
    if (!$("settings").contains(e.target)) $("settings").open = false;
  });
  // and on Escape, returning focus to its button (before Escape clears the card selection)
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !$("settings").open) return;
    $("settings").open = false;
    $("settings").querySelector("summary").focus();
    e.stopImmediatePropagation();
  }, true);
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
  // leaving a solo game ends it; an online seat can be reclaimed with the room's link
  $("leave-btn").onclick = () => {
    const solo = room && room.hints && room.seats.filter((x) => x.kind === "human").length === 1;
    if (solo && game && !game.over && !confirm("Leave this game? It will end.")) return;
    leave();
  };
  // phones get the system share sheet; elsewhere the link is copied
  if (navigator.share) $("copy-link").textContent = "Share link";
  $("copy-link").onclick = () => {
    const url = $("share-link").value;
    if (navigator.share) {
      navigator.share({ title: "Play Rainbow with me", url }).catch(() => { /* dismissed */ });
    } else {
      navigator.clipboard.writeText(url).then(() => toast("Link copied", "info"), () => $("share-link").select());
    }
  };
  const code = new URLSearchParams(location.search).get("room");
  if (code) {
    $("join-code").value = code.toUpperCase();
    // join straight away only if we know the player's name; otherwise let them enter it
    if (localStorage.getItem("rainbow.name")) {
      me.code = code.toUpperCase();
      me.token = localStorage.getItem(`rainbow.token.${me.code}`);
    } else {
      $("home").classList.add("invited");
      $("invite").classList.remove("hidden");
      $("invite-code").textContent = code.toUpperCase();
      $("name").focus();
    }
  }
  connect();
  render();
}

main();
