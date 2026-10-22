// End-to-end tests for the browser client, driven by Playwright against a running server.
//
//   server/e2e/run.sh            # builds and starts a fast-paced server, then runs this file
//
// Each check prints "ok" or throws; the process exits non-zero on the first failure.
const { chromium } = require("playwright");
const assert = require("assert");

const BASE = process.env.BASE_URL || "http://localhost:8099";
let axeSource = null;
try { axeSource = require("fs").readFileSync(require.resolve("axe-core/axe.min.js"), "utf8"); } catch { /* optional */ }

async function newPage(browser, viewport = { width: 1200, height: 900 }) {
  const page = await (await browser.newContext({ viewport })).newPage();
  page.errors = [];
  page.on("pageerror", (e) => page.errors.push(e.message));
  return page;
}

/** The cards of some legal play when it is this page's turn, or null. */
async function legalCards(page) {
  return page.evaluate(() => (game && game.legal.length ? game.legal[Math.floor(Math.random() * game.legal.length)].cards : null));
}

/** Plays legal moves until the game ends, alternating mouse (cards + Play) and keyboard. */
async function playUntilOver(page) {
  let moves = 0;
  for (let i = 0; i < 5000; i++) {
    if (await page.locator("#game-over:not(.hidden)").count()) return moves;
    const cards = await legalCards(page);
    if (cards) {
      try {
        await page.keyboard.press("Escape");
        if (moves % 2 === 0) {
          for (const r of cards) await page.keyboard.press(String(r));
          await page.keyboard.press("Enter");
        } else {
          const used = new Set();
          for (const r of cards) {
            const hand = await page.locator("#hand .card").allTextContents();
            const j = hand.findIndex((x, k) => x === String(r) && !used.has(k));
            used.add(j);
            await page.locator("#hand .card").nth(j).click({ timeout: 2000 });
          }
          await page.click("#play-selected", { timeout: 2000 });
        }
        moves++;
      } catch { /* the state changed under us; look again */ }
    }
    await page.waitForTimeout(30);
  }
  throw new Error("game did not end");
}

/** Creates a room; `seats` gives the other seats in order ("open" or an agent preset). */
async function createOnlineRoom(page, name, players, seats) {
  await page.goto(BASE);
  await page.fill("#name", name);
  await page.selectOption("#online-players", String(players));
  await page.click("#online-create");
  await page.waitForSelector("#lobby:not(.hidden)");
  for (const [i, agent] of seats.entries()) {
    if (agent === "open") continue;
    await page.selectOption(`#lobby-seats li:nth-child(${i + 2}) select`, agent);
    await page.waitForFunction(([k, id]) => room.seats[k].preset === id, [i + 1, agent]);
  }
  return (await page.textContent("#lobby-code")).trim();
}

async function joinByLink(page, code, name) {
  await page.goto(`${BASE}/?room=${code}`);
  await page.fill("#name", name);
  await page.click("#join-btn");
  await page.waitForSelector("#lobby:not(.hidden), #table:not(.hidden)");
}

async function axeScan(browser, colorScheme) {
  const p = await (await browser.newContext({ viewport: { width: 1200, height: 900 }, colorScheme })).newPage();
  const scan = async (label) => {
    await p.addScriptTag({ content: axeSource });
    const r = await p.evaluate(() => axe.run(document, { runOnly: ["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa", "best-practice"] }));
    assert.deepStrictEqual(r.violations.map((v) => `${colorScheme} ${label}: ${v.id}`), []);
  };
  await p.goto(BASE);
  await scan("home");
  await createOnlineRoom(p, "Ax", 3, ["open", "economist"]);
  await scan("lobby");
  await p.click("#lobby-leave");
  await p.click("#solo-start");
  await p.waitForFunction(() => game && game.legal.length > 0);
  await scan("table");
}

const tests = {
  async "solo game to the end, hint and remembered setup"(browser) {
    const p = await newPage(browser);
    await p.goto(BASE);
    await p.fill("#name", "Solo");
    await p.selectOption("#solo-players", "3");
    await p.selectOption("#solo-opponent", "economist");
    await p.click("#solo-custom summary");
    await p.locator("#solo-seats select").nth(1).selectOption("counter");
    await p.click("#solo-start");
    await p.waitForFunction(() => game && game.legal.length > 0);
    // double-tap selects every card of a rank
    const rank = await p.evaluate(() => game.hand[0]);
    await p.locator("#hand .card").first().dblclick();
    assert.strictEqual(await p.locator("#hand .card.sel").count(), await p.evaluate((r) => game.hand.filter((x) => x === r).length, rank));
    await p.keyboard.press("Escape");
    await p.keyboard.press("h");
    await p.waitForFunction(() => document.querySelectorAll("#hand .card.sel").length > 0);
    assert((await p.textContent("#play-selected")).startsWith("Play "), "hint names the play");
    await playUntilOver(p);
    assert(await p.locator("#chart svg").count(), "score chart shown");
    await p.click("#over-leave");
    await p.reload();
    assert.strictEqual(await p.locator("#solo-seats select").nth(1).inputValue(), "counter");
    assert.strictEqual(await p.locator("#solo-opponent").inputValue(), "economist");
    assert.deepStrictEqual(p.errors, []);
  },

  async "online room: join by link, reload mid-trick, chat, next game"(browser) {
    const a = await newPage(browser);
    const code = await createOnlineRoom(a, "Ann", 3, ["open", "economist"]);
    const b = await newPage(browser, { width: 900, height: 800 });
    await joinByLink(b, code, "Ben");
    await b.fill("#chat-input", "hi");
    await b.press("#chat-input", "Enter");
    await a.waitForFunction(() => document.querySelector("#chat-log").textContent.includes("hi"));
    assert.strictEqual(await a.evaluate(() => room.hints), false, "no hints with two people");
    await a.waitForFunction(() => !document.querySelector("#lobby-start").disabled);
    await a.click("#lobby-start");
    await b.waitForSelector("#table:not(.hidden)");
    // let a few plays happen so the trick isn't empty, then reload
    for (let i = 0; i < 3; i++) {
      for (const page of [a, b]) {
        const cards = await legalCards(page);
        if (cards) {
          for (const r of cards) await page.keyboard.press(String(r));
          await page.keyboard.press("Enter");
        }
      }
      await a.waitForTimeout(200);
    }
    await b.reload();
    await b.waitForSelector("#table:not(.hidden)", { timeout: 10000 });
    await Promise.all([playUntilOver(a), playUntilOver(b)]);
    await a.click("#next-game");
    await b.waitForFunction(() => document.querySelector("#game-over").classList.contains("hidden"));
    assert.deepStrictEqual([...a.errors, ...b.errors], []);
  },

  async "host hands an absent player's seat to an agent"(browser) {
    const a = await newPage(browser);
    const code = await createOnlineRoom(a, "Host", 2, ["open"]);
    const bctx = await browser.newContext();
    const b = await bctx.newPage();
    await joinByLink(b, code, "Guest");
    await a.waitForFunction(() => !document.querySelector("#lobby-start").disabled);
    await a.click("#lobby-start");
    await b.waitForSelector("#table:not(.hidden)");
    await bctx.close();
    await a.waitForSelector("#opponents select");
    await a.selectOption("#opponents select", "cheapest");
    await playUntilOver(a);
    assert.deepStrictEqual(a.errors, []);
  },

  async "keyboard only: start, select a card, keep focus, play it"(browser) {
    const p = await newPage(browser);
    await p.goto(BASE);
    let found = false;
    for (let i = 0; i < 25 && !found; i++) {
      await p.keyboard.press("Tab");
      found = await p.evaluate(() => document.activeElement.id === "solo-start");
    }
    assert(found, "Start reachable by Tab");
    await p.keyboard.press("Enter");
    await p.waitForFunction(() => game && game.legal.length > 0);
    let onCard = false;
    for (let i = 0; i < 80 && !onCard; i++) {
      await p.keyboard.press("Tab");
      onCard = await p.evaluate(() => document.activeElement.matches("#hand .card"));
    }
    assert(onCard, "a hand card is reachable by Tab");
    await p.keyboard.press("Enter"); // a single card is always a legal play
    assert(await p.evaluate(() => document.activeElement.matches("#hand .card.sel")), "focus kept after selecting");
    const before = await p.locator("#hand .card").count();
    let onPlay = false;
    for (let i = 0; i < 80 && !onPlay; i++) {
      await p.keyboard.press("Tab");
      onPlay = await p.evaluate(() => document.activeElement.id === "play-selected");
    }
    await p.keyboard.press("Enter");
    await p.waitForFunction((n) => document.querySelectorAll("#hand .card").length < n, before);
  },

  async "no axe-core violations on home, lobby and table, in both themes"(browser) {
    if (!axeSource) return "skipped (npm install axe-core)";
    for (const colorScheme of ["light", "dark"]) await axeScan(browser, colorScheme);
  },

  async "phone width: no horizontal scrolling"(browser) {
    const p = await newPage(browser, { width: 360, height: 780 });
    await p.goto(BASE);
    await p.selectOption("#solo-players", "6");
    await p.click("#solo-start");
    await p.waitForFunction(() => game && game.legal.length > 0);
    assert(await p.evaluate(() => document.documentElement.scrollWidth <= 360));
  },
};

(async () => {
  const browser = await chromium.launch();
  let failed = 0;
  for (const [name, test] of Object.entries(tests)) {
    try {
      const note = await test(browser);
      console.log(`ok   ${name}${note ? ` (${note})` : ""}`);
    } catch (e) {
      failed++;
      console.log(`FAIL ${name}\n     ${String(e.message).split("\n")[0]}`);
    }
  }
  await browser.close();
  process.exit(failed ? 1 : 0);
})();
