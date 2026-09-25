/* Display aggregation only. Individual tickets and ledger summaries stay intact. */
(function (root) {
  "use strict";

  function selectionKey(s) {
    const legs = s.legs || [];
    if (!Array.isArray(legs) || legs.length !== 1 || (s.slip_type && s.slip_type !== "single")) return null;
    const l = legs[0];
    const sides = { total: ["over", "under"], team_total: ["over", "under"],
      spread: ["home", "away"], moneyline: ["home", "away"] };
    if (!l || !l.game_uid || !sides[l.market]?.includes(l.side)) return null;
    if (l.market !== "moneyline" && !Number.isFinite(l.line)) return null;
    // Missing period is ambiguous; it must not silently become a full-game bet.
    if (typeof l.period !== "string" || !l.period.trim()) return null;
    let team = null;
    if (l.market === "team_total") {
      team = l.selection_team_uid;
      if (!team) return null;
      const teams = [l.home_team_uid || l.home?.team_uid, l.away_team_uid || l.away?.team_uid].filter(Boolean);
      if (teams.length && !teams.includes(team)) return null;
    } else if (l.market === "spread" || l.market === "moneyline") {
      const expected = l[`${l.side}_team_uid`] || l[l.side]?.team_uid;
      if (l.selection_team_uid && expected && l.selection_team_uid !== expected) return null;
    }
    return JSON.stringify([l.game_uid, l.market, l.period.trim(), l.side, l.line ?? null, team]);
  }

  const isOpen = (s) => s.open ?? ((s.status || "pending") === "pending");
  const moneySum = (rows, key) => rows.every((s) => Number.isFinite(s[key]))
    ? Math.round(rows.reduce((n, s) => n + s[key], 0) * 100) / 100 : null;

  function averageOdds(rows) {
    // Average profit per dollar in decimal space, weighted by stake. American
    // odds cannot be averaged directly across the -100/+100 discontinuity.
    if (!rows.every((s) => Number.isFinite(s.stake) && s.stake >= 0 &&
        Number.isFinite(s.odds_american) && Math.abs(s.odds_american) >= 100)) return null;
    const stake = rows.reduce((n, s) => n + s.stake, 0);
    if (!stake) return null;
    const profit = rows.reduce((n, s) => n + s.stake * (s.odds_american > 0
      ? s.odds_american / 100 : 100 / -s.odds_american), 0) / stake;
    return Math.round(profit >= 1 ? profit * 100 : -100 / profit);
  }

  function combine(rows) {
    const first = rows[0];
    const books = new Map();
    for (const s of rows) {
      const book = (s.book || "").trim().toLowerCase();
      if (!books.has(book)) books.set(book, { book, display: s.book_display || s.book || "—" });
    }
    const base = { ...first, books: [...books.values()], slip_count: rows.length };
    if (rows.length === 1) return base;
    const statuses = new Map();
    for (const s of rows) {
      const status = s.status || "pending";
      statuses.set(status, (statuses.get(status) || 0) + 1);
    }
    const settled = rows.filter((s) => !isOpen(s));
    const free = rows.filter((s) => s.free_play);
    const modelCount = rows.filter((s) => s.model_qualified).length;
    return {
      ...base,
      stake: moneySum(rows, "stake"),
      to_win: moneySum(rows, "to_win"),
      odds_american: averageOdds(rows),
      status: statuses.size === 1 ? statuses.keys().next().value : "mixed",
      status_label: [...statuses].map(([status, n]) => `${n} ${status.replaceAll("_", " ")}`).join(" · "),
      open: rows.some(isOpen),
      pnl: settled.length ? moneySum(settled, "pnl") : null,
      pnl_as_cash: settled.length ? moneySum(settled, "pnl_as_cash") : null,
      partial_settlement: settled.length > 0 && settled.length < rows.length,
      free_play: free.length === rows.length,
      free_play_stake: free.length ? moneySum(free, "stake") : 0,
      stake_type: [...new Set(rows.map((s) => s.stake_type || "cash"))].join(" / "),
      model_qualified: modelCount === rows.length,
      model_count: modelCount,
      // A combined block must not inherit just one ticket's reference or source.
      external_ref: null,
      placed_at_et: null,
      source: [...new Set(rows.map((s) => s.source).filter(Boolean))].join(" / "),
    };
  }

  function groupSlips(slips) {
    const groups = new Map();
    for (const s of slips) {
      // Unrecognized contracts and multi-leg slips remain individual cards.
      const key = selectionKey(s) ?? Symbol();
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(s);
    }
    return [...groups.values()].map(combine);
  }

  if (typeof module === "object" && module.exports) module.exports = { groupSlips };
  else root.BjellacSlipGroups = { groupSlips };
})(globalThis);
