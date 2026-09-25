// All ticket records, events, and monetary values below are invented test inputs.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { groupSlips } = require('../dashboard/static/js/slip_groups.js');

function slip(overrides = {}, leg = {}) {
  return {
    slip_type: 'single', book: 'polymarket', book_display: 'Polymarket',
    stake: 10, to_win: 9, odds_american: -110, status: 'pending', open: true,
    stake_type: 'cash', free_play: false, model_qualified: true,
    pnl: null, pnl_as_cash: null, source: 'sync',
    legs: [{ game_uid: 'example-north-example-south', market: 'total', period: 'full_game',
      side: 'over', line: 60.5, ...leg }], ...overrides,
  };
}

test('fictional matching lines combine while different lines stay separate', () => {
  const input = [slip({ stake: 1.25, to_win: 1.00, odds_american: -125 }),
    slip({ stake: 3.75, to_win: 3.00, odds_american: -125 }),
    slip({ stake: 8.00, to_win: 4.00, odds_american: -200 }, { line: 55.5 })];
  const original = structuredClone(input);
  const rows = groupSlips(input);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].stake, 5.00);
  assert.equal(rows[0].to_win, 4.00);
  assert.equal(rows[0].odds_american, -125);
  assert.equal(rows[0].slip_count, 2);
  assert.equal(rows[1].stake, 8.00);
  assert.deepEqual(input, original, 'raw tickets must not be mutated');
});

test('all contributing books appear once, including inconsistent casing', () => {
  const [row] = groupSlips([slip({ book: 'Polymarket' }), slip(),
    slip({ book: 'fanduel', book_display: 'FanDuel' })]);
  assert.deepEqual(row.books, [{ book: 'polymarket', display: 'Polymarket' },
    { book: 'fanduel', display: 'FanDuel' }]);
  assert.equal(row.stake, 30);
  assert.equal(row.to_win, 27);
});

test('average odds weights stake and handles favorite/underdog prices', () => {
  const [row] = groupSlips([slip({ stake: 10, odds_american: -200, to_win: 5 }),
    slip({ stake: 30, odds_american: 200, to_win: 60 })]);
  assert.equal(row.odds_american, 163); // 65 profit / 40 stake = 1.625
  assert.equal(row.to_win, 65);
  assert.equal(groupSlips([slip({ odds_american: -100 }),
    slip({ odds_american: 100 })])[0].odds_american, 100);
});

test('game, market, period, side and exact line each distinguish contracts', () => {
  const rows = [slip(), slip({}, { game_uid: 'another-game' }),
    slip({}, { side: 'under' }), slip({}, { market: 'spread', side: 'home' }),
    slip({}, { period: 'first_half' }), slip({}, { line: 60.5001 })];
  assert.equal(groupSlips(rows).length, rows.length);
});

test('spread and moneyline selections combine only for the same team and line', () => {
  const home = { market: 'spread', side: 'home', line: -3.5 };
  assert.equal(groupSlips([slip({}, home), slip({}, home),
    slip({}, { ...home, side: 'away' }), slip({}, { ...home, line: -2.5 })]).length, 3);
  const ml = { market: 'moneyline', side: 'home', line: null };
  assert.equal(groupSlips([slip({}, ml), slip({}, ml),
    slip({}, { ...ml, side: 'away' })]).length, 2);
});

test('parlays and unresolved or unsupported contracts stay individual', () => {
  for (const s of [slip({ slip_type: 'parlay' }),
    slip({ legs: [slip().legs[0], slip({}, { game_uid: 'other' }).legs[0]] }),
    slip({}, { game_uid: null }), slip({}, { side: null }),
    slip({}, { market: 'team_total' }), slip({}, { line: null })]) {
    assert.equal(groupSlips([s, structuredClone(s)]).length, 2);
  }
});

test('mixed status aggregates settled P&L without losing pending exposure', () => {
  const won = slip({ status: 'won', open: false, pnl: 9, pnl_as_cash: 9 });
  const lost = slip({ status: 'lost', open: false, pnl: -10, pnl_as_cash: -10 });
  const rows = [slip(), won, lost];
  const [all] = groupSlips(rows);
  assert.equal(all.status, 'mixed');
  assert.equal(all.status_label, '1 pending · 1 won · 1 lost');
  assert.equal(all.open, true);
  assert.equal(all.partial_settlement, true);
  assert.equal(all.pnl, -1);
  assert.equal(all.stake, 30);
  const [settled] = groupSlips(rows.filter((s) => !s.open));
  assert.equal(settled.open, false);
  assert.equal(settled.stake, 20);
  assert.equal(settled.partial_settlement, false);
});

test('mixed funding and model flags describe the actual contributing bets', () => {
  const [row] = groupSlips([slip({ external_ref: 'first', placed_at_et: 'yesterday' }),
    slip({ free_play: true, stake_type: 'free_bet', model_qualified: false,
      status: 'lost', open: false, pnl: 0, pnl_as_cash: -10, source: 'manual' })]);
  assert.equal(row.free_play, false);
  assert.equal(row.free_play_stake, 10);
  assert.equal(row.model_qualified, false);
  assert.equal(row.model_count, 1);
  assert.equal(row.pnl, 0);
  assert.equal(row.pnl_as_cash, -10);
  assert.equal(row.external_ref, null);
  assert.equal(row.placed_at_et, null);
  assert.equal(row.source, 'sync / manual');
});

test('unavailable values do not become misleading partial sums or averages', () => {
  const [row] = groupSlips([slip(), slip({ to_win: null, odds_american: null })]);
  assert.equal(row.to_win, null);
  assert.equal(row.odds_american, null);
  assert.equal(groupSlips([slip({ stake: 0 }), slip({ stake: 0 })])[0].odds_american, null);
  assert.deepEqual(groupSlips([]), []);
});

test('team totals merge by selected team, never by over/under alone', () => {
  const home = { market: 'team_total', selection_team_uid: 'home-team', line: 24.5 };
  const away = { ...home, selection_team_uid: 'away-team' };
  const rows = groupSlips([slip({}, home), slip({}, home), slip({}, away), slip({}, away)]);
  assert.equal(rows.length, 2);
  assert.deepEqual(rows.map((s) => s.slip_count), [2, 2]);
});

test('unknown periods and contradictory team identity must never merge', () => {
  for (const leg of [{ period: null }, { period: '' },
    { market: 'spread', side: 'home', selection_team_uid: 'away-team',
      home_team_uid: 'home-team', away_team_uid: 'away-team' }]) {
    assert.equal(groupSlips([slip({}, leg), slip({}, leg)]).length, 2);
  }
});

test('book casing and whitespace do not duplicate the same book', () => {
  const [row] = groupSlips([slip(), slip({ book: ' Polymarket ' })]);
  assert.equal(row.books.length, 1);
});

test('group sums and weighted prices hold across 1000 fills and input reordering', () => {
  const rows = Array.from({length: 1000}, (_, i) => slip({stake: (i % 37 + 1) / 100,
    to_win: (i % 29 + 1) / 100, odds_american: i % 2 ? -200 : 150}));
  const expectedStake = rows.reduce((n,s) => n + Math.round(s.stake * 100), 0) / 100;
  const expectedWin = rows.reduce((n,s) => n + Math.round(s.to_win * 100), 0) / 100;
  const positiveStake = rows.filter((_,i) => i % 2 === 0).reduce((n,s) => n+s.stake,0);
  const negativeStake = rows.filter((_,i) => i % 2 === 1).reduce((n,s) => n+s.stake,0);
  const ratio = (positiveStake * 1.5 + negativeStake * 0.5) / expectedStake;
  const expectedOdds = Math.round(ratio >= 1 ? ratio * 100 : -100 / ratio);
  for (const input of [rows, [...rows].reverse()]) {
    const [group] = groupSlips(input);
    assert.equal(group.stake, expectedStake);
    assert.equal(group.to_win, expectedWin);
    assert.equal(group.odds_american, expectedOdds);
    assert.equal(group.slip_count, 1000);
  }
});

test('every settlement state and funding type preserves recorded money', () => {
  const states = ['won','lost','push','void','cashed_out','half_won','half_lost'];
  for (const status of states) for (const stake_type of ['cash','free_bet','bonus','boosted']) {
    const rows = [1,2].map((n) => slip({status, open:false, stake_type,
      free_play:['free_bet','bonus'].includes(stake_type), pnl:n*0.13, pnl_as_cash:-n*0.17}));
    const [group] = groupSlips(rows);
    assert.equal(group.pnl, 0.39);
    assert.equal(group.pnl_as_cash, -0.51);
    assert.equal(group.status, status);
    assert.equal(group.open, false);
  }
});
