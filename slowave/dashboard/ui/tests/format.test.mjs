import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  EMPTY, formatCount, formatPercent, formatPercentOf, formatDuration, formatMs, formatBytes, formatTokens,
  formatRelative, formatDay, shortScope, middleEllipsis, toDateInput, fromDateInput,
} from '../src/format.ts';

test('one placeholder for missing values', () => {
  assert.equal(EMPTY, '—');
  for (const fn of [formatCount, formatPercent, formatDuration, formatMs, formatBytes, formatTokens, formatDay, formatRelative]) {
    assert.equal(fn(null), EMPTY);
    assert.equal(fn(undefined), EMPTY);
    assert.equal(fn(''), EMPTY);
  }
  assert.equal(formatPercentOf(1, 0), EMPTY);
});

test('counts use en-GB thousands separators', () => {
  assert.equal(formatCount(1234567), '1,234,567');
  assert.equal(formatCount('42'), '42');
  assert.equal(formatCount(0), '0');
  assert.equal(formatCount('abc'), EMPTY);
});

test('percentages never round to a misleading 0% or 100%', () => {
  assert.equal(formatPercent(0), '0%');
  assert.equal(formatPercent(100), '100%');
  assert.equal(formatPercent(42.4), '42%');
  assert.equal(formatPercent(0.2), '0.2%');
  assert.equal(formatPercent(99.8), '99.8%');
  assert.equal(formatPercentOf(1, 3), '33%');
});

test('durations round once so 59.6s is 1m and sub-second is <1s', () => {
  assert.equal(formatDuration(0.2), '<1s');
  assert.equal(formatDuration(1), '1s');
  assert.equal(formatDuration(59.4), '59s');
  assert.equal(formatDuration(59.6), '1m');
  assert.equal(formatDuration(60), '1m');
  assert.equal(formatDuration(65), '1m 5s');
  assert.equal(formatDuration(3599.7), '1h');
  assert.equal(formatDuration(3725), '1h 2m');
  assert.equal(formatDuration(-1), EMPTY);
  assert.equal(formatMs(0), '<1ms');
  assert.equal(formatMs(250), '250ms');
  assert.equal(formatMs(1500), '2s');
});

test('bytes and tokens are compact', () => {
  assert.equal(formatBytes(512), '512 B');
  assert.equal(formatBytes(2048), '2.0 KB');
  assert.equal(formatTokens(950), '950');
  assert.equal(formatTokens(1234), '1.2k');
  assert.equal(formatTokens(12_400), '12k');
  assert.equal(formatTokens(2_500_000), '2.5M');
});

test('relative under 7 days, absolute date after', () => {
  const now = 1_800_000_000;
  assert.equal(formatRelative(now - 10, now), 'just now');
  assert.equal(formatRelative(now - 600, now), '10m ago');
  assert.equal(formatRelative(now - 7200, now), '2h ago');
  assert.equal(formatRelative(now - 86400 * 3, now), '3d ago');
  assert.match(formatRelative(now - 86400 * 30, now), /^\d{1,2} [A-Z][a-z]{2} \d{4}$/);
});

test('scopes and ids are shortened for display only', () => {
  assert.equal(shortScope('project:slowave-private'), 'slowave-private');
  assert.equal(shortScope('user:me'), 'user:me');
  assert.equal(middleEllipsis('0f3a9c1e-1111-2222-3333-7b2d5e6f7a8b'), '0f3a9c1e…7a8b');
  assert.equal(middleEllipsis('sch_12'), 'sch_12');
});

test('date inputs round-trip in local time', () => {
  const start = fromDateInput('2026-11-06');
  assert.equal(toDateInput(start), '2026-11-06');
  assert.ok(fromDateInput('2026-11-06', true) > start);
  assert.equal(fromDateInput('nope'), undefined);
});
