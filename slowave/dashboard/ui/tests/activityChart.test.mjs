import { test } from 'node:test';
import assert from 'node:assert/strict';
import { countAxis, bucketIndex, bucketLabel, bucketDescription, bucketPeriod, useRate, ratePath } from '../src/activityChart.ts';

test('count axes include zero and never show fractional counts', () => {
  for (const value of [0, 1, 2, 3, 6, 11, 61, 250, 10001]) {
    const { max, ticks } = countAxis(value);
    assert.ok(max >= value);
    assert.equal(ticks.at(-1), 0);
    assert.ok(ticks.every(Number.isInteger));
    assert.ok(ticks.length <= 5);
  }
});

test('hover selects the actual bucket at both normal and overflowing SVG widths', () => {
  for (const width of [760, 1160]) {
    for (let index = 0; index < 50; index++) {
      const x = 240 + ((56 + (index + 0.5) * 816 / 50) / 900) * width;
      assert.equal(bucketIndex(x, 240, width, 50), index);
    }
  }
  assert.equal(bucketIndex(-1000, 0, 760, 50), 0);
  assert.equal(bucketIndex(10000, 0, 760, 50), 49);
});

test('day labels stay on the UTC bucket date across DST and local time zones', () => {
  const ts = Date.parse('2026-10-25T00:00:00Z') / 1000;
  assert.equal(bucketLabel(ts, 1440, 'en-US'), 'Oct 25, 2026');
  assert.match(bucketLabel(ts, 5, 'en-US'), /00:00/);
  assert.equal(bucketDescription(1440), 'Daily counts');
  assert.equal(bucketDescription(10080), 'Weekly counts');
});

test('day drill-down bounds exclude the next midnight and clip partial periods', () => {
  assert.deepEqual(bucketPeriod(86400, 1440, 0, 999999), {from: 86400, to: 172799});
  assert.deepEqual(bucketPeriod(86400, 1440, 90000, 100000), {from: 90000, to: 100000});
});

test('use rates use all deliveries and return a gap for no deliveries', () => {
  assert.equal(useRate(25, 100), 25);
  assert.equal(useRate(0, 10), 0);
  assert.equal(useRate(0, 0), null);
  // Period rate is weighted by deliveries, not an average of daily rates.
  assert.equal(useRate(1 + 9, 1 + 99), 10);
  assert.equal(ratePath([20, 30, null, 40], i => i, n => n), 'M 0 20 L 1 30 M 3 40');
});

test('right percentage axis margin preserves hover bucket mapping', () => {
  for (let i = 0; i < 50; i++) {
    const clientX = 240 + ((56 + (i + .5) * 788 / 50) / 900) * 1160;
    assert.equal(bucketIndex(clientX, 240, 1160, 50, 56), i);
  }
});

test('zero use plots at baseline while missing data breaks the line', () => {
  assert.equal(ratePath([50, 0, null, 0], i => i, n => 174 - n * 1.44), 'M 0 102 L 1 174 M 3 174');
});

test('no-delivery buckets draw a continuous zero baseline', () => {
  assert.equal(ratePath([50, null, 25].map(rate => rate ?? 0), i => i, n => 174 - n * 1.44), 'M 0 102 L 1 174 L 2 138');
});

test('rolling rate pools trailing buckets and stays defined across empty days', async () => {
  const { rollingRate, rollingWindow } = await import('../src/activityChart.ts');
  assert.equal(rollingWindow(1440), 7);
  assert.equal(rollingWindow(10080), 1);
  const out = rollingRate([1, 0, 3, 0], [2, 0, 4, 0], 3);
  assert.deepEqual(out, [50, 50, 4 / 6 * 100, 75]);
  assert.equal(rollingRate([0, 0], [0, 0], 7)[1], null);
});
