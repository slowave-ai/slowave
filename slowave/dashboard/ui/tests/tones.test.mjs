import { test } from 'node:test';
import assert from 'node:assert/strict';
import { toneMap, statusTone, statusLabel, statusKey } from '../src/tones.ts';

const TONES = new Set(['ok', 'warn', 'bad', 'neutral', 'info']);

test('every key resolves to a known tone and a label', () => {
  for (const key of Object.keys(toneMap)) {
    assert.ok(TONES.has(toneMap[key]), key);
    assert.ok(statusLabel(key).length > 0, key);
  }
});

test('previously unstyled keys now have an explicit tone', () => {
  for (const key of ['archived', 'retired', 'failed', 'running', 'incomplete', 'no_effect', 'none', 'unknown', 'irrelevant',
    'captured', 'recorded', 'unverified', 'no_match', 'activation', 'recall', 'accepted', 'feedback_complete']) {
    assert.ok(key in toneMap, `${key} missing from tone map`);
  }
});

test('Feedback complete matches Complete; failed matches failure', () => {
  assert.equal(statusTone('Feedback complete'), statusTone('complete'));
  assert.equal(statusTone('failed'), statusTone('failure'));
  assert.equal(statusTone('failed'), 'bad');
});

test('tone semantics', () => {
  assert.equal(statusTone('needs_review'), 'warn');
  assert.equal(statusTone('harmed'), 'bad');
  assert.equal(statusTone('no_effect'), 'neutral');
  assert.equal(statusTone('running'), 'info');
  assert.equal(statusTone('something-new'), 'neutral');
  assert.equal(statusTone(null), 'neutral');
});

test('keys are normalised from display text', () => {
  assert.equal(statusKey('Needs review'), 'needs_review');
  assert.equal(statusLabel('idle_closed'), 'Idle-closed');
  assert.equal(statusLabel('totally_new'), 'totally new');
});
