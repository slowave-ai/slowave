import { test } from 'node:test';
import assert from 'node:assert/strict';
import { carryTimeRange } from '../src/timeRange.ts';

const O = 'http://x.test';
const carry = (to, path, search) => carryTimeRange(to, path, search, O);

test('hours carry between activity pages', () => {
  assert.equal(carry('/retrieval', '/', '?hours=168'), '/retrieval?hours=168');
  assert.equal(carry('/sessions', '/retrieval', '?hours=24&type=recall'), '/sessions?hours=24');
  assert.equal(carry('/graph', '/sessions', '?hours=720'), '/graph?hours=720');
});

test('custom from/to carry to activity pages', () => {
  assert.equal(carry('/sessions', '/retrieval', '?from=100&to=200&hours=24'), '/sessions?from=100&to=200');
  assert.equal(carry('/', '/sessions', '?from=100'), '/?from=100');
});

test('library pages never receive or originate a period', () => {
  assert.equal(carry('/memory', '/', '?hours=168'), '/memory');
  assert.equal(carry('/memory/12', '/retrieval', '?from=1&to=2'), '/memory/12');
  assert.equal(carry('/procedures', '/sessions', '?hours=24'), '/procedures');
  assert.equal(carry('/procedures/3', '/', '?hours=24'), '/procedures/3');
  assert.equal(carry('/retrieval', '/memory', '?hours=168&from=1'), '/retrieval');
  assert.equal(carry('/memory?scope=a', '/', '?hours=24'), '/memory?scope=a');
});

test('explicit destination params win and links are untouched', () => {
  assert.equal(carry('/retrieval?hours=all&from=5&to=6', '/', '?hours=24'), '/retrieval?hours=all&from=5&to=6');
  assert.equal(carry('/sessions?from=9', '/', '?hours=24&from=1&to=2'), '/sessions?from=9');
  assert.equal(carry('/memory?hours=all&saved_from=1', '/', '?hours=24'), '/memory?hours=all&saved_from=1');
  assert.equal(carry('/retrieval#x', '/', ''), '/retrieval#x');
  assert.equal(carry('https://other.test/a', '/', '?hours=24'), 'https://other.test/a');
});
