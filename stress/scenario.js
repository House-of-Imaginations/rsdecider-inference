// k6 run -e SCENARIO=cold|hot|mixed|batch|overload|tokenize -e RATE=20 stress/scenario.js (run.sh sets RATE per scenario)
import exec from 'k6/execution';
import { post, questions, state, bigState, zipf, ramp } from './lib.js';

const S = __ENV.SCENARIO || 'mixed';
const DEFAULT_RATE = { cold: 20, hot: 3000, mixed: 60, batch: 5, overload: 100, tokenize: 200 };
const SHEDS = S === 'overload' || S === 'tokenize';
const RATE = Number(__ENV.RATE || DEFAULT_RATE[S]);
const ACCEPTED = 'http_req_duration{expected_response:true}';

export const options = {
  scenarios: { [S]: ramp(RATE, '2m') },
  thresholds: SHEDS
    // only 200 or 529, never 5xx/504; accepted requests finish inside the 10 s deadline
    ? { checks: ['rate>0.99'], [ACCEPTED]: ['p(99)<10000'], dropped_iterations: ['rate<1'] }
    : { http_req_failed: ['rate<0.01'], [ACCEPTED]: ['p(99)<1000'], dropped_iterations: ['rate<1'] },
};

export default function () {
  // A globally unique, monotonically increasing counter across the whole scenario. __ITER + __VU * 1e6 is
  // not: 1e6 is divisible by 8, so state()'s `i % 8` language pick collapsed to __ITER alone, and at low
  // rates with 200 pre-allocated VUs almost every VU sits at __ITER 0-2 (English), starving non-English mix.
  const i = exec.scenario.iterationInTest;
  if (S === 'cold' || S === 'overload') post('/v1/decide', { state: state(i), questions: questions() });
  else if (S === 'hot') post('/v1/decide', { state: state(0), questions: questions() });
  else if (S === 'mixed') post('/v1/decide', { state: state(zipf(100000)), questions: questions() });
  // ~90k-char unique states (~18 ms each to tokenize): above ~55 req/s per tokenize thread the queue fills and sheds 529
  else if (S === 'tokenize') post('/v1/decide', { state: bigState(i), questions: questions() });
  else if (S === 'batch') post('/v1/decide/batch', {
    items: Array.from({ length: 8 }, (_, k) => ({ state: state(i * 8 + k), questions: questions() })),
  });
}
