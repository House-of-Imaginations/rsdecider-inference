// k6 run -e SCENARIO=cold|hot|mixed|batch|overload -e RATE=200 stress/scenario.js
import { post, questions, state, zipf, ramp } from './lib.js';

const S = __ENV.SCENARIO || 'mixed';
const RATE = Number(__ENV.RATE || (S === 'overload' ? 2000 : 200));

export const options = {
  scenarios: { [S]: ramp(RATE, '2m') },
  thresholds: S === 'overload'
    ? { checks: ['rate>0.99'] } // only 200 or 529, never 5xx/timeouts
    : { http_req_failed: ['rate<0.01'], 'http_req_duration{path:/v1/decide}': ['p(99)<1000'] },
};

export default function () {
  const i = __ITER + __VU * 1e6;
  if (S === 'cold') post('/v1/decide', { state: state(i), questions: questions() });
  else if (S === 'hot') post('/v1/decide', { state: state(0), questions: questions() });
  else if (S === 'mixed' || S === 'overload') post('/v1/decide', { state: state(zipf(1000)), questions: questions() });
  else if (S === 'batch') post('/v1/decide/batch', {
    items: Array.from({ length: 8 }, (_, k) => ({ state: state(i * 8 + k), questions: questions() })),
  });
}
