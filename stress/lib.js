import http from 'k6/http';
import { check } from 'k6';

export const URL = __ENV.RSD_URL || 'http://127.0.0.1:3000';
const KEY = __ENV.RSD_KEY || 'stress-key';
const HEADERS = { 'content-type': 'application/json', authorization: `Bearer ${KEY}` };

export function questions() {
  return {
    action: { type: 'choice', instructions: 'How should we handle this?',
              criteria: { approve: 'refund', deny: 'no refund', escalate: 'human review' } },
    urgent: { type: 'noul', instructions: 'Is this urgent?' },
    severity: { type: 'score', instructions: 'Severity', criteria: ['low', 'medium', 'high'] },
  };
}

// ~25% non-English (Vietnamese, Spanish) so routing and the multilingual model get load.
export function state(i) {
  switch (i % 8) {
    case 3: return `Phiếu ${i}: khách hàng bị tính tiền hai lần cho đơn hàng và yêu cầu được hoàn tiền trong tuần này.`;
    case 7: return `Ticket ${i}: el cliente recibió dos cargos por su pedido y solicita un reembolso durante esta semana.`;
    default: return `Ticket ${i}: customer was charged twice for their order and asks for a refund within the week.`;
  }
}

// Unique ~90k-char state (under the default max_state_chars of 100k).
export function bigState(i) {
  return `Ticket ${i}: ` + 'customer was charged twice for their order and asks for a refund. '.repeat(1350);
}

// Zipf-ish over n states: small ids dominate, so ~70% of requests repeat a hot state.
export function zipf(n) {
  return Math.min(n - 1, Math.floor(Math.pow(Math.random(), 3) * n));
}

export function post(path, body) {
  const r = http.post(`${URL}${path}`, JSON.stringify(body), { headers: HEADERS, tags: { path } });
  check(r, { 'status 200/529': (x) => x.status === 200 || x.status === 529 });
  return r;
}

export function ramp(target, duration) {
  return {
    executor: 'ramping-arrival-rate',
    startRate: 1, timeUnit: '1s',
    preAllocatedVUs: 200, maxVUs: 2000,
    stages: [{ target, duration }, { target, duration: '1m' }],
  };
}
