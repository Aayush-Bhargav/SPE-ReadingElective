import http from 'k6/http';
import { check, sleep } from 'k6';

// 1. VARIABLE DEFINITION
const PORT = __ENV.PORT || '8500';
const BASE_URL = `http://localhost:${PORT}`;

// Load the test image into memory once
const binFile = open('./larry.jpeg', 'b');

export const options = {
  stages: [
    { duration: '30s', target: 5 },  // Ramp up to 5 users
    { duration: '1m', target: 5 },   // Stay at 5 users
    { duration: '10s', target: 0 },  // Ramp down
  ],
};

export default function () {
  let res;
  let params = {};

  // 2. PAYLOAD LOGIC
  params = { headers: { 'Content-Type': 'image/jpeg' } };
  res = http.post(`${BASE_URL}/predict`, binFile, params);
  
  check(res, {
    'status is 200': (r) => r.status === 200,
  });

  // Logging only on first iteration to verify
  if (__ITER === 0 && __VU === 1) {
    const type = 'Binary';
    console.log(`Testing Port ${PORT} using ${type} payload`);
  }

  sleep(0.5); 
}