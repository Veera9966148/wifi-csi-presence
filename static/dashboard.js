const statusLight = document.getElementById('statusLight');
const statusText = document.getElementById('statusText');
const statusSub = document.getElementById('statusSub');
const presenceFill = document.getElementById('presenceFill');
const presenceConfNum = document.getElementById('presenceConfNum');
const activityName = document.getElementById('activityName');
const activityFill = document.getElementById('activityFill');
const activityConfNum = document.getElementById('activityConfNum');
const lastUpdate = document.getElementById('lastUpdate');
const chips = document.querySelectorAll('.chip');

const canvas = document.getElementById('waveCanvas');
const ctx = canvas.getContext('2d');
let waveHistory = new Array(160).fill(20);

function resizeCanvas() {
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawWave() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = '#4FE0D0';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  const step = w / (waveHistory.length - 1);
  const min = Math.min(...waveHistory), max = Math.max(...waveHistory);
  const range = Math.max(max - min, 1);

  waveHistory.forEach((v, i) => {
    const x = i * step;
    const y = h - ((v - min) / range) * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function pushWaveSample(activity) {
  // Client-side ambient animation between polls, styled per current activity
  const base = 20;
  let amp = 0.5, freq = 0.05;
  if (activity === 'walking') { amp = 4; freq = 0.25; }
  else if (activity === 'standing') { amp = 1.2; freq = 0.1; }
  else if (activity === 'sitting') { amp = 0.9; freq = 0.08; }
  else { amp = 0.2; freq = 0.02; }

  const t = performance.now() / 200;
  const v = base + amp * Math.sin(t * freq * 10) + (Math.random() - 0.5) * amp * 0.6;
  waveHistory.push(v);
  waveHistory.shift();
  drawWave();
}

let currentActivity = 'empty';
function animate() {
  pushWaveSample(currentActivity);
  requestAnimationFrame(animate);
}
animate();

async function poll() {
  try {
    const res = await fetch('/api/predict');
    const data = await res.json();

    currentActivity = data.activity;

    if (data.human_detected) {
      statusLight.className = 'status-light present';
      statusText.textContent = 'HUMAN DETECTED';
      statusSub.textContent = 'multipath signature deviates from empty-room baseline';
    } else {
      statusLight.className = 'status-light absent';
      statusText.textContent = 'NO HUMAN PRESENT';
      statusSub.textContent = 'CSI matches empty-room baseline';
    }

    presenceFill.style.width = data.presence_confidence + '%';
    presenceConfNum.textContent = data.presence_confidence + '%';

    activityName.textContent = data.activity;
    activityFill.style.width = data.activity_confidence + '%';
    activityConfNum.textContent = data.activity_confidence + '%';

    chips.forEach(chip => {
      chip.classList.toggle('active', chip.dataset.act === data.activity);
    });

    lastUpdate.textContent = new Date().toLocaleTimeString();
  } catch (err) {
    statusSub.textContent = 'connection lost — check that the Flask server is running';
    console.error(err);
  }
}

poll();
setInterval(poll, 2000);
