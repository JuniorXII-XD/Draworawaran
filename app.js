import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/vision_bundle.mjs";

// โครงสร้างเส้นเชื่อมต่อของมือ (Skeleton)
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17]
];
const FINGER_TIPS  = [4, 8, 12, 16, 20];
const FINGER_BASES = [2, 6, 10, 14, 18];

// ดึงองค์ประกอบต่างๆ จาก HTML DOM
const video      = document.getElementById("video");
const camCanvas  = document.getElementById("cam-canvas");
const drawCanvas = document.getElementById("draw-canvas");
const camCtx     = camCanvas.getContext("2d", { alpha: false });
camCtx.imageSmoothingEnabled = false;
const drawCtx    = drawCanvas.getContext("2d");

const gestureEl  = document.getElementById("gesture");
const handCountEl= document.getElementById("hand-count");
const fpsEl      = document.getElementById("fps");
const statusEl   = document.getElementById("status");
const statusScr  = document.getElementById("status-screen");
const toggleBtn  = document.getElementById("toggle-draw");
const clearBtn   = document.getElementById("clear-draw");
const modeEl     = document.getElementById("mode-indicator");
const wrap       = document.getElementById("canvas-wrap");
const camPill    = document.getElementById("nav-cam-pill");

// ตั้งค่าตัวแปรเริ่มต้น
let landmarker   = null;
let frameCount   = 0, fps = 0, fpsTimer = 0;
let drawEnabled  = true;

const smoothed   = { Left: null, Right: null };
const lastDrawPt = { Left: null, Right: null };
const wasPinch   = { Left: false, Right: false };

const ALPHA        = 0.5;      // ตัวคูณสำหรับทำเส้นเคลื่อนไหวให้นุ่มนวล (Lerp)
const PINCH_PX_THR = 0.075;    // ระยะห่างขั้นต่ำที่นับว่าเป็นการบีบนิ้ว
const ERASER_R     = 40;       // รัศมีของยางลบ

let PEN_W     = 5;
let penColor  = "#ffffff";
let penGlow   = "#b48ef5";

const DETECT_INTERVAL = 33;    // หน่วงเวลาการประมวลผลโมเดล (ประมาณ 30 FPS)
let lastDetectTime = 0;
let lastResult     = null;

// ฟังก์ชันคำนวณตำแหน่งให้นิ่งและนุ่มนวลขึ้น
function lerpPt(prev, next, a) {
  if (!prev) return { x: next.x, y: next.y };
  return { x: prev.x + (next.x - prev.x) * a, y: prev.y + (next.y - prev.y) * a };
}

// คำนวณระยะห่างระหว่างจุด
function ndist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

// เริ่มต้นระบบ โหลดโมเดล และเปิดใช้งานกล้อง
async function init() {
  statusEl.textContent = "Loading...";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
  );
  landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.4,
    minTrackingConfidence: 0.4
  });

  document.getElementById("nav-model-pill").classList.add("live");

  statusEl.textContent = "l";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } } 
    });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      const W = video.videoWidth, H = video.videoHeight;
      camCanvas.width = drawCanvas.width = W;
      camCanvas.height = drawCanvas.height = H;
      wrap.style.aspectRatio = `${W} / ${H}`;
      camCanvas.style.transform = drawCanvas.style.transform = "scaleX(-1)"; // กลับด้านกระจกเพื่อความคุ้นชิน

      camPill.classList.add("live");
      camPill.querySelector("span:last-child").textContent = "Live";

      video.play();
      statusScr.style.display = "none";
      requestAnimationFrame(detect);
    };
  } catch (e) {
    statusEl.textContent = "❌ Unable to access camera";
    console.error(e);
  }
}

// ฟังก์ชันวิเคราะห์นิ้วมือเพื่อแปลงเป็นข้อความบ่งบอกท่าทาง (Gesture)
function gestureLabel(lm) {
  const tips  = FINGER_TIPS.map(i => lm[i]);
  const bases = FINGER_BASES.map(i => lm[i]);
  const thumbOpen = lm[4].x < lm[3].x;
  const ext = [1,2,3,4].map(i => tips[i].y < bases[i].y);
  const n = ext.filter(Boolean).length + (thumbOpen ? 1 : 0);

  if (!thumbOpen && ext.every(e => !e)) return "✊ Fist";
  if (thumbOpen && ext.every(e => e))   return "🖐 Open Hand";
  if (!thumbOpen && ext[0] && !ext[1] && !ext[2] && !ext[3]) return "☝️ Point";
  if (!thumbOpen && ext[0] &&  ext[1] && !ext[2] && !ext[3]) return "✌️ Peace";
  if (thumbOpen && !ext[0] && !ext[1] && !ext[2] && !ext[3]) return "👍 Thumbs Up";
  if (thumbOpen && !ext[0] && !ext[1] && !ext[2] &&  ext[3]) return "🤙 Call Me";
  return `🖐 ${n} Finger${n !== 1 ? 's' : ''}`;
}

// วาดเส้นโครงสร้างกระดูกและจุดพิกัดบนมือ
function drawSkeleton(lm, hand) {
  const W = camCanvas.width, H = camCanvas.height;
  const isL = hand === "Left";
  const col1 = isL ? "#ff85c2" : "#b48ef5";
  const tipC = isL ? "#ffffff" : "#ede0ff";

  camCtx.beginPath();
  camCtx.strokeStyle = col1 + "cc";
  camCtx.lineWidth = 3;
  camCtx.lineCap = "round";
  CONNECTIONS.forEach(([a, b]) => {
    camCtx.moveTo(lm[a].x * W, lm[a].y * H);
    camCtx.lineTo(lm[b].x * W, lm[b].y * H);
  });
  camCtx.stroke();

  camCtx.fillStyle = col1;
  camCtx.beginPath();
  lm.forEach((pt, i) => {
    if (!FINGER_TIPS.includes(i)) {
      camCtx.moveTo(pt.x * W + 4, pt.y * H);
      camCtx.arc(pt.x * W, pt.y * H, 4, 0, Math.PI * 2);
    }
  });
  camCtx.fill();

  camCtx.shadowColor = col1;
  camCtx.shadowBlur = 12;
  camCtx.fillStyle = tipC;
  camCtx.beginPath();
  FINGER_TIPS.forEach(i => {
    const x = lm[i].x * W, y = lm[i].y * H;
    camCtx.moveTo(x + 7, y);
    camCtx.arc(x, y, 7, 0, Math.PI * 2);
  });
  camCtx.fill();
  camCtx.shadowBlur = 0;
}

// วาดวงกลมหรือเป้าเล็งตามตำแหน่งนิ้วชี้
function drawCursor(x, y, isEraser, active) {
  if (isEraser) {
    camCtx.beginPath(); camCtx.arc(x, y, ERASER_R, 0, Math.PI * 2);
    camCtx.strokeStyle = active ? "#ff85c2" : "rgba(255,133,194,0.45)";
    camCtx.lineWidth = active ? 3 : 1.5;
    camCtx.setLineDash([6, 4]); camCtx.stroke(); camCtx.setLineDash([]);
  } else {
    const r = active ? PEN_W : 6;
    camCtx.beginPath(); camCtx.arc(x, y, r, 0, Math.PI * 2);
    camCtx.fillStyle = active ? penColor : "rgba(255,255,255,0.5)";
    if (active) { camCtx.shadowColor = penGlow; camCtx.shadowBlur = 14; }
    camCtx.fill();
    camCtx.shadowBlur = 0;
    if (!active) {
      camCtx.strokeStyle = "rgba(255,255,255,0.4)"; camCtx.lineWidth = 1.5;
      camCtx.beginPath();
      camCtx.moveTo(x - 14, y); camCtx.lineTo(x + 14, y);
      camCtx.moveTo(x, y - 14); camCtx.lineTo(x, y + 14);
      camCtx.stroke();
    }
  }
}

// สร้างแผ่นฟิล์มลดความสว่างของภาพวิดีโอข้างหลังให้เห็นเส้นวาดชัดเจนขึ้น
const dimOverlay = (() => {
  const oc = new OffscreenCanvas(2, 2);
  const c = oc.getContext("2d");
  c.fillStyle = "rgba(0,0,0,0.18)";
  c.fillRect(0, 0, 2, 2);
  return oc;
})();

// ลูปประมวลผลภาพหลัก (Main Loop)
function render(ts) {
  frameCount++;
  if (ts - fpsTimer > 1000) {
    fps = frameCount; frameCount = 0; fpsTimer = ts;
    fpsEl.textContent = `${fps} FPS`;
  }

  if (ts - lastDetectTime >= DETECT_INTERVAL) {
    lastDetectTime = ts;
    lastResult = landmarker.detectForVideo(video, ts);
  }

  const res = lastResult;
  if (!res) { requestAnimationFrame(render); return; }

  const W = camCanvas.width, H = camCanvas.height;

  camCtx.drawImage(video, 0, 0, W, H);
  camCtx.drawImage(dimOverlay, 0, 0, W, H);

  const hands = res.landmarks || [];
  handCountEl.textContent = `${hands.length} มือ`;

  const present = new Set();
  const labels  = [];

  hands.forEach((lm, i) => {
    const hand = res.handedness?.[i]?.[0]?.categoryName || "Right";
    present.add(hand);
    drawSkeleton(lm, hand);
    labels.push(gestureLabel(lm));

    if (!drawEnabled) return;

    const thumb = lm[4], index = lm[8];
    const pinching = ndist(thumb, index) < PINCH_PX_THR;
    const raw = { x: (thumb.x + index.x) / 2 * W, y: (thumb.y + index.y) / 2 * H };
    smoothed[hand] = lerpPt(smoothed[hand], raw, ALPHA);
    const sx = smoothed[hand].x, sy = smoothed[hand].y;

    drawCursor(sx, sy, hand === "Left", pinching);

    if (pinching) {
      if (hand === "Left") {
        // โหมดยางลบ (ลบเส้น)
        drawCtx.save();
        drawCtx.globalCompositeOperation = "destination-out";
        drawCtx.beginPath(); drawCtx.arc(sx, sy, ERASER_R, 0, Math.PI * 2);
        drawCtx.fill();
        drawCtx.restore();
      } else {
        // โหมดปากกา (วาดเส้น)
        if (wasPinch[hand] && lastDrawPt[hand]) {
          drawCtx.beginPath();
          drawCtx.moveTo(lastDrawPt[hand].x, lastDrawPt[hand].y);
          drawCtx.lineTo(sx, sy);
          drawCtx.strokeStyle = penColor;
          drawCtx.lineWidth = PEN_W;
          drawCtx.lineCap = "round"; drawCtx.lineJoin = "round";
          drawCtx.shadowColor = penGlow; drawCtx.shadowBlur = 10;
          drawCtx.stroke(); drawCtx.shadowBlur = 0;
        }
        lastDrawPt[hand] = { x: sx, y: sy };
      }
      wasPinch[hand] = true;
    } else {
      wasPinch[hand] = false;
      lastDrawPt[hand] = null;
    }
  });

  // ล้างค่าเมื่อมือหลุดออกนอกจอ
  ["Left", "Right"].forEach(h => {
    if (!present.has(h)) { smoothed[h] = null; wasPinch[h] = false; lastDrawPt[h] = null; }
  });

  gestureEl.textContent = labels.length ? labels.join("  ") : "ยกมือขึ้น";
  gestureEl.style.opacity = labels.length ? "1" : "0.35";

  requestAnimationFrame(render);
}

function detect(ts) { render(ts); }

// ══════════════════ EVENT LISTENERS (การควบคุม) ══════════════════

// เปิด-ปิดระบบวาดภาพ
toggleBtn.addEventListener("click", () => {
  drawEnabled = !drawEnabled;
  toggleBtn.querySelector(".btn-label").textContent = drawEnabled ? "วาด: เปิด" : "วาด: ปิด";
  toggleBtn.classList.toggle("off", !drawEnabled);
  toggleBtn.classList.toggle("primary", drawEnabled);
  modeEl.style.display = drawEnabled ? "flex" : "none";
});

// ล้างกระดานวาดรูป
clearBtn.addEventListener("click", () => {
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
});

// จับคู่สีนีออนเปล่งประกาย (Neon Glow)
const glowMap = {
  "#ffffff":  "#b48ef5",
  "#ff85c2":  "#ff85c2",
  "#b48ef5":  "#b48ef5",
  "#6ec6ff":  "#6ec6ff",
  "#5dddb8":  "#5dddb8",
  "#ffd95a":  "#ffd95a",
  "#ffad6b":  "#ffad6b",
  "#ff6b6b":  "#ff6b6b",
};

// เลือกสีปากกา
const swatches = document.querySelectorAll(".color-swatch");
const sizeDisplay = document.getElementById("size-display");

swatches.forEach(sw => {
  sw.addEventListener("click", () => {
    swatches.forEach(s => s.classList.remove("active"));
    sw.classList.add("active");
    penColor = sw.dataset.color;
    penGlow  = glowMap[penColor] || penColor;
  });
});

// เพิ่มขนาดปากกา
document.getElementById("size-up").addEventListener("click", () => {
  PEN_W = Math.min(30, PEN_W + 2);
  sizeDisplay.textContent = PEN_W;
});

// ลดขนาดปากกา
document.getElementById("size-down").addEventListener("click", () => {
  PEN_W = Math.max(1, PEN_W - 2);
  sizeDisplay.textContent = PEN_W;
});

// รันระบบเมื่อทุกอย่างพร้อม
init();


