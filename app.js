// Wizard Screens
const screenWelcome = document.getElementById('screen-welcome');
const screenCalibrate = document.getElementById('screen-calibrate');
const screenTrain = document.getElementById('screen-train');
const screenLive = document.getElementById('screen-live');

// UI Elements
const holeCountSelect = document.getElementById('holeCount');
const btnStartWizard = document.getElementById('btnStartWizard');
const targetStateText = document.getElementById('targetStateText');
const btnRecordState = document.getElementById('btnRecordState');
const progressLabel = document.getElementById('progressLabel');
const captureProgressBar = document.getElementById('captureProgressBar');
const trainingLogText = document.getElementById('trainingLogText');
const liveStateText = document.getElementById('liveStateText');
const btnRecalibrate = document.getElementById('btnRecalibrate');

// Audio (Native Web Audio API - bypasses p5.FFT to avoid master-output contamination)
let osc;                    // p5.Oscillator for the sweep tone
let analyserNode = null;    // Native AnalyserNode connected ONLY to mic
let micStream = null;
let micSourceNode = null;
let audioCtx = null;
let audioStarted = false;
let sweepInterval = null;

const SWEEP_MIN = 12500;    // 12.5 kHz
const SWEEP_MAX = 18000;    // 18.0 kHz
const SWEEP_DUR_SEC = 0.05; // 50ms per sweep

// Feature extraction: only use bins in our sweep frequency range
let sweepBinStart = 0;
let sweepBinEnd = 512;
let featureCount = 512;

// ML
let nn;
let isModelReady = false;
let isClassifying = false;
let stateList = [];
let currentStateIndex = 0;
let currentSamples = 0;
const SAMPLES_NEEDED = 80;
let recordingInterval = null;
let isRecording = false;
let lastSpectrum = [];
let spectrumBuffer = []; // Buffer for temporal averaging
const AVG_WINDOW = 5;    // Average last 5 frames for stability
const REDUCED_BINS = 64; // Reduce feature count for better generalization

// ------------------- p5.js setup ------------------- //

function setup() {
  let cnv = createCanvas(windowWidth, 100);
  cnv.parent('fftContainer');

  // Oscillator for sweep output
  osc = new p5.Oscillator('sine');
  osc.amp(1.0);

  setupEvents();
}

function draw() {
  background(13, 17, 23);

  if (!analyserNode) {
    // Show "waiting for audio" message
    fill(100);
    noStroke();
    textSize(12);
    textAlign(CENTER, CENTER);
    text('Tap "Initialize & Start audio" to begin', width / 2, height / 2);
    return;
  }

  // Read spectrum from native analyser
  const freqData = new Uint8Array(analyserNode.frequencyBinCount);
  analyserNode.getByteFrequencyData(freqData);
  const currentFrame = Array.from(freqData);

  // Temporal Averaging: Maintain a sliding window of frames
  spectrumBuffer.push(currentFrame);
  if (spectrumBuffer.length > AVG_WINDOW) spectrumBuffer.shift();

  // Compute the average spectrum
  lastSpectrum = new Array(currentFrame.length).fill(0);
  for (let frame of spectrumBuffer) {
    for (let i = 0; i < frame.length; i++) {
      lastSpectrum[i] += frame[i];
    }
  }
  for (let i = 0; i < lastSpectrum.length; i++) {
    lastSpectrum[i] /= spectrumBuffer.length;
  }

  // Draw spectrum visually
  noStroke();
  for (let i = 0; i < lastSpectrum.length; i++) {
    let x = map(i, 0, lastSpectrum.length, 0, width);
    let barW = width / lastSpectrum.length;
    let h = map(lastSpectrum[i], 0, 255, 0, height - 14);

    if (i >= sweepBinStart && i < sweepBinEnd) {
      fill(63, 185, 80);
    } else {
      fill(30, 40, 50);
    }
    rect(x, height, barW, -h);
  }

  // Signal level indicator
  const sweepSlice = lastSpectrum.slice(sweepBinStart, sweepBinEnd);
  const signalLevel = sweepSlice.reduce((a,b) => a+b, 0) / (sweepSlice.length || 1);
  fill(255, 200, 0);
  noStroke();
  textSize(10);
  textAlign(LEFT, TOP);
  text(`Signal: ${signalLevel.toFixed(0)}/255 | Averaging: ${spectrumBuffer.length} frames | Features: ${REDUCED_BINS}`, 6, 2);

  // Throttled inference in live mode
  if (isModelReady && !isClassifying && screenLive.classList.contains('active')) {
    isClassifying = true;
    const features = getProcessedFeatures(lastSpectrum);
    nn.classify(features, handleClassification);
  }
}

// Function to reduce FFT bins into super-bins for better ML generalization
function getProcessedFeatures(spectrum) {
  const sweepSlice = spectrum.slice(sweepBinStart, sweepBinEnd);
  const binSize = Math.floor(sweepSlice.length / REDUCED_BINS);
  let reduced = [];

  for (let i = 0; i < REDUCED_BINS; i++) {
    let sum = 0;
    let count = 0;
    for (let j = 0; j < binSize; j++) {
      let idx = i * binSize + j;
      if (idx < sweepSlice.length) {
        sum += sweepSlice[idx];
        count++;
      }
    }
    reduced.push((sum / (count || 1)) / 255.0);
  }
  return reduced;
}

function windowResized() {
  resizeCanvas(windowWidth, 100);
}

// ------------------- Audio Engine ------------------- //

async function startAudioEngine() {
  // Step 1: Resume AudioContext (mandatory user-gesture requirement)
  await userStartAudio();
  audioCtx = getAudioContext();

  // Step 2: Compute sweep bin range from actual sample rate
  const binCount = 1024; // analyserNode.fftSize = 2048 → frequencyBinCount = 1024
  const nyquist = audioCtx.sampleRate / 2;
  const hzPerBin = nyquist / binCount;
  sweepBinStart = Math.max(0, Math.floor(SWEEP_MIN / hzPerBin) - 2);
  sweepBinEnd   = Math.min(binCount, Math.ceil(SWEEP_MAX / hzPerBin) + 2);
  featureCount  = sweepBinEnd - sweepBinStart;
  console.log(`SR: ${audioCtx.sampleRate} Hz | Hz/bin: ${hzPerBin.toFixed(1)} | Sweep bins: ${sweepBinStart}-${sweepBinEnd} (${featureCount} features)`);

  // Step 3: Create a native AnalyserNode — NOT connected to master output
  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 2048;              // 1024 frequency bins
  analyserNode.smoothingTimeConstant = 0.1; // Fast response

  // Step 4: Request microphone with all processing disabled
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        latency: 0
      }
    });
    console.log('Microphone acquired with echo cancellation disabled.');
  } catch (err) {
    console.warn('Could not disable echo cancellation, falling back to default:', err);
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  }

  // Step 5: Connect mic → analyser ONLY (no oscillator leaking into FFT)
  micSourceNode = audioCtx.createMediaStreamSource(micStream);
  micSourceNode.connect(analyserNode);
  // Do NOT connect analyserNode to audioCtx.destination (we don't want to hear the mic)

  // Step 6: Reinitialize ML model with reduced feature count
  const holes = parseInt(holeCountSelect.value);
  stateList = generatePermutations(holes);
  nn = ml5.neuralNetwork({
    inputs: REDUCED_BINS,
    task: 'classification',
    debug: false
  });

  // Step 7: Start the oscillator sweep
  osc.start();
  audioStarted = true;
  if (sweepInterval) clearInterval(sweepInterval);
  triggerSweep();
  sweepInterval = setInterval(triggerSweep, SWEEP_DUR_SEC * 1000);
}

function stopAudioEngine() {
  if (sweepInterval) { clearInterval(sweepInterval); sweepInterval = null; }
  osc.stop();
  if (micSourceNode) { micSourceNode.disconnect(); micSourceNode = null; }
  if (micStream)     { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  analyserNode = null;
  audioStarted = false;
  lastSpectrum = [];
}

function triggerSweep() {
  if (!audioStarted) return;
  osc.freq(SWEEP_MIN, 0);           // instant snap to low end
  osc.freq(SWEEP_MAX, SWEEP_DUR_SEC); // linear ramp to high end
}

// ------------------- Screen Management ------------------- //

function showScreen(screenEl) {
  [screenWelcome, screenCalibrate, screenTrain, screenLive].forEach(s => {
    s.classList.remove('active');
    s.classList.add('hidden');
  });
  screenEl.classList.remove('hidden');
  setTimeout(() => screenEl.classList.add('active'), 10);
}

function generatePermutations(holes) {
  if (holes === 1) return ['O', 'C'];
  if (holes === 2) return ['OO', 'OC', 'CO', 'CC'];
  if (holes === 3) return ['OOO', 'OOC', 'OCO', 'OCC', 'COO', 'COC', 'CCO', 'CCC'];
  return [];
}

function setupEvents() {
  btnStartWizard.addEventListener('click', async () => {
    await startAudioEngine(); // startAudioEngine now sets stateList + nn
    currentStateIndex = 0;
    setupNextState();
    showScreen(screenCalibrate);
  });

  const startBtnAction = (e) => { e.preventDefault(); startRecording(); };
  const stopBtnAction  = (e) => { e.preventDefault(); stopRecording(); };

  btnRecordState.addEventListener('mousedown', startBtnAction);
  btnRecordState.addEventListener('touchstart', startBtnAction);
  btnRecordState.addEventListener('mouseup',    stopBtnAction);
  btnRecordState.addEventListener('mouseleave', stopBtnAction);
  btnRecordState.addEventListener('touchend',   stopBtnAction);
  btnRecordState.addEventListener('touchcancel',stopBtnAction);

  btnRecalibrate.addEventListener('click', () => {
    isModelReady = false;
    stopAudioEngine();
    showScreen(screenWelcome);
  });
}

// ------------------- Calibration ------------------- //

function setupNextState() {
  const targetClass = stateList[currentStateIndex];
  targetStateText.textContent = targetClass;
  currentSamples = 0;
  updateProgressUI();
  btnRecordState.disabled = false;
  btnRecordState.textContent = 'Hold to Calibrate State';
  progressLabel.textContent = `Awaiting input for ${targetClass}...`;
}

function startRecording() {
  if (isRecording) return;
  isRecording = true;
  btnRecordState.textContent = 'Recording... Keep holding!';
  progressLabel.textContent = 'Capturing acoustic profile...';

  recordingInterval = setInterval(() => {
    if (!audioStarted || lastSpectrum.length === 0) return;

    // Extract processed features
    const inputs = getProcessedFeatures(lastSpectrum);
    const target = [stateList[currentStateIndex]];

    nn.addData(inputs, target);
    currentSamples++;
    updateProgressUI();

    if (currentSamples >= SAMPLES_NEEDED) {
      stopRecording();
      handleStateComplete();
    }
  }, 50);
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  clearInterval(recordingInterval);
  if (currentSamples > 0 && currentSamples < SAMPLES_NEEDED) {
    btnRecordState.textContent = 'Hold to Resume Calibration';
    progressLabel.textContent = 'Please hold until the bar is full!';
  }
}

function updateProgressUI() {
  const pct = Math.min((currentSamples / SAMPLES_NEEDED) * 100, 100);
  captureProgressBar.style.width = `${pct}%`;
}

function handleStateComplete() {
  btnRecordState.disabled = true;
  btnRecordState.textContent = 'Great!';
  progressLabel.textContent = 'Captured successfully.';
  currentStateIndex++;

  if (currentStateIndex < stateList.length) {
    setTimeout(setupNextState, 1000);
  } else {
    setTimeout(startTraining, 1000);
  }
}

// ------------------- Training ------------------- //

function startTraining() {
  showScreen(screenTrain);
  trainingLogText.textContent = 'Normalizing acoustic data...';
  nn.normalizeData();
  nn.train({ epochs: 50, batchSize: 16 }, whileTraining, finishedTraining);
}

function whileTraining(epoch, loss) {
  trainingLogText.textContent = `Epoch ${epoch + 1}/50 — Loss: ${loss.loss.toFixed(4)}`;
}

function finishedTraining() {
  isModelReady = true;
  showScreen(screenLive);
}

// ------------------- Inference ------------------- //

function handleClassification(error, results) {
  isClassifying = false;
  if (error || !isModelReady) return;
  if (results && results.length > 0) {
    const best = results[0];
    if (best.confidence > 0.5) {
      liveStateText.textContent = best.label;
      liveStateText.style.color = 'var(--text-main)';
    } else {
      liveStateText.textContent = '---';
      liveStateText.style.color = 'var(--text-muted)';
    }
  }
}
