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

// Audio (p5.sound)
let osc, fft;
let micStream = null;     // raw MediaStream
let micSourceNode = null; // Web Audio source node
let audioCtx = null;
let audioStarted = false;
let sweepInterval = null;
const SWEEP_MIN = 12500; // 12.5 kHz
const SWEEP_MAX = 18000; // 18.0 kHz
const SWEEP_DUR_SEC = 0.05; // 50ms rapid sweep for stability

// ML / Flow Variables
let nn;
let isModelReady = false;
let isClassifying = false;  // FIX: prevent 60fps classify() flooding
let stateList = [];
let currentStateIndex = 0;
let currentSamples = 0;
const SAMPLES_NEEDED = 60; // 60 ticks * 50ms = 3 seconds per state (more data)
let recordingInterval = null;
let isRecording = false;
let lastSpectrum = [];  // FIX: single source of truth for spectrum

// Initialize p5.js
function setup() {
  let cnv = createCanvas(windowWidth, 100);
  cnv.parent('fftContainer');

  // Oscillator only — no p5.AudioIn (we bypass it below for echo cancellation)
  osc = new p5.Oscillator('sine');
  osc.amp(1.0);

  // FFT — we'll connect it manually to the raw mic stream
  fft = new p5.FFT(0.1, 1024);
  // NOTE: do NOT call fft.setInput() here — we connect directly to fft.analyser below

  setupEvents();
}

function draw() {
  background(13, 17, 23);

  // FIX: Single source of truth — analyze once per frame, share everywhere
  lastSpectrum = fft.analyze();

  // Draw FFT Spectrum Visually
  noStroke();
  fill(88, 166, 255);
  for (let i = 0; i < lastSpectrum.length; i++) {
    let x = map(i, 0, lastSpectrum.length, 0, width);
    let h = -height + map(lastSpectrum[i], 0, 255, height, 0);
    rect(x, height, width / lastSpectrum.length, h);
  }

  // FIX: Throttled inference — only classify when previous call has returned
  if (isModelReady && !isClassifying && screenLive.classList.contains('active')) {
    isClassifying = true;
    let inputs = lastSpectrum.map(v => v / 255.0);
    nn.classify(inputs, handleClassification);
  }
}

function windowResized() {
  resizeCanvas(windowWidth, 100);
}

// ----------------- Sweep Logic ----------------- //

function triggerSweep() {
  if (!audioStarted) return;
  // Snap down immediately
  osc.freq(SWEEP_MIN, 0);
  // Linear ramp up
  osc.freq(SWEEP_MAX, SWEEP_DUR_SEC);
}

async function startAudioEngine() {
  // 1. Resume Web Audio context (required by browser autoplay policy)
  await userStartAudio();
  audioCtx = getAudioContext();

  // 2. Get microphone with echo cancellation DISABLED.
  //    This is critical — without this, the browser strips the very
  //    speaker signal we are trying to measure from the mic input.
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        latency: 0
      }
    });
  } catch (err) {
    console.warn('Could not disable echo cancellation, falling back:', err);
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  }

  // 3. Connect the raw mic stream directly into the FFT analyser node,
  //    bypassing any p5.AudioIn processing.
  micSourceNode = audioCtx.createMediaStreamSource(micStream);
  micSourceNode.connect(fft.analyser);

  // 4. Start the oscillator sweep
  osc.start();
  audioStarted = true;

  if (sweepInterval) clearInterval(sweepInterval);
  triggerSweep();
  sweepInterval = setInterval(triggerSweep, SWEEP_DUR_SEC * 1000);
}

function stopAudioEngine() {
  if (sweepInterval) clearInterval(sweepInterval);
  osc.stop();
  if (micSourceNode) { micSourceNode.disconnect(); micSourceNode = null; }
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  audioStarted = false;
}

// ----------------- Flow Management ----------------- //

function showScreen(screenEl) {
  // Hide all
  [screenWelcome, screenCalibrate, screenTrain, screenLive].forEach(s => {
    s.classList.remove('active');
    s.classList.add('hidden');
  });
  // Show target
  screenEl.classList.remove('hidden');
  // Small timeout to allow display:block to render before opacity fade
  setTimeout(() => {
    screenEl.classList.add('active');
  }, 10);
}

function generatePermutations(holes) {
  if (holes === 1) return ['O', 'C'];
  if (holes === 2) return ['OO', 'OC', 'CO', 'CC'];
  if (holes === 3) return ['OOO', 'OOC', 'OCO', 'OCC', 'COO', 'COC', 'CCO', 'CCC'];
  return [];
}

function setupEvents() {
  // Screen 1: Start Wizard
  btnStartWizard.addEventListener('click', () => {
    const holes = parseInt(holeCountSelect.value);
    stateList = generatePermutations(holes);
    
    // Initialize standard ML5 Neural Network
    const options = {
      inputs: 1024,
      task: 'classification',
      debug: false
    };
    nn = ml5.neuralNetwork(options);
    
    // Kickoff Audio
    startAudioEngine();
    
    // Move to Calibration
    currentStateIndex = 0;
    setupNextState();
    showScreen(screenCalibrate);
  });

  // Screen 2: Record Button bindings
  const startBtnAction = (e) => { e.preventDefault(); startRecording(); };
  const stopBtnAction = (e) => { e.preventDefault(); stopRecording(); };

  btnRecordState.addEventListener('mousedown', startBtnAction);
  btnRecordState.addEventListener('touchstart', startBtnAction);
  
  btnRecordState.addEventListener('mouseup', stopBtnAction);
  btnRecordState.addEventListener('mouseleave', stopBtnAction);
  btnRecordState.addEventListener('touchend', stopBtnAction);
  btnRecordState.addEventListener('touchcancel', stopBtnAction);

  // Screen 4: Recalibrate
  btnRecalibrate.addEventListener('click', () => {
    isModelReady = false;
    stopAudioEngine();
    showScreen(screenWelcome);
  });
}

// ----------------- Calibration Logic ----------------- //

function setupNextState() {
  const targetClass = stateList[currentStateIndex];
  targetStateText.textContent = targetClass;
  currentSamples = 0;
  updateProgressUI();
  btnRecordState.disabled = false;
  btnRecordState.textContent = "Hold to Calibrate State";
  progressLabel.textContent = `Awaiting input for ${targetClass}...`;
}

function startRecording() {
  if (isRecording) return;
  isRecording = true;
  btnRecordState.textContent = "Recording... Keep holding!";
  progressLabel.textContent = `Capturing acoustic profile...`;
  
  recordingInterval = setInterval(() => {
    if (!audioStarted || lastSpectrum.length === 0) return;

    // FIX: Use lastSpectrum from draw() — no duplicate fft.analyze() call
    let inputs = lastSpectrum.map(v => v / 255.0);
    let target = [stateList[currentStateIndex]];
    
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
  
  // If they stopped before it was done, prompt them to continue
  if (currentSamples > 0 && currentSamples < SAMPLES_NEEDED) {
    btnRecordState.textContent = "Hold to Resume Calibration";
    progressLabel.textContent = "Please hold until the bar is full!";
  }
}

function updateProgressUI() {
  const percent = Math.min((currentSamples / SAMPLES_NEEDED) * 100, 100);
  captureProgressBar.style.width = `${percent}%`;
}

function handleStateComplete() {
  btnRecordState.disabled = true;
  btnRecordState.textContent = "Great!";
  progressLabel.textContent = "Captured successfully.";
  
  currentStateIndex++;
  
  if (currentStateIndex < stateList.length) {
    // Slight pause before continuing
    setTimeout(() => {
      setupNextState();
    }, 1000);
  } else {
    // All done! Transition to training automatically.
    setTimeout(() => {
      startTraining();
    }, 1000);
  }
}

// ----------------- ML Training ----------------- //

function startTraining() {
  showScreen(screenTrain);
  trainingLogText.textContent = "Normalizing acoustic data...";
  
  nn.normalizeData();
  
  const trainingOptions = {
    epochs: 40,
    batchSize: 12
  };
  
  trainingLogText.textContent = "Optimizing neural weights...";
  nn.train(trainingOptions, whileTraining, finishedTraining);
}

function whileTraining(epoch, loss) {
  trainingLogText.textContent = `Training Epoch ${epoch + 1}/40 (Loss: ${loss.loss.toFixed(3)})`;
}

function finishedTraining() {
  isModelReady = true;
  showScreen(screenLive);
}

// ----------------- ML Inference ----------------- //

function handleClassification(error, results) {
  isClassifying = false;  // FIX: unblock the next classify() call
  if (error || !isModelReady) return;
  
  if (results && results.length > 0) {
    const best = results[0];
    if (best.confidence > 0.5) {
      liveStateText.textContent = best.label;
      liveStateText.style.color = 'var(--text-main)';
    } else {
      liveStateText.textContent = "---";
      liveStateText.style.color = 'var(--text-muted)';
    }
  }
}
