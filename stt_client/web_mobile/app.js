const TARGET_SAMPLE_RATE = 16000;
const WS_PATH = "/ws/stream";
const APP_VERSION = "v20";
const THEME_STORAGE_KEY = "whisper_theme_mode";
const THEME_MODES = new Set(["system", "dark", "light"]);
const THEME_COLORS = {
  dark: "#1c2128",
  light: "#f6f8fa",
};

const PROFILE_CONFIG = {
  realtime: {
    label: "Low latency",
    chunkMs: 80,
    partial: true,
    vadMode: "continue",
    decodeProfile: "realtime",
  },
  balanced: {
    label: "Balanced",
    chunkMs: 120,
    partial: true,
    vadMode: "continue",
    decodeProfile: "realtime",
  },
  saver: {
    label: "Low power/data",
    chunkMs: 240,
    partial: false,
    vadMode: "auto_end",
    decodeProfile: "accurate",
  },
};

const els = {
  appContainer: document.querySelector(".app"),
  inputMode: document.getElementById("inputMode"),
  themeMode: document.getElementById("themeMode"),
  fileField: document.getElementById("fileField"),
  wavFile: document.getElementById("wavFile"),
  chooseWavBtn: document.getElementById("chooseWavBtn"),
  fileHint: document.getElementById("fileHint"),
  networkProfile: document.getElementById("networkProfile"),
  languageCode: document.getElementById("languageCode"),
  wsUrl: document.getElementById("wsUrl"),
  sessionId: document.getElementById("sessionId"),
  realtimePacing: document.getElementById("realtimePacing"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  clearBtn: document.getElementById("clearBtn"),
  transcriptList: document.getElementById("transcriptList"),
  transcriptPlaceholder: document.getElementById("transcriptPlaceholder"),
  logList: document.getElementById("logList"),
  connectionState: document.getElementById("connectionState"),
  captureState: document.getElementById("captureState"),
  transferState: document.getElementById("transferState"),
  resultState: document.getElementById("resultState"),
};

const runtime = {
  ws: null,
  mode: "idle",
  pendingSamples: [],
  chunkSamples: chunkSamplesForProfile("balanced"),
  mediaStream: null,
  audioContext: null,
  sourceNode: null,
  processorNode: null,
  liveLine: null,
  lastFinalText: "",
  hasResult: false,
  filePcm: null,
  fileOffset: 0,
};

let themeModeState = "system";
let systemThemeMedia = null;

function chunkSamplesForProfile(profile) {
  const conf = PROFILE_CONFIG[profile] || PROFILE_CONFIG.balanced;
  return Math.max(1, Math.floor((TARGET_SAMPLE_RATE * conf.chunkMs) / 1000));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function now() {
  return new Date().toISOString().slice(11, 19);
}

function addLog(message, level = "info") {
  const line = document.createElement("div");
  line.className = `log ${level}`;
  line.textContent = `[${now()}] ${message}`;
  els.logList.prepend(line);
  const maxLines = 120;
  while (els.logList.childElementCount > maxLines) {
    els.logList.removeChild(els.logList.lastElementChild);
  }
}

function setState(el, text, kind = "idle") {
  el.textContent = text;
  el.dataset.state = kind;
}

function updateIdleStates() {
  setState(els.connectionState, "Idle", "idle");
  setState(els.captureState, "Idle", "idle");
  setState(els.transferState, "Idle", "idle");
  setState(els.resultState, "Waiting", "idle");
}

function setRunFocusMode(enabled) {
  if (!els.appContainer) {
    return;
  }
  els.appContainer.classList.toggle("is-running", enabled);
}

function setInputModeUI(mode) {
  const isFile = mode === "file";
  els.fileField.classList.toggle("is-collapsed", !isFile);
  els.fileField.setAttribute("aria-hidden", String(!isFile));
  els.realtimePacing.disabled = !isFile;
  els.startBtn.textContent = isFile ? "Send file" : "Start";
}

function getSystemTheme() {
  if (systemThemeMedia) {
    return systemThemeMedia.matches ? "light" : "dark";
  }
  if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
    return "light";
  }
  return "dark";
}

function setThemeColorMeta(resolvedTheme) {
  const meta = document.querySelector('meta[name="theme-color"]');
  if (!meta) {
    return;
  }
  meta.setAttribute("content", THEME_COLORS[resolvedTheme] || THEME_COLORS.dark);
}

function applyThemeMode(mode, persist = true) {
  const nextMode = THEME_MODES.has(mode) ? mode : "system";
  themeModeState = nextMode;

  if (nextMode === "system") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.setAttribute("data-theme", nextMode);
  }

  if (els.themeMode && els.themeMode.value !== nextMode) {
    els.themeMode.value = nextMode;
  }

  const resolvedTheme = nextMode === "system" ? getSystemTheme() : nextMode;
  setThemeColorMeta(resolvedTheme);

  if (persist) {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, nextMode);
    } catch (_error) {
      // no-op
    }
  }
}

function initThemeMode() {
  try {
    const savedMode = localStorage.getItem(THEME_STORAGE_KEY);
    if (savedMode && THEME_MODES.has(savedMode)) {
      themeModeState = savedMode;
    }
  } catch (_error) {
    // no-op
  }

  if (window.matchMedia) {
    systemThemeMedia = window.matchMedia("(prefers-color-scheme: light)");
    const handleThemeChange = () => {
      if (themeModeState === "system") {
        applyThemeMode("system", false);
      }
    };
    if (typeof systemThemeMedia.addEventListener === "function") {
      systemThemeMedia.addEventListener("change", handleThemeChange);
    } else if (typeof systemThemeMedia.addListener === "function") {
      systemThemeMedia.addListener(handleThemeChange);
    }
  }

  applyThemeMode(themeModeState, false);
}

function clearTranscript() {
  runtime.liveLine = null;
  runtime.lastFinalText = "";
  runtime.hasResult = false;
  els.transcriptList.innerHTML = "";
  const placeholder = document.createElement("p");
  placeholder.className = "muted placeholder";
  placeholder.id = "transcriptPlaceholder";
  placeholder.textContent = "Results will appear here.";
  els.transcriptList.appendChild(placeholder);
  els.transcriptPlaceholder = placeholder;
}

function commonPrefixLength(a, b) {
  const aChars = Array.from(a || "");
  const bChars = Array.from(b || "");
  const limit = Math.min(aChars.length, bChars.length);
  let i = 0;
  while (i < limit && aChars[i] === bChars[i]) {
    i += 1;
  }
  return i;
}

function normalizeCompareText(value) {
  return String(value || "")
    .normalize("NFKC")
    .replace(/[\s.,!?;:]+$/g, "")
    .trim()
    .toLowerCase();
}

function extractIncrementalText(currentText, baselineText) {
  const current = String(currentText || "").trim();
  const baseline = String(baselineText || "").trim();
  if (!current) {
    return "";
  }
  if (!baseline) {
    return current;
  }
  if (current.startsWith(baseline)) {
    return current.slice(baseline.length).trimStart();
  }

  const baselineNoPunct = baseline.replace(/[\s.,!?;:]+$/g, "").trim();
  if (baselineNoPunct && current.startsWith(baselineNoPunct)) {
    return current.slice(baselineNoPunct.length).trimStart();
  }

  const currentLower = current.toLowerCase();
  const baselineLower = baseline.toLowerCase();
  if (baselineLower && currentLower.startsWith(baselineLower)) {
    return current.slice(baseline.length).trimStart();
  }

  const baselineNoPunctLower = baselineNoPunct.toLowerCase();
  if (baselineNoPunctLower && currentLower.startsWith(baselineNoPunctLower)) {
    return current.slice(baselineNoPunct.length).trimStart();
  }

  const baselineIndex = current.indexOf(baseline);
  if (baselineIndex >= 0) {
    return current.slice(baselineIndex + baseline.length).trimStart();
  }

  const baselineNoPunctIndex = baselineNoPunct
    ? current.indexOf(baselineNoPunct)
    : -1;
  if (baselineNoPunctIndex >= 0) {
    return current.slice(baselineNoPunctIndex + baselineNoPunct.length).trimStart();
  }

  const baselineLowerIndex = baselineLower
    ? currentLower.indexOf(baselineLower)
    : -1;
  if (baselineLowerIndex >= 0) {
    return current.slice(baselineLowerIndex + baselineLower.length).trimStart();
  }

  const baselineNoPunctLowerIndex = baselineNoPunctLower
    ? currentLower.indexOf(baselineNoPunctLower)
    : -1;
  if (baselineNoPunctLowerIndex >= 0) {
    return current
      .slice(baselineNoPunctLowerIndex + baselineNoPunctLower.length)
      .trimStart();
  }

  const currentNormalized = normalizeCompareText(current);
  const baselineNormalized = normalizeCompareText(baseline);
  if (currentNormalized && baselineNormalized) {
    if (currentNormalized.startsWith(baselineNormalized)) {
      return current.slice(baseline.length).trimStart();
    }
    const normalizedIndex = currentNormalized.indexOf(baselineNormalized);
    if (normalizedIndex >= 0) {
      return current.slice(baseline.length).trimStart();
    }
  }

  const lcp = commonPrefixLength(current, baseline);
  if (lcp > 0) {
    return current.slice(lcp).trimStart();
  }
  return current;
}

function promoteLiveLineToFinal() {
  if (!runtime.liveLine) {
    return false;
  }
  runtime.liveLine.classList.remove("partial");
  runtime.liveLine.classList.add("final");
  const rawText = runtime.liveLine.dataset.rawText;
  if (rawText) {
    runtime.lastFinalText = rawText;
  }
  runtime.liveLine = null;
  return true;
}

function ensureTranscriptStarted() {
  if (els.transcriptPlaceholder && els.transcriptPlaceholder.parentElement) {
    els.transcriptPlaceholder.remove();
  }
}

function renderResult(payload) {
  const hadResultBefore = runtime.hasResult;
  ensureTranscriptStarted();
  runtime.hasResult = true;

  const isFinal = Boolean(payload.is_final);
  const rawText = String(payload.text || "");
  const committedText = String(payload.committed_text || "");
  const unstableText = String(payload.unstable_text || "");

  const sourceText = (rawText || committedText || "").trim();
  let displayCommitted = extractIncrementalText(sourceText, runtime.lastFinalText);
  const showUnstable = !rawText && !isFinal ? unstableText : "";
  const hasRenderableText = Boolean(displayCommitted || showUnstable);
  if (!hasRenderableText) {
    if (isFinal) {
      addLog("Final duplicate skipped (no incremental text).", "warn");
    }
    return;
  }

  const div = runtime.liveLine || document.createElement("div");
  div.className = `line ${isFinal ? "final" : "partial"}`;
  div.dataset.rawText = sourceText;
  div.innerHTML = "";

  const textP = document.createElement("p");
  if (displayCommitted) {
    const stableSpan = document.createElement("span");
    stableSpan.textContent = displayCommitted;
    textP.appendChild(stableSpan);
  }
  if (showUnstable) {
    const unstableSpan = document.createElement("span");
    unstableSpan.className = "unstable";
    unstableSpan.textContent = showUnstable;
    textP.appendChild(unstableSpan);
  }

  const time = document.createElement("small");
  time.className = "meta";
  const lang = String(payload.language_code || "auto");
  const startSec = Number(payload.start_sec || 0).toFixed(2);
  const endSec = Number(payload.end_sec || 0).toFixed(2);
  const langPill = document.createElement("span");
  langPill.className = "lang-pill";
  langPill.textContent = lang.toUpperCase();
  const timeRange = document.createElement("span");
  timeRange.className = "time-range";
  timeRange.textContent = `${startSec}s - ${endSec}s`;
  time.appendChild(langPill);
  time.appendChild(timeRange);

  div.appendChild(textP);
  div.appendChild(time);

  if (!runtime.liveLine) {
    els.transcriptList.prepend(div);
  }
  els.transcriptList.scrollTop = 0;
  if (!hadResultBefore) {
    const transcriptSection = els.transcriptList.closest(".transcript");
    if (transcriptSection) {
      transcriptSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  if (isFinal) {
    addLog(
      `Final text: ${displayCommitted || "(empty)"} | lang=${payload.language_code || "auto"}`
    );
    runtime.lastFinalText = sourceText || runtime.lastFinalText;
    runtime.liveLine = null;
    setState(els.resultState, "Final", "ok");
  } else {
    runtime.liveLine = div;
    setState(els.resultState, "Partial", "active");
  }
}

function downsampleBuffer(input, inputRate, outputRate) {
  if (outputRate >= inputRate) {
    return input;
  }
  const ratio = inputRate / outputRate;
  const newLength = Math.round(input.length / ratio);
  const result = new Float32Array(newLength);
  let inputOffset = 0;
  for (let i = 0; i < newLength; i += 1) {
    const nextOffset = Math.round((i + 1) * ratio);
    let sum = 0;
    let count = 0;
    for (let j = inputOffset; j < nextOffset && j < input.length; j += 1) {
      sum += input[j];
      count += 1;
    }
    result[i] = sum / Math.max(1, count);
    inputOffset = nextOffset;
  }
  return result;
}

function floatToInt16(floatBuffer) {
  const out = new Int16Array(floatBuffer.length);
  for (let i = 0; i < floatBuffer.length; i += 1) {
    const s = Math.max(-1, Math.min(1, floatBuffer[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function enqueueSamples(int16Buffer) {
  for (let i = 0; i < int16Buffer.length; i += 1) {
    runtime.pendingSamples.push(int16Buffer[i]);
  }

  while (runtime.pendingSamples.length >= runtime.chunkSamples) {
    const chunk = runtime.pendingSamples.splice(0, runtime.chunkSamples);
    sendPcmChunk(new Int16Array(chunk));
  }
}

function sendPcmChunk(int16Array) {
  if (!runtime.ws || runtime.ws.readyState !== WebSocket.OPEN) {
    return;
  }
  const byteOffset = int16Array.byteOffset;
  const byteLength = int16Array.byteLength;
  const payload = int16Array.buffer.slice(byteOffset, byteOffset + byteLength);
  runtime.ws.send(payload);
}

function flushPendingSamples() {
  if (!runtime.pendingSamples.length) {
    return;
  }
  sendPcmChunk(new Int16Array(runtime.pendingSamples));
  runtime.pendingSamples = [];
}

function stopCaptureGraph() {
  if (runtime.processorNode) {
    runtime.processorNode.disconnect();
    runtime.processorNode = null;
  }
  if (runtime.sourceNode) {
    runtime.sourceNode.disconnect();
    runtime.sourceNode = null;
  }
  if (runtime.audioContext) {
    runtime.audioContext.close().catch(() => null);
    runtime.audioContext = null;
  }
  if (runtime.mediaStream) {
    runtime.mediaStream.getTracks().forEach((track) => track.stop());
    runtime.mediaStream = null;
  }
}

function closeSocket(sendEnd = true) {
  if (!runtime.ws) {
    return;
  }
  if (sendEnd && runtime.ws.readyState === WebSocket.OPEN) {
    runtime.ws.send(JSON.stringify({ type: "end" }));
  }
  runtime.ws.close();
  runtime.ws = null;
}

function resetRuntime() {
  runtime.mode = "idle";
  runtime.pendingSamples = [];
  runtime.filePcm = null;
  runtime.fileOffset = 0;
}

function finalizeRunFromServer(connectionText = "Done", connectionKind = "ok") {
  stopCaptureGraph();
  if (runtime.ws) {
    try {
      runtime.ws.close();
    } catch (_error) {
      // no-op
    }
    runtime.ws = null;
  }

  setRunFocusMode(false);

  const hadResult = runtime.hasResult;
  resetRuntime();

  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  setState(els.connectionState, connectionText, connectionKind);
  setState(els.captureState, hadResult ? "Completed" : "Idle", hadResult ? "ok" : "idle");
  if (!hadResult) {
    setState(els.resultState, "No result", "warn");
  }
}

function buildSessionPayload() {
  const profileName = els.networkProfile.value;
  const profile = PROFILE_CONFIG[profileName] || PROFILE_CONFIG.balanced;

  const customId = els.sessionId.value.trim();
  const sessionId = customId || `web-${Date.now()}`;
  const languageCode = els.languageCode.value === "auto" ? "" : els.languageCode.value;

  runtime.chunkSamples = chunkSamplesForProfile(profileName);

  return {
    type: "start",
    session_id: sessionId,
    sample_rate: TARGET_SAMPLE_RATE,
    task: "transcribe",
    language_code: languageCode,
    vad_mode: profile.vadMode,
    decode_profile: profile.decodeProfile,
    vad_silence: 0.8,
    vad_threshold: 0.5,
    attributes: {
      partial: profile.partial ? "true" : "false",
      emit_final_on_vad: "true",
    },
  };
}

function defaultWsUrl() {
  const isHttps = window.location.protocol === "https:";
  const host = window.location.hostname || "localhost";
  const hostWithPort = window.location.host || host;
  const fromQuery = new URLSearchParams(window.location.search).get("ws");
  if (fromQuery) {
    return fromQuery;
  }
  if (isHttps) {
    return `wss://${hostWithPort}${WS_PATH}`;
  }
  return `ws://${host}:8001${WS_PATH}`;
}

async function openSession() {
  const wsUrl = els.wsUrl.value.trim();
  if (!wsUrl) {
    throw new Error("WebSocket URL is empty.");
  }

  const payload = buildSessionPayload();
  addLog(
    `Open session ${payload.session_id} | profile=${els.networkProfile.value} (${runtime.chunkSamples} samples/chunk)`
  );

  setState(els.connectionState, "Connecting", "active");

  return new Promise((resolve, reject) => {
    let settled = false;
    let timeoutId = null;
    const ws = new WebSocket(wsUrl);
    runtime.ws = ws;
    ws.binaryType = "arraybuffer";

    const fail = (err) => {
      if (settled) {
        return;
      }
      settled = true;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      reject(err instanceof Error ? err : new Error(String(err)));
    };

    ws.addEventListener("open", () => {
      ws.send(JSON.stringify(payload));
      setState(els.connectionState, "Session start", "active");
    });

    ws.addEventListener("message", (event) => {
      if (typeof event.data !== "string") {
        return;
      }
      let message;
      try {
        message = JSON.parse(event.data);
      } catch (error) {
        addLog(`Invalid JSON from server: ${String(error)}`, "warn");
        return;
      }

      if (message.type === "session") {
        if (!settled) {
          settled = true;
          if (timeoutId) {
            clearTimeout(timeoutId);
          }
          setState(els.connectionState, "Ready", "active");
          resolve(message);
        }
        addLog(`Session ready: ${message.session_id}`);
        return;
      }

      if (message.type === "result") {
        renderResult(message);
        return;
      }

      if (message.type === "done") {
        promoteLiveLineToFinal();
        setState(els.resultState, runtime.hasResult ? "Done" : "No result", "ok");
        setState(els.transferState, "Completed", "ok");
        addLog("Server reported done.");
        finalizeRunFromServer("Done", "ok");
        return;
      }

      if (message.type === "error") {
        const text = message.message || "unknown error";
        setState(els.connectionState, "Error", "error");
        setState(els.resultState, "Error", "error");
        addLog(`Server error: ${text}`, "error");
        if (!settled) {
          fail(new Error(text));
        }
      }
    });

    ws.addEventListener("close", () => {
      if (!settled) {
        fail(new Error("Connection closed before session was ready."));
      }
      if (runtime.mode === "idle") {
        setState(els.connectionState, "Idle", "idle");
      } else {
        promoteLiveLineToFinal();
        addLog("Socket closed by server. Auto-stopping run.", "warn");
        setState(els.transferState, "Closed", "warn");
        finalizeRunFromServer("Closed", "warn");
      }
    });

    ws.addEventListener("error", () => {
      fail(new Error("WebSocket connection failed."));
    });

    timeoutId = setTimeout(() => {
      fail(new Error("Session start timeout."));
      ws.close();
    }, 10000);
  });
}

async function startMicCapture() {
  if (!window.isSecureContext) {
    throw new Error(
      "Microphone is blocked on insecure origin. Use HTTPS (or localhost), or switch to file mode."
    );
  }

  if (
    !navigator.mediaDevices ||
    typeof navigator.mediaDevices.getUserMedia !== "function"
  ) {
    throw new Error(
      "This browser/environment does not expose getUserMedia. Check browser permissions and secure context."
    );
  }

  const mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    video: false,
  });

  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) {
    throw new Error("AudioContext is not supported in this browser.");
  }

  const context = new AudioCtx();
  await context.resume();

  const source = context.createMediaStreamSource(mediaStream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const down = downsampleBuffer(input, context.sampleRate, TARGET_SAMPLE_RATE);
    enqueueSamples(floatToInt16(down));
  };

  source.connect(processor);
  processor.connect(context.destination);

  runtime.mediaStream = mediaStream;
  runtime.audioContext = context;
  runtime.sourceNode = source;
  runtime.processorNode = processor;

  setState(els.captureState, "Recording", "active");
  setState(els.transferState, "Streaming", "active");
}

function mixToMono(audioBuffer) {
  const { numberOfChannels, length } = audioBuffer;
  if (numberOfChannels === 1) {
    return audioBuffer.getChannelData(0);
  }
  const mono = new Float32Array(length);
  for (let channel = 0; channel < numberOfChannels; channel += 1) {
    const data = audioBuffer.getChannelData(channel);
    for (let i = 0; i < length; i += 1) {
      mono[i] += data[i] / numberOfChannels;
    }
  }
  return mono;
}

async function decodeWavToPcm16(file) {
  const bytes = await file.arrayBuffer();
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) {
    throw new Error("AudioContext is not supported in this browser.");
  }
  const context = new AudioCtx();
  try {
    const audioBuffer = await context.decodeAudioData(bytes.slice(0));
    const mono = mixToMono(audioBuffer);
    const down = downsampleBuffer(mono, audioBuffer.sampleRate, TARGET_SAMPLE_RATE);
    return floatToInt16(down);
  } finally {
    await context.close();
  }
}

async function sendFileAudio() {
  if (!runtime.filePcm || !runtime.filePcm.length) {
    throw new Error("No decoded audio samples.");
  }

  const profileName = els.networkProfile.value;
  const profile = PROFILE_CONFIG[profileName] || PROFILE_CONFIG.balanced;
  const chunkSamples = runtime.chunkSamples;
  const useRealtime = els.realtimePacing.checked;

  setState(els.transferState, "Sending", "active");

  while (runtime.fileOffset < runtime.filePcm.length) {
    if (!runtime.ws || runtime.ws.readyState !== WebSocket.OPEN) {
      throw new Error("Socket closed while sending file.");
    }

    const end = Math.min(runtime.fileOffset + chunkSamples, runtime.filePcm.length);
    const chunk = runtime.filePcm.subarray(runtime.fileOffset, end);
    sendPcmChunk(chunk);
    runtime.fileOffset = end;

    const pct = Math.round((runtime.fileOffset / runtime.filePcm.length) * 100);
    setState(els.transferState, `${pct}%`, pct >= 100 ? "ok" : "active");

    if (useRealtime) {
      await sleep(profile.chunkMs);
    } else if (runtime.fileOffset % (chunkSamples * 10) === 0) {
      await sleep(0);
    }
  }

  if (runtime.ws && runtime.ws.readyState === WebSocket.OPEN) {
    runtime.ws.send(JSON.stringify({ type: "end" }));
    setState(els.transferState, "Uploaded", "ok");
  }
}

async function startFlow() {
  if (runtime.mode !== "idle") {
    addLog("Already running. Stop current stream first.", "warn");
    return;
  }

  const mode = els.inputMode.value;
  runtime.mode = mode;
  runtime.pendingSamples = [];
  runtime.fileOffset = 0;
  runtime.lastFinalText = "";
  runtime.liveLine = null;
  setRunFocusMode(true);

  els.startBtn.disabled = true;
  els.stopBtn.disabled = false;
  setState(els.resultState, "Waiting", "idle");

  try {
    if (mode === "file") {
      const file = els.wavFile.files?.[0];
      if (!file) {
        throw new Error("Select a WAV file first.");
      }
      setState(els.captureState, "Decoding", "active");
      addLog(`Decoding file: ${file.name}`);
      runtime.filePcm = await decodeWavToPcm16(file);
      setState(els.captureState, "Ready", "active");
      addLog(`Decoded ${runtime.filePcm.length} samples at 16kHz.`);
    }

    await openSession();

    if (mode === "mic") {
      await startMicCapture();
      addLog("Microphone capture started.");
    } else {
      await sendFileAudio();
      addLog("File transfer completed.");
    }
  } catch (error) {
    addLog(String(error), "error");
    await stopFlow(false);
    setState(els.connectionState, "Error", "error");
    setState(els.captureState, "Error", "error");
    setState(els.transferState, "Error", "error");
  }
}

async function stopFlow(manual = true) {
  if (runtime.mode === "idle") {
    setRunFocusMode(false);
    return;
  }

  flushPendingSamples();
  stopCaptureGraph();
  closeSocket(true);

  if (manual) {
    addLog("Stopped by user.");
  }

  setRunFocusMode(false);
  resetRuntime();

  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  setState(els.captureState, "Stopped", "idle");
  setState(els.transferState, "Stopped", "idle");
  setState(els.connectionState, "Closed", "idle");
}

function bindEvents() {
  els.inputMode.addEventListener("change", () => {
    setInputModeUI(els.inputMode.value);
  });

  if (els.themeMode) {
    els.themeMode.addEventListener("change", () => {
      applyThemeMode(els.themeMode.value);
    });
  }

  els.chooseWavBtn.addEventListener("click", () => {
    els.wavFile.click();
  });

  els.wavFile.addEventListener("change", () => {
    const file = els.wavFile.files?.[0];
    els.fileHint.textContent = file ? `${file.name} (${Math.round(file.size / 1024)}KB)` : "No file selected";
  });

  els.startBtn.addEventListener("click", () => {
    startFlow();
  });

  els.stopBtn.addEventListener("click", () => {
    stopFlow(true);
  });

  els.clearBtn.addEventListener("click", () => {
    clearTranscript();
  });

  window.addEventListener("beforeunload", () => {
    stopCaptureGraph();
    closeSocket(true);
  });
}

async function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    return;
  }
  try {
    await navigator.serviceWorker.register("./sw.js");
    addLog("PWA service worker registered.");
  } catch (error) {
    addLog(`Service worker registration failed: ${String(error)}`, "warn");
  }
}

function init() {
  initThemeMode();
  document.title = `Whisper Web Client ${APP_VERSION}`;
  els.wsUrl.value = defaultWsUrl();
  setRunFocusMode(false);
  setInputModeUI(els.inputMode.value);
  clearTranscript();
  updateIdleStates();
  bindEvents();
  registerServiceWorker();
  addLog(
    `Env secure=${window.isSecureContext} mediaDevices=${Boolean(
      navigator.mediaDevices
    )}`
  );
  if (!window.isSecureContext) {
    addLog("Microphone needs HTTPS or localhost origin.", "warn");
  }
  addLog(`Ready (${APP_VERSION}).`);
}

init();
