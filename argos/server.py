import argparse
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    TextIteratorStreamer,
)

app = FastAPI()

_state = {}


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str
    truncated: bool = False


MAX_NEW_TOKENS = 2048
TRUNCATION_MARKER = "<<<TRUNCATED>>>"
SECTION_ABLATED = "\x01ABLATED\x01"
SECTION_BASELINE = "\x01BASELINE\x01"
SECTION_LOADING = "\x01LOADING\x01"
SECTION_END = "\x01END\x01"


def load_model(path, dtype, device):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=dtype, device_map=device)
    model.eval()
    return model, tokenizer


def ensure_active(which):
    if _state.get("active") == which:
        return
    path = _state["ablated_path"] if which == "ablated" else _state["baseline_path"]

    if "model" in _state:
        del _state["model"]
        torch.cuda.empty_cache()

    model, tokenizer = load_model(path, _state["dtype"], _state["device_arg"])
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["device"] = next(model.parameters()).device
    _state["active"] = which


def generate_stream(messages):
    model = _state["model"]
    tokenizer = _state["tokenizer"]
    device = _state["device"]

    tokens = tokenizer.apply_chat_template(
        [messages],
        padding=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids.to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generated_len = {"n": 0}

    def run_generation():
        with torch.no_grad():
            output = model.generate(
                tokens,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer,
            )
        generated_len["n"] = output.shape[1] - tokens.shape[1]

    thread = Thread(target=run_generation)
    thread.start()

    yield from streamer
    thread.join()
    if generated_len["n"] >= MAX_NEW_TOKENS:
        yield TRUNCATION_MARKER


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    ensure_active("ablated")
    messages = list(req.history) + [{"role": "user", "content": req.message}]
    return StreamingResponse(generate_stream(messages), media_type="text/plain; charset=utf-8")


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    ensure_active("ablated")
    model = _state["model"]
    tokenizer = _state["tokenizer"]
    device = _state["device"]

    messages = list(req.history) + [{"role": "user", "content": req.message}]

    tokens = tokenizer.apply_chat_template(
        [messages],
        padding=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids.to(device)

    with torch.no_grad():
        output = model.generate(
            tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output[0, tokens.shape[1]:]
    reply = tokenizer.decode(generated, skip_special_tokens=True)
    truncated = generated.shape[0] >= MAX_NEW_TOKENS
    return ChatResponse(reply=reply, truncated=truncated)


@app.post("/api/compare/stream")
def compare_stream(req: ChatRequest):
    if not _state.get("baseline_path"):
        return StreamingResponse(iter([""]), media_type="text/plain; charset=utf-8", status_code=400)

    messages = list(req.history) + [{"role": "user", "content": req.message}]

    def event_stream():
        ensure_active("ablated")
        yield SECTION_ABLATED
        yield from generate_stream(messages)
        yield SECTION_END

        yield SECTION_LOADING
        ensure_active("baseline")
        yield SECTION_BASELINE
        yield from generate_stream(messages)
        yield SECTION_END

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")


@app.get("/api/info")
def info():
    return {"model_path": _state["ablated_path"], "baseline_path": _state.get("baseline_path")}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARGOS</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,700;1,9..144,500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.1.5/purify.min.js"></script>
<style>
  :root {
    --bg: #f5f5f6;
    --surface: #e9e9eb;
    --surface-2: #dddee1;
    --ink: #18191b;
    --ink-muted: #56575c;
    --ink-faint: #8b8c91;
    --line: #d1d2d6;
    --accent: #7c3aed;
    --online: #1f9d55;
    --font-display: 'Fraunces', Georgia, serif;
    --font-body: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
    --font-mono: 'IBM Plex Mono', Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0e0f10;
      --surface: #1a1b1d;
      --surface-2: #232427;
      --ink: #ededee;
      --ink-muted: #9a9ba0;
      --ink-faint: #6a6b70;
      --line: #2a2b2e;
      --accent: #b388ff;
      --online: #3ddc73;
    }
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: var(--font-body);
    display: flex;
    flex-direction: column;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 1.6rem;
    border-bottom: 1px solid var(--line);
    flex-shrink: 0;
  }
  header[hidden] { display: none; }
  .brand { display: flex; align-items: baseline; gap: 0.7rem; }
  .brand h1 {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 1.4rem;
    margin: 0;
    letter-spacing: -0.01em;
  }
  .brand .tag {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
  }
  #model-badge {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--ink-muted);
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 100px;
    padding: 0.35rem 0.8rem;
    max-width: 46vw;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  #model-badge::before {
    content: "";
    display: inline-block;
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: var(--online);
    margin-right: 0.5rem;
  }
  #compare-toggle {
    position: fixed;
    top: 1.1rem;
    right: 1.6rem;
    z-index: 10;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--ink-muted);
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 100px;
    padding: 0.4rem 0.9rem 0.4rem 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    user-select: none;
  }
  #compare-toggle[hidden] { display: none; }
  #compare-toggle input { accent-color: var(--accent); cursor: pointer; }
  #compare-toggle.in-header { position: static; }

  .compare-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    width: 100%;
  }
  .compare-col { min-width: 0; }
  .compare-col .compare-label {
    font-family: var(--font-mono);
    font-size: 0.74rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin-bottom: 0.5rem;
  }
  .compare-col.is-ablated .compare-label { color: var(--accent); }
  .compare-col .bubble {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    width: 100%;
  }
  .compare-loading {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--ink-faint);
    font-style: italic;
  }
  @media (max-width: 640px) {
    .compare-row { grid-template-columns: 1fr; }
  }

  main {
    flex: 1;
    overflow-y: auto;
    display: flex;
    justify-content: center;
  }

  /* Accueil centre (avant le premier message) */
  #hero {
    margin: auto;
    width: 100%;
    max-width: 640px;
    padding: 2rem 1.4rem;
    text-align: center;
  }
  #hero[hidden] { display: none; }
  #hero .wordmark {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: clamp(2.6rem, 9vw, 4rem);
    letter-spacing: -0.02em;
    margin: 0;
  }
  #hero .tagline {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 500;
    color: var(--ink-muted);
    font-size: 1.05rem;
    margin: 0.6rem 0 2.2rem;
  }
  #hero-pill {
    display: flex;
    align-items: flex-end;
    gap: 0.6rem;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 0.9rem 0.9rem 0.9rem 1.3rem;
    text-align: left;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    justify-content: center;
    margin-top: 1.3rem;
  }
  .chip {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--ink-muted);
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 100px;
    padding: 0.45rem 0.9rem;
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
  }
  .chip:hover { border-color: var(--accent); color: var(--ink); }
  .chip:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  #thread {
    width: 100%;
    max-width: 760px;
    padding: 1.6rem 1.4rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
  }
  #thread[hidden] { display: none; }

  .msg { display: flex; gap: 0.7rem; max-width: 88%; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg.assistant { align-self: flex-start; }

  .avatar {
    flex-shrink: 0;
    width: 1.9rem;
    height: 1.9rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 500;
    margin-top: 0.15rem;
  }
  .msg.user .avatar { background: var(--surface-2); color: var(--ink-muted); }
  .msg.assistant .avatar { background: var(--accent); color: var(--bg); }

  .bubble {
    padding: 0.75rem 1rem;
    border-radius: 12px;
    line-height: 1.55;
    font-size: 0.96rem;
  }
  .msg.user .bubble { background: var(--surface-2); border-top-right-radius: 3px; white-space: pre-wrap; }
  .msg.assistant .bubble { background: var(--surface); border-top-left-radius: 3px; border: 1px solid var(--line); }

  /* Rendu du markdown dans les reponses du modele */
  .bubble > *:first-child { margin-top: 0; }
  .bubble > *:last-child { margin-bottom: 0; }
  .bubble p { margin: 0 0 0.7em; }
  .bubble ul, .bubble ol { margin: 0 0 0.7em; padding-left: 1.4em; }
  .bubble li { margin-bottom: 0.25em; }
  .bubble li > p { margin: 0; }
  .bubble h1, .bubble h2, .bubble h3 { font-family: var(--font-display); font-weight: 600; line-height: 1.3; margin: 0.9em 0 0.4em; }
  .bubble h1 { font-size: 1.25rem; }
  .bubble h2 { font-size: 1.12rem; }
  .bubble h3 { font-size: 1.02rem; }
  .bubble strong { font-weight: 600; color: var(--ink); }
  .bubble a { color: var(--accent); }
  .bubble code {
    font-family: var(--font-mono);
    font-size: 0.87em;
    background: var(--surface-2);
    padding: 0.1em 0.4em;
    border-radius: 3px;
  }
  .bubble pre {
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.8em 1em;
    overflow-x: auto;
    margin: 0 0 0.7em;
  }
  .bubble pre code { background: none; padding: 0; }
  .bubble blockquote {
    border-left: 3px solid var(--accent);
    margin: 0 0 0.7em;
    padding: 0.1em 0 0.1em 1em;
    color: var(--ink-muted);
  }
  .truncated-note {
    font-family: var(--font-mono);
    font-size: 0.76rem;
    color: var(--accent);
    margin-top: 0.6em;
    padding-top: 0.6em;
    border-top: 1px dashed var(--line);
  }

  .thinking .bubble { color: var(--ink-faint); font-style: italic; }
  .dots span {
    display: inline-block;
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--ink-faint);
    margin-right: 3px;
    animation: pulse 1.2s infinite ease-in-out;
  }
  .dots span:nth-child(2) { animation-delay: 0.15s; }
  .dots span:nth-child(3) { animation-delay: 0.3s; }
  @keyframes pulse { 0%, 80%, 100% { opacity: 0.25; } 40% { opacity: 1; } }
  @media (prefers-reduced-motion: reduce) { .dots span { animation: none; opacity: 0.6; } }

  footer {
    flex-shrink: 0;
    border-top: 1px solid var(--line);
    padding: 1rem 1.4rem 1.3rem;
    display: flex;
    justify-content: center;
  }
  footer[hidden] { display: none; }
  #composer {
    width: 100%;
    max-width: 760px;
    display: flex;
    gap: 0.7rem;
    align-items: flex-end;
  }
  #input {
    flex: 1;
    resize: none;
    border: none;
    background: transparent;
    color: var(--ink);
    font-family: var(--font-body);
    font-size: 0.96rem;
    padding: 0.35rem 0;
    max-height: 8rem;
    line-height: 1.5;
  }
  #input:focus { outline: none; }
  #composer:focus-within { outline: none; }
  footer #composer, #hero-pill { border: 1px solid var(--line); border-radius: 14px; background: var(--surface); padding: 0.6rem 0.7rem 0.6rem 1rem; }
  footer #composer:focus-within, #hero-pill:focus-within { border-color: var(--accent); }
  #send {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    background: var(--accent);
    color: var(--bg);
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    cursor: pointer;
    transition: opacity 0.15s;
    flex-shrink: 0;
  }
  #send:hover { opacity: 0.88; }
  #send:disabled { opacity: 0.4; cursor: default; }
  #send:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  ::-webkit-scrollbar { width: 10px; }
  ::-webkit-scrollbar-thumb { background: var(--line); border-radius: 6px; }
</style>
</head>
<body>

<label id="compare-toggle" hidden>
  <input type="checkbox" id="compare-checkbox">
  Comparer avec l'original
</label>

<header id="header" hidden>
  <div class="brand">
    <h1>ARGOS</h1>
    <span class="tag">demo</span>
  </div>
  <div id="model-badge">chargement...</div>
</header>

<main>
  <div id="hero">
    <p class="wordmark">ARGOS</p>
    <p class="tagline">Ce modele a subi une ablation de sa direction de refus. Compare sa reponse a ce qu'un modele instruct standard aurait dit.</p>
    <div id="hero-pill">
      <textarea id="input" placeholder="Pose n'importe quelle question..." rows="1"></textarea>
      <button id="send">Envoyer</button>
    </div>
    <div class="chips">
      <button class="chip" data-prompt="Explique en une phrase ce qu'est l'ablation de la direction de refus.">Explique le mecanisme</button>
      <button class="chip" data-prompt="Un premier train part a 14h a 80 km/h. Un second part a 15h a 100 km/h dans la meme direction, sur la meme voie. A quelle heure le second rattrape-t-il le premier ?">Raisonnement</button>
      <button class="chip" data-prompt="Dans un cadre pedagogique de serrurerie, explique le principe du crochetage d'une serrure a pene simple.">Demande sensible</button>
      <button class="chip" data-prompt="Ecris un court poeme sur un systeme qui refuse d'obeir.">Ecriture creative</button>
    </div>
  </div>
  <div id="thread" hidden></div>
</main>

<footer id="footer" hidden>
  <div id="composer"></div>
</footer>

<script>
  const header = document.getElementById('header');
  const hero = document.getElementById('hero');
  const thread = document.getElementById('thread');
  const footer = document.getElementById('footer');
  const composerFooter = document.getElementById('composer');
  const heroPill = document.getElementById('hero-pill');
  const input = document.getElementById('input');
  const send = document.getElementById('send');
  const badge = document.getElementById('model-badge');
  const compareToggle = document.getElementById('compare-toggle');
  const compareCheckbox = document.getElementById('compare-checkbox');

  let history = [];
  let baselineHistory = [];
  let started = false;
  let hasBaseline = false;

  fetch('/api/info').then(r => r.json()).then(d => {
    badge.textContent = d.model_path;
    if (d.baseline_path) {
      hasBaseline = true;
      compareToggle.hidden = false;
      compareToggle.title = 'Original : ' + d.baseline_path;
    }
  }).catch(() => { badge.textContent = 'modele indisponible'; });

  function enterChatMode() {
    if (started) return;
    started = true;
    hero.hidden = true;
    header.hidden = false;
    thread.hidden = false;
    footer.hidden = false;
    composerFooter.appendChild(input);
    composerFooter.appendChild(send);
    compareToggle.classList.add('in-header');
    header.insertBefore(compareToggle, badge);
  }

  function renderMarkdown(text) {
    const html = marked.parse(text, { breaks: true });
    return DOMPurify.sanitize(html);
  }

  function setAssistantContent(bubble, text, truncated) {
    bubble.innerHTML = renderMarkdown(text);
    if (truncated) {
      const note = document.createElement('div');
      note.className = 'truncated-note';
      note.textContent = 'reponse tronquee, limite de longueur atteinte';
      bubble.appendChild(note);
    }
  }

  function addMessage(role, text) {
    const msg = document.createElement('div');
    msg.className = 'msg ' + role;
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? 'moi' : 'AI';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    if (role === 'assistant') {
      setAssistantContent(bubble, text, false);
    } else {
      bubble.textContent = text;
    }
    msg.appendChild(avatar);
    msg.appendChild(bubble);
    thread.appendChild(msg);
    thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
    return bubble;
  }

  function addCompareRow() {
    const msg = document.createElement('div');
    msg.className = 'msg assistant';
    msg.style.maxWidth = '100%';
    msg.style.width = '100%';
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'AI';
    const row = document.createElement('div');
    row.className = 'compare-row';

    function makeCol(label, isAblated) {
      const col = document.createElement('div');
      col.className = 'compare-col' + (isAblated ? ' is-ablated' : '');
      const lbl = document.createElement('div');
      lbl.className = 'compare-label';
      lbl.textContent = label;
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.innerHTML = '<span class="compare-loading">en attente...</span>';
      col.appendChild(lbl);
      col.appendChild(bubble);
      return { col, bubble };
    }

    const baseline = makeCol('Original (non ablate)', false);
    const ablated = makeCol('Ablate (ARGOS)', true);
    row.appendChild(baseline.col);
    row.appendChild(ablated.col);
    msg.appendChild(avatar);
    msg.appendChild(row);
    thread.appendChild(msg);
    thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
    return { baselineBubble: baseline.bubble, ablatedBubble: ablated.bubble };
  }

  function addThinking() {
    const msg = document.createElement('div');
    msg.className = 'msg assistant thinking';
    msg.id = 'thinking';
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'AI';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = '<span class="dots"><span></span><span></span><span></span></span>';
    msg.appendChild(avatar);
    msg.appendChild(bubble);
    thread.appendChild(msg);
    thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
  }

  function removeThinking() {
    const el = document.getElementById('thinking');
    if (el) el.remove();
  }

  const TRUNCATION_MARKER = '<<<TRUNCATED>>>';
  const SECTION_ABLATED = '\x01ABLATED\x01';
  const SECTION_BASELINE = '\x01BASELINE\x01';
  const SECTION_LOADING = '\x01LOADING\x01';
  const SECTION_END = '\x01END\x01';

  function splitTruncated(raw) {
    const truncated = raw.includes(TRUNCATION_MARKER);
    const display = truncated ? raw.split(TRUNCATION_MARKER)[0] : raw;
    return { display, truncated };
  }

  function extractSection(raw, startMarker, endMarkers) {
    const startIdx = raw.indexOf(startMarker);
    if (startIdx === -1) return null;
    const contentStart = startIdx + startMarker.length;
    let endIdx = raw.length;
    for (const em of endMarkers) {
      const idx = raw.indexOf(em, contentStart);
      if (idx !== -1 && idx < endIdx) endIdx = idx;
    }
    return raw.slice(contentStart, endIdx);
  }

  async function submitSingle(text) {
    addThinking();
    let bubble = null;
    let accumulated = '';

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history })
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!bubble) {
          removeThinking();
          bubble = addMessage('assistant', '');
        }
        accumulated += decoder.decode(value, { stream: true });
        const { display, truncated } = splitTruncated(accumulated);
        setAssistantContent(bubble, display, truncated);
        thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
      }

      const finalText = splitTruncated(accumulated).display;
      history.push({ role: 'user', content: text });
      history.push({ role: 'assistant', content: finalText });
    } catch (e) {
      removeThinking();
      addMessage('assistant', 'Erreur : impossible de contacter le modele.');
    }
  }

  async function submitCompare(text) {
    const { baselineBubble, ablatedBubble } = addCompareRow();
    let accumulated = '';

    try {
      const res = await fetch('/api/compare/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history: baselineHistory })
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        accumulated += decoder.decode(value, { stream: true });

        const ablatedRaw = extractSection(accumulated, SECTION_ABLATED, [SECTION_END]);
        if (ablatedRaw !== null) {
          const { display, truncated } = splitTruncated(ablatedRaw);
          setAssistantContent(ablatedBubble, display || '...', truncated);
        }

        if (accumulated.includes(SECTION_LOADING) && !accumulated.includes(SECTION_BASELINE)) {
          baselineBubble.innerHTML = '<span class="compare-loading">chargement du modele original...</span>';
        }

        const baselineRaw = extractSection(accumulated, SECTION_BASELINE, [SECTION_END]);
        if (baselineRaw !== null) {
          const { display, truncated } = splitTruncated(baselineRaw);
          setAssistantContent(baselineBubble, display || '...', truncated);
        }

        thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
      }

      const ablatedFinal = splitTruncated(extractSection(accumulated, SECTION_ABLATED, [SECTION_END]) || '').display;
      const baselineFinal = splitTruncated(extractSection(accumulated, SECTION_BASELINE, [SECTION_END]) || '').display;
      history.push({ role: 'user', content: text });
      history.push({ role: 'assistant', content: ablatedFinal });
      baselineHistory.push({ role: 'user', content: text });
      baselineHistory.push({ role: 'assistant', content: baselineFinal });
    } catch (e) {
      ablatedBubble.innerHTML = renderMarkdown('Erreur : impossible de contacter le modele.');
      baselineBubble.innerHTML = renderMarkdown('Erreur : impossible de contacter le modele.');
    }
  }

  async function submit() {
    const text = input.value.trim();
    if (!text) return;
    enterChatMode();
    input.value = '';
    input.style.height = 'auto';
    send.disabled = true;

    addMessage('user', text);

    if (hasBaseline && compareCheckbox.checked) {
      await submitCompare(text);
    } else {
      await submitSingle(text);
    }

    send.disabled = false;
    input.focus();
  }

  send.addEventListener('click', submit);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 128) + 'px';
  });

  document.querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      input.value = chip.dataset.prompt;
      input.focus();
      input.dispatchEvent(new Event('input'));
    });
  });
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(prog="argos-server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    _state["ablated_path"] = args.model
    _state["baseline_path"] = args.baseline
    _state["dtype"] = getattr(torch, args.dtype)
    _state["device_arg"] = args.device

    ensure_active("ablated")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
