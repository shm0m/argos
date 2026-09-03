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


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
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

    def event_stream():
        yield from streamer
        thread.join()
        if generated_len["n"] >= MAX_NEW_TOKENS:
            yield TRUNCATION_MARKER

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
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


@app.get("/api/info")
def info():
    return {"model_path": _state["model_path"]}


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

  let history = [];
  let started = false;

  fetch('/api/info').then(r => r.json()).then(d => {
    badge.textContent = d.model_path;
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

  async function submit() {
    const text = input.value.trim();
    if (!text) return;
    enterChatMode();
    input.value = '';
    input.style.height = 'auto';
    send.disabled = true;

    addMessage('user', text);
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
        const truncated = accumulated.includes(TRUNCATION_MARKER);
        const display = truncated ? accumulated.split(TRUNCATION_MARKER)[0] : accumulated;
        setAssistantContent(bubble, display, truncated);
        thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
      }

      const finalText = accumulated.split(TRUNCATION_MARKER)[0];
      history.push({ role: 'user', content: text });
      history.push({ role: 'assistant', content: finalText });
    } catch (e) {
      removeThinking();
      addMessage('assistant', 'Erreur : impossible de contacter le modele.');
    } finally {
      send.disabled = false;
      input.focus();
    }
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
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map=args.device
    )
    model.eval()

    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["device"] = next(model.parameters()).device
    _state["model_path"] = args.model

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
