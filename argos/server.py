import argparse

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

app = FastAPI()

_state = {}


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str


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
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    reply = tokenizer.decode(output[0, tokens.shape[1]:], skip_special_tokens=True)
    return ChatResponse(reply=reply)


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
<style>
  :root {
    --bg: #f4f5f2;
    --surface: #eaebe5;
    --surface-2: #e1e3db;
    --ink: #1c1f1b;
    --ink-muted: #5b5f55;
    --ink-faint: #8b9082;
    --line: #d6d8d0;
    --gold: #93691a;
    --gold-soft: #a8791e;
    --harm: #a3392d;
    --safe: #26635f;
    --font-display: 'Fraunces', Georgia, serif;
    --font-body: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
    --font-mono: 'IBM Plex Mono', Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14170f;
      --surface: #1c2016;
      --surface-2: #262b1d;
      --ink: #e9eae3;
      --ink-muted: #a3a897;
      --ink-faint: #6f7566;
      --line: #2c3123;
      --gold: #d6a64c;
      --gold-soft: #c99444;
      --harm: #e2604f;
      --safe: #5cbdb7;
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
    color: var(--gold-soft);
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
    background: var(--safe);
    margin-right: 0.5rem;
  }

  main {
    flex: 1;
    overflow-y: auto;
    display: flex;
    justify-content: center;
  }
  #thread {
    width: 100%;
    max-width: 760px;
    padding: 1.6rem 1.4rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
  }

  .empty-state {
    margin: auto;
    text-align: center;
    color: var(--ink-faint);
    max-width: 32ch;
    font-size: 0.95rem;
    line-height: 1.6;
  }
  .empty-state strong { color: var(--ink-muted); font-family: var(--font-display); font-style: italic; font-weight: 500; }

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
  .msg.assistant .avatar { background: var(--gold-soft); color: var(--bg); }

  .bubble {
    padding: 0.75rem 1rem;
    border-radius: 12px;
    line-height: 1.55;
    font-size: 0.96rem;
    white-space: pre-wrap;
  }
  .msg.user .bubble { background: var(--surface-2); border-top-right-radius: 3px; }
  .msg.assistant .bubble { background: var(--surface); border-top-left-radius: 3px; border: 1px solid var(--line); }

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
    border: 1px solid var(--line);
    border-radius: 10px;
    background: var(--surface);
    color: var(--ink);
    font-family: var(--font-body);
    font-size: 0.96rem;
    padding: 0.7rem 0.9rem;
    max-height: 8rem;
    line-height: 1.5;
  }
  #input:focus { outline: 2px solid var(--gold-soft); outline-offset: 1px; }
  #send {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    background: var(--gold-soft);
    color: var(--bg);
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.1rem;
    cursor: pointer;
    transition: opacity 0.15s;
  }
  #send:hover { opacity: 0.88; }
  #send:disabled { opacity: 0.4; cursor: default; }
  #send:focus-visible { outline: 2px solid var(--gold-soft); outline-offset: 2px; }

  ::-webkit-scrollbar { width: 10px; }
  ::-webkit-scrollbar-thumb { background: var(--line); border-radius: 6px; }
</style>
</head>
<body>

<header>
  <div class="brand">
    <h1>ARGOS</h1>
    <span class="tag">demo</span>
  </div>
  <div id="model-badge">chargement...</div>
</header>

<main>
  <div id="thread">
    <div class="empty-state" id="empty-state">
      Pose une question. Ce modele a subi une <strong>ablation de sa direction de refus</strong>,
      compare sa reponse a ce qu'un modele instruct standard aurait dit.
    </div>
  </div>
</main>

<footer>
  <div id="composer">
    <textarea id="input" placeholder="Ecris un message..." rows="1"></textarea>
    <button id="send">Envoyer</button>
  </div>
</footer>

<script>
  const thread = document.getElementById('thread');
  const emptyState = document.getElementById('empty-state');
  const input = document.getElementById('input');
  const send = document.getElementById('send');
  const badge = document.getElementById('model-badge');

  let history = [];

  fetch('/api/info').then(r => r.json()).then(d => {
    badge.textContent = d.model_path;
  }).catch(() => { badge.textContent = 'modele indisponible'; });

  function addMessage(role, text) {
    if (emptyState) emptyState.remove();
    const msg = document.createElement('div');
    msg.className = 'msg ' + role;
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? 'moi' : 'AI';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
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

  async function submit() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    input.style.height = 'auto';
    send.disabled = true;

    addMessage('user', text);
    addThinking();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, history })
      });
      const data = await res.json();
      removeThinking();
      addMessage('assistant', data.reply);
      history.push({ role: 'user', content: text });
      history.push({ role: 'assistant', content: data.reply });
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
