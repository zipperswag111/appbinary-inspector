"""Local browser chat UI for a trained CardStack model."""
from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .cardstack import CardStackLM

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CardStack Chat</title><style>
:root{--bg:#0a0d13;--panel:#121823;--line:#263142;--text:#f1f5f9;--muted:#9dacbf;--accent:#67e8f9;--user:#193352}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px system-ui,-apple-system,Segoe UI,sans-serif}.shell{max-width:920px;min-height:100vh;margin:auto;padding:28px 18px;display:flex;flex-direction:column;gap:20px}header{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid var(--line);padding-bottom:18px}h1{font-size:22px;margin:0 0 5px}p{margin:0;color:var(--muted)}.badge{font-size:12px;color:var(--accent);border:1px solid #1d6976;border-radius:999px;padding:6px 9px}.chat{flex:1;display:flex;flex-direction:column;gap:14px}.empty{margin:auto;text-align:center;max-width:470px}.empty h2{font-size:24px;margin-bottom:8px}.message{max-width:84%;padding:13px 15px;border:1px solid var(--line);border-radius:14px;white-space:pre-wrap;line-height:1.45}.assistant{align-self:flex-start;background:var(--panel)}.user{align-self:flex-end;background:var(--user)}form{display:flex;gap:10px;position:sticky;bottom:0;background:var(--bg);padding-top:12px}textarea{resize:none;min-height:54px;max-height:180px;flex:1;border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:12px;padding:15px;font:inherit}button{border:0;border-radius:12px;padding:0 18px;background:var(--accent);color:#062027;font-weight:700;cursor:pointer}button:disabled{opacity:.55;cursor:wait}.hint{font-size:12px;color:var(--muted);text-align:center}@media(max-width:560px){.shell{padding:18px 12px}.badge{display:none}.message{max-width:94%}button{padding:0 14px}}</style></head><body><main class="shell"><header><div><h1>CardStack Chat</h1><p>Disk-paged character model · cards load only when routed</p></div><span class="badge">LOCAL ONLY</span></header><section class="chat" id="chat"><div class="empty" id="empty"><h2>Start a conversation</h2><p>This model produces one next character at a time. Quality depends on the text used to train its cards.</p></div></section><form id="form"><textarea id="prompt" aria-label="Message" placeholder="Type a message" required></textarea><button id="send" type="submit">Send</button></form><div class="hint">Enter to send · Shift+Enter for a new line</div></main><script>const chat=document.querySelector('#chat'),empty=document.querySelector('#empty'),form=document.querySelector('#form'),prompt=document.querySelector('#prompt'),send=document.querySelector('#send');function add(kind,text){empty?.remove();const el=document.createElement('div');el.className='message '+kind;el.textContent=text;chat.append(el);el.scrollIntoView({block:'end',behavior:'smooth'});return el}form.addEventListener('submit',async e=>{e.preventDefault();const text=prompt.value.trim();if(!text)return;add('user',text);prompt.value='';send.disabled=true;const reply=add('assistant','Thinking…');try{const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:text,characters:180})});const body=await r.json();reply.textContent=body.text||body.error||'No response.'}catch(err){reply.textContent='Could not reach the local model: '+err.message}finally{send.disabled=false;prompt.focus()}});prompt.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();form.requestSubmit()}});prompt.focus();</script></body></html>"""


def serve_chat(model_directory: str | Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Serve a local-only chat box for a disk-paged CardStack model."""
    model = CardStackLM.load(model_directory)
    model_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path != "/":
                self._send(HTTPStatus.NOT_FOUND, "text/plain", "Not found")
                return
            self._send(HTTPStatus.OK, "text/html", PAGE)

        def do_POST(self) -> None:
            if self.path != "/api/generate":
                self._send(HTTPStatus.NOT_FOUND, "application/json", '{"error":"Not found"}')
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(min(size, 20_000)))
                prompt = str(request.get("prompt", "")).strip()
                characters = max(1, min(int(request.get("characters", 180)), 500))
                if not prompt:
                    raise ValueError("Write a message first.")
                with model_lock:
                    text = model.generate(prompt, new_characters=characters)
                self._send(HTTPStatus.OK, "application/json", json.dumps({"text": text}))
            except Exception as error:
                self._send(HTTPStatus.BAD_REQUEST, "application/json", json.dumps({"error": str(error)}))

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    address = f"http://{host}:{port}"
    if open_browser:
        webbrowser.open(address)
    print(f"CardStack Chat is running at {address}. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
