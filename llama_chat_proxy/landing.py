"""The landing page: hydrate the WebUI's IndexedDB, then hand off.

When a *fresh* browser (no ``lcx-seeded`` cookie) opens the proxy, the server
serves this self-contained HTML page instead of the WebUI.  The page:

1. fetches the stored history from ``/proxy/api/history`` (decrypted server side),
2. writes it into the browser's ``LlamaUi`` IndexedDB (the exact Dexie schema
   the WebUI uses),
3. drops a cookie so the very next request is proxied straight to llama.cpp,
4. reloads into the WebUI, whose sidebar now lists the restored conversations.

Everything is same-origin, no build step, no external assets.
"""

from __future__ import annotations

import json

# IndexedDB database name used by the current llama.cpp WebUI (STORAGE_APP_NAME).
DB_NAME = "LlamaUi"

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>llama-chat-proxy</title>
  <style>
    :root { color-scheme: light dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: radial-gradient(1200px 600px at 50% -10%, #1e293b, #0b1120);
      color: #e2e8f0;
    }
    .card {
      width: min(560px, 92vw); padding: 32px 28px; border-radius: 16px;
      background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
      box-shadow: 0 20px 60px rgba(0,0,0,.45); text-align: center;
    }
    h1 { font-size: 1.15rem; margin: 0 0 6px; letter-spacing: .2px; }
    p.sub { margin: 0 0 22px; opacity: .7; font-size: .9rem; }
    .status { font-variant-numeric: tabular-nums; min-height: 1.2em; font-size: .95rem; }
    .detail { min-height: 1.2em; margin-top: 10px; font-size: .85rem; opacity: .8; }
    .bar {
      height: 6px; border-radius: 999px; margin-top: 22px; overflow: hidden;
      background: rgba(255,255,255,.08);
    }
    .bar > span { display: block; height: 100%; width: 0; border-radius: 999px;
      background: linear-gradient(90deg, #38bdf8, #818cf8); transition: width .4s ease; }
    body.error .status { color: #fca5a5; }
    body.error .bar > span { background: #ef4444; width: 100% !important; }
  </style>
</head>
<body>
  <div class="card">
    <h1>llama-chat-proxy</h1>
    <p class="sub">Restoring your saved chats&hellip;</p>
    <div class="status" id="status" aria-live="polite">Connecting&hellip;</div>
    <div class="detail" id="detail"></div>
    <div class="bar"><span id="bar"></span></div>
  </div>

  <script>
    (function () {
      var HISTORY_URL = __HISTORY_URL__;
      var COOKIE_NAME = __COOKIE__;
      var DB_NAME = __DB_NAME__;
      var statusEl = document.getElementById('status');
      var detailEl = document.getElementById('detail');
      var barEl = document.getElementById('bar');

      function step(label, pct) {
        statusEl.textContent = label;
        barEl.style.width = (pct || 0) + '%';
      }

      function fail(message) {
        document.body.classList.add('error');
        statusEl.textContent = 'Could not restore your chats.';
        detailEl.textContent = message || 'Unknown error';
      }

      function createStores(db) {
        if (!db.objectStoreNames.contains('conversations')) {
          var cs = db.createObjectStore('conversations', { keyPath: 'id' });
          cs.createIndex('lastModified', 'lastModified', { unique: false });
          cs.createIndex('currNode', 'currNode', { unique: false });
          cs.createIndex('name', 'name', { unique: false });
        }
        if (!db.objectStoreNames.contains('messages')) {
          var ms = db.createObjectStore('messages', { keyPath: 'id' });
          ['convId', 'type', 'role', 'timestamp', 'parent', 'children'].forEach(function (k) {
            ms.createIndex(k, k, { unique: false });
          });
        }
      }

      function openDb() {
        return new Promise(function (resolve, reject) {
          var version = 1;
          function doOpen() {
            var req = indexedDB.open(DB_NAME, version);
            req.onupgradeneeded = function (ev) { createStores(ev.target.result); };
            req.onsuccess = function (ev) { resolve(ev.target.result); };
            req.onerror = function () { reject(req.error || new Error('open failed')); };
          }
          if (indexedDB.databases) {
            // Never open below an existing version (that throws VersionError).
            indexedDB.databases().then(function (dbs) {
              var existing = dbs.find(function (d) { return d.name === DB_NAME; });
              if (existing && existing.version > version) version = existing.version;
              doOpen();
            }).catch(doOpen);
          } else {
            doOpen();
          }
        });
      }

      function writeAll(db, items) {
        return new Promise(function (resolve, reject) {
          var tx = db.transaction(['conversations', 'messages'], 'readwrite');
          var cs = tx.objectStore('conversations');
          var ms = tx.objectStore('messages');
          var nConv = 0, nMsg = 0;
          items.forEach(function (item) {
            cs.put(item.conv);
            nConv += 1;
            (item.messages || []).forEach(function (m) { ms.put(m); nMsg += 1; });
          });
          tx.oncomplete = function () { db.close(); resolve({ conversations: nConv, messages: nMsg }); };
          tx.onerror = function () { db.close(); reject(tx.error || new Error('write failed')); };
          tx.onabort = function () { db.close(); reject(tx.error || new Error('transaction aborted')); };
        });
      }

      function finish() {
        // Hand off to the WebUI. We navigate to the root route (which the
        // upstream always serves) and let the WebUI's own sidebar list the
        // restored conversations. The cookie set here is what makes the very
        // next request go straight to llama.cpp instead of back here.
        document.cookie = COOKIE_NAME + '=1; path=/; max-age=31536000; SameSite=Lax';
        step('Ready \u2014 resuming\u2026', 100);
        setTimeout(function () { location.replace('/'); }, 250);
      }

      (async function run() {
        try {
          step('Loading history\u2026', 15);
          var res = await fetch(HISTORY_URL, { credentials: 'same-origin', headers: { 'Accept': 'application/json' } });
          if (!res.ok) throw new Error('history endpoint returned HTTP ' + res.status);
          var data = await res.json();
          var items = Array.isArray(data.conversations) ? data.conversations : [];

          step('Writing to browser storage\u2026', 55);
          var db = await openDb();
          var summary = await writeAll(db, items);
          detailEl.textContent = summary.conversations + ' conversations, ' + summary.messages + ' messages restored.';
          step('Done', 100);
          finish();
        } catch (err) {
          fail(err && err.message ? err.message : String(err));
        }
      })();
    })();
  </script>
</body>
</html>
"""


def landing_html(history_url: str, seed_cookie: str) -> str:
    """Render the landing page with the given history URL and cookie name."""
    html = _TEMPLATE
    html = html.replace("__HISTORY_URL__", json.dumps(history_url))
    html = html.replace("__COOKIE__", json.dumps(seed_cookie))
    html = html.replace("__DB_NAME__", json.dumps(DB_NAME))
    return html
