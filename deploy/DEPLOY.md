# Deploying the Dialect Analysis viz

How the viz is hosted on the ZHAW-internal server and how to maintain it. The deployment lives at
`/opt/dialect-analysis` and runs under a dedicated service account, `dialectanalysis`.

## What it is

A Streamlit web app (`visualization/Home.py` plus two pages) for exploring dialect-specific words.
It compares the two ASR transcripts (DAT and DIT) against the Standard German reference, for the
STT4SG-350 and SDS-200 corpora. It reads precomputed word-alignment parquets and transcript TSVs.
Nothing is computed from audio at runtime.

## How it works

- One long-running Streamlit process (Python, serves HTTP and uses WebSockets for live updates).
- Runs under the **`dialectanalysis`** service account (no login shell), which also owns the files.
- A systemd service named `streamlit` runs it: restarts on crash (`Restart=always`) and starts on
  boot (`enable`). It is never started by hand.
- Plain HTTP on port 80, no reverse proxy. Reachable at `http://<server-hostname>` from the ZHAW
  network or VPN (`hostname -f` prints the address). The unit grants `CAP_NET_BIND_SERVICE` so it
  binds port 80 without running as root.
- A single shared password gates the app so the audio isn't freely crawlable (anti-crawl, not
  per-user login). `visualization/_auth.py` reads it from `/opt/dialect-analysis/.streamlit/secrets.toml`.
  Every page is gated, since Streamlit lets you deep-link straight to a page.
- Git auth is a repo-scoped **deploy key** at `/opt/dialect-analysis/.ssh/dialect_deploy`.
- Dependencies live in a uv virtualenv at `/opt/dialect-analysis/.venv`.

## Server layout

Everything is under `/opt/dialect-analysis`, owned by `dialectanalysis`.

| What | Path (under /opt/dialect-analysis) | In git? |
|---|---|---|
| App code | `visualization/` | yes |
| Alignment parquets | `experiments/analysis/{stt4sg,sds-200}/*.parquet` | yes |
| Transcript TSVs | `transcripts/**`, `datasets/STT4SG-350 v2.1/train_balanced.tsv` | yes |
| Audio clips (~41 GB) | `datasets/STT4SG-350 v2.1/clips__*`, `datasets/SDS-200 Corpus/export_20211220_clips-001` | no |
| Python env (uv) | `.venv` | no |
| Password | `.streamlit/secrets.toml` | no |
| Deploy key | `.ssh/dialect_deploy` | no |

The audio, password, and deploy key are not in git; they live only on the server.

## Operations

The repo is owned by `dialectanalysis`, so run git as that account; `systemctl` needs `sudo`.
A handy shell alias: `alias dgit='sudo -u dialectanalysis -H git -C /opt/dialect-analysis'`.

Deploy an update (the common case):
```bash
sudo -u dialectanalysis -H git -C /opt/dialect-analysis pull # or `dgit pull` if alias activated
sudo -u dialectanalysis -H sh -c 'cd /opt/dialect-analysis && uv pip install -r requirements.txt' # only if Python dependencies changed
sudo systemctl restart streamlit
systemctl status streamlit          # expect: active (running)
```

Service control and inspection:
```bash
systemctl status streamlit          # running?
sudo systemctl restart streamlit    # apply changes
sudo systemctl stop streamlit       # take offline
sudo systemctl start streamlit      # bring back
journalctl -u streamlit -e          # recent logs (errors here)
journalctl -u streamlit -f          # follow live
```

Change the password:
```bash
printf 'password = "<new-password>"\n' | sudo -u dialectanalysis tee /opt/dialect-analysis/.streamlit/secrets.toml >/dev/null
sudo chmod 600 /opt/dialect-analysis/.streamlit/secrets.toml
sudo systemctl restart streamlit
```

## First-time setup or rebuild

Only needed on a new server or after a wipe.

1. Service account:
   ```bash
   sudo useradd --system --home-dir /opt/dialect-analysis --shell /usr/sbin/nologin dialectanalysis
   ```
2. Code (public repo, clone over HTTPS for the initial read, then hand it to the account):
   ```bash
   sudo git clone https://github.com/michamettler/BA-FS26_ciel_Mettler_Dialektanalyse.git /opt/dialect-analysis
   sudo chown -R dialectanalysis:dialectanalysis /opt/dialect-analysis
   ```
3. Audio: the corpora are not in git. Stage them under `/opt/dialect-analysis/datasets/` following the
   layout in the parent README's [Datasets](../README.md#datasets) section.
4. Python env (make `uv` available system-wide first if needed, e.g. `sudo cp "$(command -v uv)" /usr/local/bin/uv`):
   ```bash
   sudo -u dialectanalysis -H sh -c 'cd /opt/dialect-analysis && uv venv && uv pip install -r requirements.txt'
   ```
5. Password:
   ```bash
   printf 'password = "<pick>"\n' | sudo -u dialectanalysis tee /opt/dialect-analysis/.streamlit/secrets.toml >/dev/null
   sudo chmod 600 /opt/dialect-analysis/.streamlit/secrets.toml
   ```
6. Deploy key (repo-scoped, write-enabled):
   ```bash
   sudo -u dialectanalysis -H mkdir -p /opt/dialect-analysis/.ssh && sudo -u dialectanalysis chmod 700 /opt/dialect-analysis/.ssh
   sudo -u dialectanalysis -H ssh-keygen -t ed25519 -f /opt/dialect-analysis/.ssh/dialect_deploy -N "" -C "ZHAW dialect-analysis server"
   sudo -u dialectanalysis cat /opt/dialect-analysis/.ssh/dialect_deploy.pub
   ```
   Add that public key on GitHub: repo -> Settings -> Deploy keys -> Add deploy key -> Allow write access. Then:
   ```bash
   sudo -u dialectanalysis git -C /opt/dialect-analysis config core.sshCommand "ssh -i /opt/dialect-analysis/.ssh/dialect_deploy -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
   sudo -u dialectanalysis git -C /opt/dialect-analysis remote set-url origin git@github.com:michamettler/BA-FS26_ciel_Mettler_Dialektanalyse.git
   sudo -u dialectanalysis git -C /opt/dialect-analysis config user.name  "ZHAW Server"
   sudo -u dialectanalysis git -C /opt/dialect-analysis config user.email "noreply@dialectanalysis-bipartitematching.engineering.zhaw.ch"
   ```
7. Service:
   ```bash
   sudo cp /opt/dialect-analysis/deploy/streamlit.service /etc/systemd/system/
   sudo systemctl daemon-reload && sudo systemctl enable --now streamlit
   ```
8. Firewall: open 80 to the internal subnet with `sudo ufw allow 80/tcp` (firewalld:
   `sudo firewall-cmd --add-port=80/tcp --permanent && sudo firewall-cmd --reload`).
9. Check: `curl -sI http://localhost` returns `HTTP/1.1 200`, then open `http://<server-hostname>` in
   a browser and the password prompt shows.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Service won't start (`status` = failed) | `journalctl -u streamlit -e`. Usually a wrong path in the unit, a bind error on port 80 (check `sudo ss -ltnp` for another listener), or a missing venv (rebuild step 4). |
| App loads but shows "Access is not configured" | `/opt/dialect-analysis/.streamlit/secrets.toml` is missing or has no `password`. Recreate it and restart. |
| Browser stuck on "connecting" or reconnect loop | The process was killed (often out of memory) and restarted. Look for `oom` in `journalctl -u streamlit -e` and check `free -h`. |
| "Audio file not found locally" on a clip | That corpus's audio isn't staged at the expected `datasets/...` path. |
| `git pull` says "dubious ownership" | You ran git as the wrong user. Run it as the owner: `sudo -u dialectanalysis -H git -C /opt/dialect-analysis pull`. |
| Slow on the first open after a restart | Expected. The first visit warms an in-memory cache (data load and TF-IDF); later interactions are fast. |
