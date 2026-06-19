# Deploying the Dialect Analysis viz

How the viz is hosted on the ZHAW-internal server and how to maintain it.
`<repo>` is the repository root on the server.

## What it is

A Streamlit web app (`visualization/Home.py` plus two pages) for exploring dialect-specific words.
It compares the two ASR transcripts (DAT and DIT) against the Standard German reference, for the
STT4SG-350 and SDS-200 corpora. It reads precomputed word-alignment parquets and transcript TSVs.
Nothing is computed from audio at runtime.

## How it works

- One long-running Streamlit process (Python, serves HTTP and uses WebSockets for live updates).
- A systemd service named `streamlit` runs it. It restarts on crash (`Restart=always`) and starts
  on boot (`enable`). You never start it by hand.
- Served as plain HTTP on port 80, no reverse proxy. Reachable at `http://<server-hostname>` from
  the ZHAW network or VPN (`hostname -f` prints the address). The unit grants
  `CAP_NET_BIND_SERVICE` so it can bind port 80 while still running as an unprivileged user.
- A single shared password gates the app so the audio isn't freely crawlable. It's not per-user
  login. The gate is `visualization/_auth.py` and reads the password from
  `<repo>/.streamlit/secrets.toml`. Every page is gated, since Streamlit lets you deep-link straight
  to a page.
- Dependencies live in a uv virtualenv at `<repo>/.venv`.

## Server layout

| What | Path | In git? |
|---|---|---|
| App code | `<repo>/visualization/` | yes |
| Alignment parquets | `<repo>/experiments/analysis/{stt4sg,sds-200}/*.parquet` | yes |
| Transcript TSVs | `<repo>/transcripts/**`, `<repo>/datasets/STT4SG-350 v2.1/train_balanced.tsv` | yes |
| Audio clips (~41 GB) | `<repo>/datasets/STT4SG-350 v2.1/clips__*`, `<repo>/datasets/SDS-200 Corpus/export_20211220_clips-001` | no |
| Python env (uv) | `<repo>/.venv` | no |
| Password | `<repo>/.streamlit/secrets.toml` | no |

The audio and the password are not in git; they live only on the server. So a `git pull` updates the
code and the parquets/TSVs, but never the audio or the password.

## Operations

Run on the server over SSH. `systemctl` needs `sudo`.

Deploy an update (the common case):
```bash
cd <repo>
git pull
# only if Python dependencies changed:  uv pip install -r requirements.txt
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
cd <repo>
printf 'password = "<new-password>"\n' > .streamlit/secrets.toml
chmod 600 .streamlit/secrets.toml
sudo systemctl restart streamlit
```

## First-time setup or rebuild

Only needed on a new server or after a wipe.

1. Code: `git clone <repo-url>` (public repo). Brings the code, parquets, and TSVs.
2. Audio: stage the clip directories at the repo-relative paths in the layout table. They're not in
   git, and the app expects them under `<repo>/datasets/...` exactly as listed.
3. Python env: `cd <repo> && uv venv && uv pip install -r requirements.txt`.
4. Password: create `.streamlit/secrets.toml` (see "Change the password").
5. Service: `sudo cp deploy/streamlit.service /etc/systemd/system/`, set `User=` and the paths in the
   unit to match the install location, then
   `sudo systemctl daemon-reload && sudo systemctl enable --now streamlit`.
6. Firewall: open 80 to the internal subnet with `sudo ufw allow 80/tcp` (firewalld:
   `sudo firewall-cmd --add-port=80/tcp --permanent && sudo firewall-cmd --reload`).
7. Check: `curl -sI http://localhost` returns `HTTP/1.1 200`, then open
   `http://<server-hostname>` in a browser and the password prompt shows.