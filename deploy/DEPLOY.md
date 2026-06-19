# Deployment runbook — Streamlit viz (ZHAW-internal, HTTP, port 8501)

Hosts the dialect-analysis viz (`visualization/Home.py`) on the internal network, reachable in a
browser at `http://<server-hostname>:8501`. Plain HTTP, no reverse proxy. Access is gated behind a
shared password (`visualization/_auth.py`, read from `.streamlit/secrets.toml`).

Assumes the server already has: SSH + sudo, the public repo cloned, a working `uv` environment, and
the audio staged at `<repo>/datasets/…` (same layout as the repo). Everything else the viz reads
(alignment parquets, metadata TSVs, `train_balanced.tsv`) is git-tracked and arrives with the clone.

## Steps

1. **Update code** — in the existing clone:
   ```bash
   cd <repo-path-on-server>
   git pull
   ```
   If new dependencies ever appear (none for this change — the gate uses stdlib `hmac`):
   `uv pip install -r <whatever the project uses>`.

2. **Create the password file** (never committed; `.streamlit/secrets.toml` is gitignored):
   ```bash
   mkdir -p .streamlit
   printf 'password = "<pick-a-password>"\n' > .streamlit/secrets.toml
   chmod 600 .streamlit/secrets.toml
   ```
   (Template: `.streamlit/secrets.toml.example`.)

3. **Install the systemd service**:
   ```bash
   # find the streamlit binary in the uv env:
   uv run which streamlit          # e.g. <repo>/.venv/bin/streamlit
   sudo cp deploy/streamlit.service /etc/systemd/system/streamlit.service
   sudoedit /etc/systemd/system/streamlit.service   # fill <deploy-user>, <repo-path-on-server>, <uv-env>
   sudo systemctl daemon-reload
   sudo systemctl enable --now streamlit
   systemctl status streamlit       # should be active (running)
   ```

4. **Open the firewall** for the internal subnet:
   ```bash
   sudo ufw status                  # or: firewall-cmd --state
   sudo ufw allow 8501/tcp          # ufw; for firewalld: firewall-cmd --add-port=8501/tcp --permanent && firewall-cmd --reload
   ```

5. **Verify**:
   ```bash
   curl -sI http://localhost:8501   # expect HTTP/1.1 200
   ```
   Then from a campus machine open `http://<server-hostname>:8501` → password prompt → app. Deep-link
   to `/Dialect_Word_Lexicon` and confirm it also prompts. Play an SDS-200 example clip (audio resolves
   `.flac`→`.mp3`). Logs: `journalctl -u streamlit -f`.

6. **Reboot test** (optional): `sudo reboot`, then confirm `systemctl status streamlit` is active —
   `enable` makes it start on boot.

## Operations

- Restart after a change: `git pull && sudo systemctl restart streamlit`.
- Change the password: edit `.streamlit/secrets.toml`, then `sudo systemctl restart streamlit`.
- Tear down: `sudo systemctl disable --now streamlit` (optionally remove the unit file).
- Logs: `journalctl -u streamlit` (add `-f` to follow, `-e` to jump to the end).
