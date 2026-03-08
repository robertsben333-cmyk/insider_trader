# IBKR Gateway Deployment Runbook

This document captures the working deployment path for `IB Gateway` on the Oracle Linux VM used by this repo, plus the failure modes we hit and the fixes that worked.

This is the runbook to reproduce the environment from scratch.

## Target Environment

- VM host: `143.47.182.234`
- VM user: `opc`
- Repo path on VM: `/home/opc/insider_trader`
- OS: `Oracle Linux 9`
- Broker runtime: `IB Gateway 10.37`
- Gateway API socket: `4002`
- VNC display: `:1`
- VNC tunnel from Windows: `localhost:5901 -> VM localhost:5901`

## Final Working Architecture

- `TigerVNC` provides a persistent desktop on the VM.
- `IB Gateway` is launched on that desktop with `DISPLAY=:1` and kept detached from the SSH session.
- `insider-live-scoring.service` runs the scoring loop continuously.
- `insider-ibkr-paper-trader.service` runs the IBKR trader continuously.
- `insider-dashboard-sync.service` runs the read-only dashboard collector continuously.
- `insider-strategy-dashboard.service` serves the Streamlit dashboard continuously.
- `RealVNC` is only needed when `IB Gateway` needs login or recovery.
- `Git Bash` is only needed for setup, inspection, and occasional maintenance.

There should be only one `IB Gateway` process and one paper-account login. Scoring, trading, and dashboard sync all connect to that same API socket on `127.0.0.1:4002`.

## What Not To Do

- Do not try to run `IB Gateway` from a plain SSH shell with no display.
  You will get `HeadlessException` / `No X11 DISPLAY variable was set`.
- Do not rely on `ssh -Y` / X forwarding for the long-term setup.
  It was unreliable for Java GUI startup on this VM.
- Do not store IBKR usernames or passwords in the repo.
- Do not run SSH commands that use `~/.ssh/oracle_insider.key` from inside the VM.
  That key path only exists on the Windows machine.

## Part 1: VM Prerequisites

On the VM:

```bash
sudo dnf install -y python3.11 python3.11-pip git
sudo dnf install -y xorg-x11-server-Xvfb tigervnc-server tigervnc-server-module
```

If the repo is not already present:

```bash
git clone https://github.com/robertsben333-cmyk/insider_trader.git
cd /home/opc/insider_trader
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install joblib xgboost polygon-api-client python-dotenv requests beautifulsoup4 PyYAML tqdm retry
pip install "numpy==2.4.0" "scipy==1.16.3" "scikit-learn==1.8.0" "pandas==2.2.3"
```

## Part 2: Install IB Gateway

The installer was already used successfully and the working launcher path is:

```bash
~/Jts/ibgateway/1037/ibgateway
```

To rediscover it:

```bash
find ~/Jts -maxdepth 3 -type f -name ibgateway
```

## Part 3: Configure TigerVNC

### 1. Set the VNC password as user `opc`

Do not use `sudo`:

```bash
vncpasswd
```

On this Oracle Linux 9 setup, TigerVNC stores the password here:

```bash
~/.config/tigervnc/passwd
```

Not here:

```bash
~/.vnc/passwd
```

Fix permissions:

```bash
chmod 700 ~/.config/tigervnc
chmod 600 ~/.config/tigervnc/passwd
```

### 2. Map display `:1` to user `opc`

```bash
echo ':1=opc' | sudo tee -a /etc/tigervnc/vncserver.users
```

### 3. Configure the VNC session

The working session is `gnome`, not `twm`.

```bash
printf 'session=gnome\ngeometry=1280x1024\n' | sudo tee /etc/tigervnc/vncserver-config-defaults
```

### 4. Ensure the GUI stack exists

This VM originally failed because `gnome-session` was missing.

The fix was:

```bash
sudo dnf group install -y "Server with GUI"
```

Verify:

```bash
which gnome-session
```

Expected:

```text
/usr/bin/gnome-session
```

### 5. Enable and start VNC

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vncserver@:1.service
sudo systemctl status vncserver@:1.service --no-pager
```

When healthy, status should show `active (running)`.

## Part 4: Connect From Windows

### 1. Open the VNC tunnel from Windows Git Bash

```bash
ssh -L 5901:localhost:5901 -i ~/.ssh/oracle_insider.key opc@143.47.182.234
```

Leave that terminal open while using VNC.

### 2. Open RealVNC Viewer

Connect to:

```text
localhost:5901
```

Use the VNC password set with `vncpasswd`.

## Part 5: Start and Configure IB Gateway

Launch from a normal SSH session on the VM, detached from the SSH shell:

```bash
pkill -f '/home/opc/Jts/ibgateway/1037/ibgateway' || true
nohup env DISPLAY=:1 ~/Jts/ibgateway/1037/ibgateway > ~/ibgateway.log 2>&1 &
disown
```

Then in the VNC desktop:

1. Log in with the **paper trading username** and password.
2. Go to `Configure -> Settings -> API -> Settings`
3. Verify:
   - `Read-Only API` is unchecked
   - `Socket port` is `4002`

### Verify the listener

On the VM:

```bash
sudo ss -ltnp | grep 4002
```

Expected:

```text
LISTEN 0 50 *:4002 *:* users:(("java",pid=...,fd=...))
```

## Part 6: Repo Configuration

The trader config lives in:

- [strategy_settings.py](/c:/Users/XavierFriesen/insider_trades_predictor/live_trading/strategy_settings.py)

Important values:

- `IBKR_CONFIG.account_id="DUN175042"`
- `IBKR_CONFIG.port=4002`
- `TRADING_BUDGET.initial_strategy_budget=10000.0`

The IBKR paper username is not stored in code. It is only used in the Gateway login window.

## Part 7: Environment Variables

The VM uses:

```bash
/etc/insider_trades.env
```

At minimum it needs:

```ini
POLYGON_API_KEY=...
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=...
SMTP_PASS=...
```

For manual shell testing, a readable copy in the home directory is useful:

```bash
sudo cp /etc/insider_trades.env /home/opc/insider_trades.env
sudo chown opc:opc /home/opc/insider_trades.env
chmod 600 /home/opc/insider_trades.env
```

Load it:

```bash
set -a
source /home/opc/insider_trades.env
set +a
```

## Part 8: Systemd Services

### Live scoring service

`/etc/systemd/system/insider-live-scoring.service`

```ini
[Unit]
Description=Insider Trades Live Scoring
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=opc
WorkingDirectory=/home/opc/insider_trader
EnvironmentFile=/etc/insider_trades.env
ExecStart=/home/opc/insider_trader/.venv/bin/python3.11 /home/opc/insider_trader/live_trading/run_live_scoring.py --no-email
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Trader service

`/etc/systemd/system/insider-ibkr-paper-trader.service`

```ini
[Unit]
Description=Insider Trades IBKR Paper Trader
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=opc
WorkingDirectory=/home/opc/insider_trader
EnvironmentFile=/etc/insider_trades.env
ExecStart=/home/opc/insider_trader/.venv/bin/python3.11 /home/opc/insider_trader/live_trading/run_ibkr_paper_trader.py
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Dashboard sync service

`/etc/systemd/system/insider-dashboard-sync.service`

```ini
[Unit]
Description=Insider Trades IBKR Dashboard Sync
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=opc
WorkingDirectory=/home/opc/insider_trader
EnvironmentFile=/etc/insider_trades.env
ExecStart=/home/opc/insider_trader/scripts/vm/run_dashboard_sync.sh
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Dashboard service

`/etc/systemd/system/insider-strategy-dashboard.service`

```ini
[Unit]
Description=Insider Trades Strategy Dashboard
After=network-online.target insider-dashboard-sync.service
Wants=network-online.target

[Service]
Type=simple
User=opc
WorkingDirectory=/home/opc/insider_trader
Environment=STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ExecStart=/home/opc/insider_trader/scripts/vm/run_strategy_dashboard.sh
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable vncserver@:1.service
sudo systemctl enable insider-live-scoring.service
sudo systemctl enable insider-ibkr-paper-trader.service
sudo systemctl enable insider-dashboard-sync.service
sudo systemctl enable insider-strategy-dashboard.service

sudo systemctl start insider-live-scoring.service
sudo systemctl start insider-ibkr-paper-trader.service
sudo systemctl start insider-dashboard-sync.service
sudo systemctl start insider-strategy-dashboard.service
```

Do not rely on `insider-ibgateway.service` unless you have separately validated that launcher path on the VM. The stable operating model is:

1. `vncserver@:1.service` stays up
2. `IB Gateway` is launched manually once, detached from SSH
3. the four Python services run under `systemd`

### Check status

```bash
sudo systemctl status insider-live-scoring.service --no-pager
sudo systemctl status insider-ibkr-paper-trader.service --no-pager
sudo systemctl status insider-dashboard-sync.service --no-pager
sudo systemctl status insider-strategy-dashboard.service --no-pager
```

### Follow logs

```bash
sudo journalctl -u insider-live-scoring.service -f
sudo journalctl -u insider-ibkr-paper-trader.service -f
sudo journalctl -u insider-dashboard-sync.service -f
sudo journalctl -u insider-strategy-dashboard.service -f
```

## Part 9: Manual Tests That Passed

### IBKR connectivity test

```bash
cd /home/opc/insider_trader
source .venv/bin/activate
python3.11 live_trading/run_ibkr_paper_trader.py --once
```

Successful result included:

- `Connected`
- `API connection ready`
- `Synchronization complete`

### Live scoring test

```bash
python3.11 live_trading/run_live_scoring.py --once --no-email
```

Successful result included:

- models loaded
- Polygon key worked
- OpenInsider scrape ran
- no new filings is a valid idle outcome

### Forced paper order test

A forced `AAPL` candidate was injected and the trader submitted a real paper order:

- `broker_order_id=6`
- `status=submitted`
- `reserved_cash=261.59`

This proved the path:

`signal -> trader -> IB Gateway -> paper order submission`

## Lessons Learned

### 1. X forwarding was not the right approach

Symptoms:
- `No X11 DISPLAY variable was set`
- `Can't connect to X11 window server using 'localhost:10.0'`

Fix:
- stop relying on `ssh -Y`
- use `TigerVNC` + VM-local display `:1`

### 2. `x11vnc` was not available in the enabled repos

Fix:
- use `tigervnc-server` and `tigervnc-server-module`

### 3. VNC password location was not `~/.vnc/passwd`

Actual path on this setup:

```bash
~/.config/tigervnc/passwd
```

### 4. VNC service exited immediately until a desktop session existed

Symptoms:
- `vncserver@:1.service` started and then immediately deactivated

Root cause:
- `session=gnome`
- `gnome-session` missing

Fix:
- `sudo dnf group install -y "Server with GUI"`

### 5. Commands with `~/.ssh/oracle_insider.key` must be run on Windows

If you run them inside the VM, they fail because the key file is not there.

### 6. IB Gateway API must not be read-only

The trader needs:
- `Read-Only API` unchecked
- port `4002`

### 7. Launching Gateway directly from the SSH shell ties its lifetime to that shell

Symptoms:
- closing the local VNC/Git Bash session drops the Gateway process

Fix:
- launch Gateway detached:

```bash
nohup env DISPLAY=:1 ~/Jts/ibgateway/1037/ibgateway > ~/ibgateway.log 2>&1 &
disown
```

### 8. The dashboard and trader should share the same Gateway session

Do not start a second Gateway or a second IBKR login for the dashboard.

Correct model:
- one `IB Gateway`
- one login
- scorer, trader, and dashboard sync all use `127.0.0.1:4002`

### 9. A clean trader run with no orders can still be correct

If `latest_alert_candidates.csv` only contains stale entries, the trader should mark them as expired and place nothing.

That is expected behavior, not a failure.

### 10. Runtime data on the VM can block `git pull`

Files under `live/data/` changed on the VM prevented `git pull`.

Safe fix:

```bash
git stash push --include-untracked -m "vm-live-data-before-pull"
git pull
```

### 11. The latest code must actually be present on the VM

The first trader run failed because the VM repo did not yet include:

- `live_trading/ibkr_paper_trader.py`
- `live_trading/run_ibkr_paper_trader.py`

## Weekly Operating Routine

IBKR requires periodic re-authentication, especially around Sunday.

When Gateway needs login again:

1. From Windows Git Bash:

```bash
bash scripts/vm/open_ibkr_gateway_relogin_session.sh
```

2. RealVNC opens to `localhost:5901`
3. On the VM, launch detached if Gateway is not already running:

```bash
pkill -f '/home/opc/Jts/ibgateway/1037/ibgateway' || true
nohup env DISPLAY=:1 ~/Jts/ibgateway/1037/ibgateway > ~/ibgateway.log 2>&1 &
disown
```

4. Log into IB Gateway manually
5. Verify:

```bash
sudo ss -ltnp | grep 4002
```

6. Optionally check:

```bash
sudo systemctl status insider-live-scoring.service --no-pager
sudo systemctl status insider-ibkr-paper-trader.service --no-pager
sudo systemctl status insider-dashboard-sync.service --no-pager
sudo systemctl status insider-strategy-dashboard.service --no-pager
```

7. Close RealVNC when done

You do not need to keep RealVNC or Git Bash open permanently once:

- `IB Gateway` is logged in
- `4002` is listening
- `vncserver@:1.service` is active
- scorer, trader, dashboard sync, and dashboard services are active

## Related Files

- [IBKR_GATEWAY_RELOGIN.md](/c:/Users/XavierFriesen/insider_trades_predictor/live_trading/IBKR_GATEWAY_RELOGIN.md)
- [strategy_settings.py](/c:/Users/XavierFriesen/insider_trades_predictor/live_trading/strategy_settings.py)
- [open_ibkr_gateway_relogin_session.sh](/c:/Users/XavierFriesen/insider_trades_predictor/scripts/vm/open_ibkr_gateway_relogin_session.sh)
- [ORACLE_CLOUD_DEPLOYMENT.md](/c:/Users/XavierFriesen/insider_trades_predictor/ORACLE_CLOUD_DEPLOYMENT.md)
