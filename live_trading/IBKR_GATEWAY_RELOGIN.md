# IBKR Gateway Relogin From A New Git Bash Session

This is the repeatable workflow to reopen the IB Gateway session after a reboot, logout, or Sunday re-auth.

What this automates:
- opens the SSH tunnel from your Windows machine to the VM VNC port
- launches RealVNC Viewer if it is installed in a standard location

What still stays manual:
- entering the IBKR paper username and password inside the Gateway login window
- clicking through any IBKR login prompts

IBKR credentials are intentionally not stored in this repo.

## One-command helper

From a fresh Windows Git Bash session at the repo root, run:

```bash
bash scripts/vm/open_ibkr_gateway_relogin_session.sh
```

Default assumptions:
- VM host: `143.47.182.234`
- VM user: `opc`
- SSH key: `~/.ssh/oracle_insider.key`
- local VNC port: `5901`
- remote VNC port: `5901`

Override any of those for one run like this:

```bash
VM_HOST=143.47.182.234 \
VM_USER=opc \
SSH_KEY_PATH="$HOME/.ssh/oracle_insider.key" \
LOCAL_VNC_PORT=5901 \
REMOTE_VNC_PORT=5901 \
bash scripts/vm/open_ibkr_gateway_relogin_session.sh
```

If RealVNC Viewer is installed in a non-standard path:

```bash
REALVNC_VIEWER_EXE="/c/Program Files/RealVNC/VNC Viewer/vncviewer.exe" \
bash scripts/vm/open_ibkr_gateway_relogin_session.sh
```

## What to do after the script runs

1. RealVNC should open to `localhost:5901`.
2. Log into `IB Gateway` with the paper username and password.
3. If Gateway is not already open, start it on the VM from a separate SSH shell:

```bash
ssh -i ~/.ssh/oracle_insider.key opc@143.47.182.234
DISPLAY=:1 ~/Jts/ibgateway/1037/ibgateway
```

4. Verify the API listener on the VM:

```bash
ssh -i ~/.ssh/oracle_insider.key opc@143.47.182.234
sudo ss -ltnp | grep 4002
```

Expected result:

```text
LISTEN 0 50 *:4002 *:* users:(("java",pid=...,fd=...))
```

## Service checks

Once Gateway is logged in, the scorer and trader services should already be able to run.

Check them from a separate SSH shell:

```bash
ssh -i ~/.ssh/oracle_insider.key opc@143.47.182.234
sudo systemctl status insider-live-scoring.service --no-pager
sudo systemctl status insider-ibkr-paper-trader.service --no-pager
```

Watch trader logs:

```bash
sudo journalctl -u insider-ibkr-paper-trader.service -f
```

## Weekly/Sunday routine

- Open a new Git Bash session.
- Run the helper script.
- Log back into IB Gateway in RealVNC.
- Confirm `4002` is listening.
- Close RealVNC when done.

You do not need to keep Git Bash or RealVNC open after the services are running and Gateway is logged in.

## Stop the VNC tunnel later

From Windows Git Bash:

```bash
pkill -f "ssh .*5901:localhost:5901.*opc@143.47.182.234"
```
