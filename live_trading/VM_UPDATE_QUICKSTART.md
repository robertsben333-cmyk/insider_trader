# VM Update Quickstart

Use this when you have updated code locally and want the Oracle VM to run the new scoring or trading scripts.

This assumes:

- VM host: `143.47.182.234`
- VM user: `opc`
- repo on VM: `/home/opc/insider_trader`
- local SSH key: `~/.ssh/oracle_insider.key`

## 1. Push your local changes first

From your Windows machine in this repo:

```bash
git status
git add .
git commit -m "Update scoring/trading logic"
git push
```

If you do not push first, the VM cannot pull your latest code.

## 2. SSH into the VM

From Windows Git Bash:

```bash
ssh -i ~/.ssh/oracle_insider.key opc@143.47.182.234
```

## 3. Pull the latest repo safely

On the VM:

```bash
cd /home/opc/insider_trader
git status
```

If `git status` shows local changes under runtime folders such as `live/data/`, stash them first:

```bash
git stash push --include-untracked -m "vm-live-data-before-pull"
```

Then pull:

```bash
git pull
```

## 4. Update Python packages only if needed

If you changed `requirements.txt` or added a new package:

```bash
cd /home/opc/insider_trader
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If dependencies did not change, skip this step.

## 5. Restart the live services

For the current live trading setup:

```bash
sudo systemctl restart insider-live-scoring.service
sudo systemctl restart insider-ibkr-paper-trader.service
```

If you also changed service files:

```bash
sudo systemctl daemon-reload
sudo systemctl restart insider-live-scoring.service
sudo systemctl restart insider-ibkr-paper-trader.service
```

## 6. Verify both services are healthy

```bash
sudo systemctl status insider-live-scoring.service --no-pager
sudo systemctl status insider-ibkr-paper-trader.service --no-pager
```

Watch logs if needed:

```bash
sudo journalctl -u insider-live-scoring.service -n 50 --no-pager
sudo journalctl -u insider-ibkr-paper-trader.service -n 50 --no-pager
```

For live tailing:

```bash
sudo journalctl -u insider-live-scoring.service -f
sudo journalctl -u insider-ibkr-paper-trader.service -f
```

## 7. If the trader cannot connect to IBKR

Check that `IB Gateway` is logged in and port `4002` is listening:

```bash
sudo ss -ltnp | grep 4002
```

If needed, use the relogin runbook:

- [IBKR_GATEWAY_RELOGIN.md](/c:/Users/XavierFriesen/insider_trades_predictor/live_trading/IBKR_GATEWAY_RELOGIN.md)

## Fastest repeatable update flow

Local machine:

```bash
git add .
git commit -m "Your change"
git push
ssh -i ~/.ssh/oracle_insider.key opc@143.47.182.234
```

VM:

```bash
cd /home/opc/insider_trader
git stash push --include-untracked -m "vm-live-data-before-pull"
git pull
sudo systemctl restart insider-live-scoring.service
sudo systemctl restart insider-ibkr-paper-trader.service
sudo systemctl status insider-live-scoring.service --no-pager
sudo systemctl status insider-ibkr-paper-trader.service --no-pager
```

## Related docs

- [IBKR_GATEWAY_DEPLOYMENT.md](/c:/Users/XavierFriesen/insider_trades_predictor/live_trading/IBKR_GATEWAY_DEPLOYMENT.md)
- [ORACLE_CLOUD_DEPLOYMENT.md](/c:/Users/XavierFriesen/insider_trades_predictor/ORACLE_CLOUD_DEPLOYMENT.md)
