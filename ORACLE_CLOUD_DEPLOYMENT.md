# Oracle Cloud Deployment Guide — insider_trader

## Instance Details (current)
- **Provider**: Oracle Cloud Free Tier
- **Region**: Netherlands Northwest (eu-amsterdam-1)
- **Shape**: VM.Standard.E2.1 (AMD, 1 OCPU, 8 GB RAM)
- **OS**: Oracle Linux 9 (NOT Ubuntu — username is `opc`, not `ubuntu`)
- **Public IP**: 143.47.182.234
- **Repo on VM**: `/home/opc/insider_trader`
- **GitHub repo**: https://github.com/robertsben333-cmyk/insider_trader (public)

---

## PART 1 — Oracle Cloud Console Setup

### 1. Create account
- Go to https://cloud.oracle.com and sign up for Free Tier
- Enter credit card (identity verification only, not charged for Always Free resources)
- Choose Home Region close to you — **cannot be changed later**

### 2. Create compute instance
- Hamburger menu (≡) → Compute → Instances → Create Instance
- **Image**: Oracle Linux 9
- **Shape**: VM.Standard.E2.1 (AMD — always has capacity; A1 Flex ARM is often out of capacity)
- **SSH keys**: generate a key pair and download the private key, OR upload your own public key
- Click **Create**
- Wait until status shows **Running**

### 3. Assign a Public IP
- Go to instance → Networking tab → Attached VNICs → click `inside_predictor`
- Click **IP Administration** tab
- Click three dots (⋮) next to the private IP → Edit
- Under "Public IP address" select **Ephemeral public IP** → Save
- Note the public IP (e.g. `143.47.182.234`)

### 4. Start the instance if stopped
- On the instance details page, click **Start** in the top-right corner

---

## PART 2 — Local Machine Setup (Git Bash on Windows)

### 5. Save your SSH private key
```bash
mkdir -p ~/.ssh
nano ~/.ssh/oracle_insider.key
# Paste the full private key content (including -----BEGIN and -----END lines)
# Save: Ctrl+O → Enter → Ctrl+X
chmod 600 ~/.ssh/oracle_insider.key
```

### 6. Connect to the VM
```bash
ssh -i ~/.ssh/oracle_insider.key opc@143.47.182.234
```
- Type `yes` when asked about host authenticity
- Ignore the post-quantum warning (harmless)
- You are now inside the Oracle VM when you see `[opc@inside-predictor ~]$`

> **Paste tip**: Use **Shift+Insert** in Git Bash — regular paste adds `^[[200~` garbage

---

## PART 3 — VM Setup (run all commands inside the SSH session)

### 7. Install Python and Git
```bash
sudo dnf install -y python3.11 python3.11-pip git
```

### 8. Clone the repo
```bash
git clone https://github.com/robertsben333-cmyk/insider_trader.git
cd insider_trader
```

If repo is private, use a GitHub personal access token:
```bash
git clone https://USERNAME:YOUR_TOKEN@github.com/USERNAME/insider_trader.git
cd insider_trader
```
Get a token at: github.com → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate → check "repo"

### 9. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Your prompt should now show `(.venv)` at the start.

### 10. Install dependencies — CRITICAL: pin exact versions to match local machine
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install joblib xgboost polygon-api-client python-dotenv requests beautifulsoup4 PyYAML tqdm retry
pip install "numpy==2.4.0" "scipy==1.16.3" "scikit-learn==1.8.0" "pandas==2.2.3"
```

**Why pinning is critical — errors you will get without it:**

| Wrong version | Error message | Fix |
|---|---|---|
| pandas >= 3.0 | `KeyError: 'ticker'` in groupby | `pandas==2.2.3` |
| wrong scipy | `BSpline object has no attribute '_asarray'` | match local scipy |
| wrong numpy | `PCG64 is not a known BitGenerator` | match local numpy |
| wrong scikit-learn | `No module named '_loss'` | match local sklearn |

To check your local versions (run in Git Bash on your Windows machine):
```bash
py -c "import numpy, scipy, sklearn, pandas; print(numpy.__version__, scipy.__version__, sklearn.__version__, pandas.__version__)"
```

### 11. Create the .env file
Place it in `/etc/` — NOT in `/home/opc/` (systemd cannot read files there due to SELinux):
```bash
sudo nano /etc/insider_trades.env
```
Paste contents:
```ini
POLYGON_API_KEY=your_polygon_key_here
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=yourgmail@gmail.com
SMTP_PASS=your_16_char_gmail_app_password_here
```
Save: `Ctrl+O` → `Enter` → `Ctrl+X`
```bash
sudo chmod 600 /etc/insider_trades.env
```

Use Gmail app-password SMTP for the current setup:
- `SMTP_SERVER=smtp.gmail.com`
- `SMTP_PORT=587`
- `SMTP_USER` should be the Gmail sender account
- `SMTP_PASS` must be a Google app password, not the normal Gmail password

### 12. Test run manually
The script does NOT support `--env-file`. Load env vars with `set -a / source`:
```bash
cd /home/opc/insider_trader
source .venv/bin/activate
set -a
source /etc/insider_trades.env
set +a
python live_scoring.py --model-dir models/prod4 --no-email
```
You should see INFO log lines and a LIVE SCORING REPORT after ~2 minutes. Hit `Ctrl+C` once confirmed working.

---

## PART 4 — Run as a Permanent Service (24/7)

### 13. Create the systemd service file
```bash
sudo nano /etc/systemd/system/insider-trades.service
```
Paste exactly:
```ini
[Unit]
Description=Insider Trades Live Scorer
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=opc
WorkingDirectory=/home/opc/insider_trader
EnvironmentFile=/etc/insider_trades.env
ExecStart=/home/opc/insider_trader/.venv/bin/python3.11 live_scoring.py --model-dir models/prod4
Restart=always
RestartSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```
Save: `Ctrl+O` → `Enter` → `Ctrl+X`

> **IMPORTANT**: Use `python3.11` (the real binary), NOT `python` (symlink — causes status=203/EXEC)

### 14. Fix SELinux context on venv binaries
Without this, systemd gets "Permission denied" on the venv executables:
```bash
sudo chcon -R -t bin_t /home/opc/insider_trader/.venv/bin/
```

### 15. Enable and start the service
```bash
sudo systemctl daemon-reload
sudo systemctl enable insider-trades
sudo systemctl start insider-trades
```

### 16. Verify it's running
```bash
sudo systemctl status insider-trades
```
You should see `Active: active (running)`.

Watch the live logs:
```bash
sudo journalctl -u insider-trades -f
```
Press `q` to exit the log view.

---

## Monitoring & Management

```bash
# Live log stream
sudo journalctl -u insider-trades -f

# Check status
sudo systemctl status insider-trades

# Restart
sudo systemctl restart insider-trades

# Stop
sudo systemctl stop insider-trades

# View last 50 log lines
sudo journalctl -u insider-trades -n 50
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Permission denied (publickey)` | Wrong SSH key | Generate new key pair, add public key to Oracle instance |
| `Could not resolve hostname` | Used tenancy name instead of IP | Use the actual public IP from OCI console |
| `status=203/EXEC Permission denied` | SELinux blocking venv symlink | `sudo chcon -R -t bin_t /home/opc/insider_trader/.venv/bin/` |
| `EnvironmentFile: Permission denied` | .env file in /home/opc/ | Move to `/etc/insider_trades.env` |
| `ModuleNotFoundError` | System python used instead of venv | Use full venv path `python3.11` in ExecStart |
| `KeyError: 'ticker'` | pandas 3.0 groupby change | `pip install "pandas==2.2.3"` |
| `BSpline _asarray` error | scipy version mismatch | `pip install "scipy==1.16.3"` |
| `PCG64 BitGenerator` error | numpy version mismatch | `pip install "numpy==2.4.0"` |
| `No module named '_loss'` | scikit-learn version mismatch | `pip install "scikit-learn==1.8.0"` |
| `^[[200~git: command not found` | Bad paste in Git Bash | Use **Shift+Insert** to paste |
| Instance has no public IP | Not assigned during creation | Networking tab → IP Administration → Ephemeral public IP |
| `Out of capacity for VM.Standard.A1.Flex` | ARM free tier oversubscribed | Use AMD shape `VM.Standard.E2.1` instead |
