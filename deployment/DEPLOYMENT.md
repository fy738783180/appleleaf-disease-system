# Ubuntu Deployment Guide

This guide shows one simple way to deploy the app on Ubuntu with `systemd`.

## 1. Upload Project

Example target directory:

```bash
/opt/appleleaf-streamlit/app
```

## 2. Create Virtual Environment

```bash
sudo apt-get update
sudo apt-get install -y python3-venv

cd /opt/appleleaf-streamlit
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision
pip install -r /opt/appleleaf-streamlit/app/requirements.txt
```

## 4. Prepare Models

Put the model files in:

```text
/opt/appleleaf-streamlit/app/models/
```

Required filenames:

- `vit_resnet_multi_task_model.pth`
- `ld_deeplabv3plus_best_model_3.pth`

## 5. Install systemd Service

Copy the service file:

```bash
sudo cp /opt/appleleaf-streamlit/app/deployment/appleleaf-streamlit.service /etc/systemd/system/appleleaf-streamlit.service
```

Reload and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now appleleaf-streamlit.service
```

Check status:

```bash
sudo systemctl status appleleaf-streamlit.service
```

## 6. Open Port

If the app listens on `8501`, open:

- Ubuntu firewall:

```bash
sudo ufw allow 8501/tcp
```

- Cloud firewall / security policy:
  - add inbound TCP 8501

## 7. Access

```text
http://YOUR_SERVER_IP:8501
```

## Notes

- If you already run VPN, Nginx, or HTTPS services on `443`, use a different application port such as `8501`
- Do not overwrite existing VPN-related services
- For production hardening, you can later place Nginx in front of Streamlit
