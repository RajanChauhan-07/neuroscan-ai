# 🧠 NeuroScan AI — Alzheimer's Clinical Dashboard

> An end-to-end AI-powered clinical research tool that combines deep learning MRI analysis with machine learning risk prediction to assist neurologists in Alzheimer's diagnosis.
>
> 🌐 **Live at: https://neuroscan.ddns.net**

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Models](#models)
  - [Model A — MRI Image Classification](#model-a--mri-image-classification-efficientnetb3)
  - [Model B — Clinical Risk Prediction](#model-b--clinical-risk-prediction-xgboost)
- [Dashboard](#dashboard)
- [Project Structure](#project-structure)
- [Setup & Run Locally](#setup--run-locally)
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [Domain & HTTPS Setup](#domain--https-setup)
- [Managing the Server](#managing-the-server)
- [Troubleshooting](#troubleshooting)
- [Key Files](#key-files)
- [Results Summary](#results-summary)
- [Tech Stack](#tech-stack)
- [Cost Breakdown](#cost-breakdown)

---

## Project Overview

NeuroScan AI is a full-stack clinical research dashboard that:

1. **Classifies Alzheimer's stage** from a brain MRI scan using a fine-tuned EfficientNetB3 deep learning model (4 classes, 99.82% accuracy)
2. **Predicts progression risk** from 32 clinical/demographic features using an XGBoost classifier (94.88% accuracy)
3. **Generates a unified clinical report** combining both model outputs into a single actionable summary for neurologists

---

## Architecture

### Local / Project Structure

```
alzheimer_dashboard/
│
├── app.py                        ← Flask backend (model loading + inference)
├── requirements.txt              ← Python dependencies
├── Procfile                      ← Gunicorn server entry point
│
├── models/
│   ├── model_A.pth               ← EfficientNetB3 weights (MRI classifier)
│   ├── model_B.pkl               ← XGBoost model (risk predictor)
│   └── class_labels.json         ← Class index → label mapping
│
├── templates/
│   └── index.html                ← Apple-style frontend UI
│
└── static/
    ├── css/style.css             ← Styling (Apple aesthetic)
    └── js/main.js                ← Frontend logic, animations, API calls
```

### Request Flow

```
Browser → Flask /analyze endpoint
         → Model A: transform image → EfficientNetB3 → softmax → stage + confidence
         → Model B: parse form fields → XGBoost → risk probability
         → generate_report() → unified clinical summary
         → JSON response → rendered results UI
```

### Production Deployment Architecture (AWS EC2)

```
                        Internet
                           │
                    ┌──────▼──────┐
                    │  Route DNS  │
                    │neuroscan.   │
                    │ ddns.net    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Elastic IP │
                    │54.211.52.67 │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │      AWS EC2            │
              │   Ubuntu 22.04 LTS      │
              │    t3.medium            │
              │                         │
              │  ┌──────────────────┐   │
              │  │   Nginx (80/443) │   │
              │  │  Reverse Proxy   │   │
              │  │  + SSL/TLS       │   │
              │  └────────┬─────────┘   │
              │           │ Unix Socket  │
              │  ┌────────▼─────────┐   │
              │  │    Gunicorn      │   │
              │  │   2 Workers      │   │
              │  │  WSGI Server     │   │
              │  └────────┬─────────┘   │
              │           │             │
              │  ┌────────▼─────────┐   │
              │  │   Flask App      │   │
              │  │  application.py  │   │
              │  └────────┬─────────┘   │
              │           │             │
              │  ┌────────▼─────────┐   │
              │  │   ML Models      │   │
              │  │  model_A.pth     │   │
              │  │  model_B.pkl     │   │
              │  └──────────────────┘   │
              │                         │
              │  Managed by: systemd    │
              └─────────────────────────┘
```

---

## Models

### Model A — MRI Image Classification (EfficientNetB3)

#### Dataset
- **Source:** Alzheimer's Disease Multiclass Images Dataset (Kaggle: `aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented`)
- **Size:** 44,000 MRI JPG images, 4 classes, pre-skull-stripped, balanced
- **Classes:**
  - `NonDemented` — 12,800 images
  - `VeryMildDemented` — 11,200 images
  - `MildDemented` — 10,000 images
  - `ModerateDemented` — 10,000 images
- **Split:** 80/10/10 → Train: 35,200 | Val: 4,400 | Test: 4,400

#### Architecture & Config

| Parameter | Value |
|---|---|
| Base Model | EfficientNetB3 (timm) |
| Pretrained | ImageNet |
| Total Parameters | 10,702,380 (all trainable) |
| Image Size | 224×224 |
| Batch Size | 32 |
| Epochs | 15 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss |
| Seed | 42 |

#### Image Transforms

```python
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

#### Training Progress (A100 GPU, ~15 mins)

| Epoch | Train Acc | Val Acc |
|---|---|---|
| 1 | 77.09% | 93.84% |
| 2 | 97.14% | 97.64% |
| 5 | 99.51% | 98.66% |
| 11 | 99.95% | 99.80% |
| **12** | **99.99%** | **99.86% ← Best** |
| 15 | 100.0% | 99.82% |

#### Test Results (4,400 images)

| Class | Correct | Total | Accuracy |
|---|---|---|---|
| MildDemented | 999 | 1000 | 99.9% |
| ModerateDemented | 983 | 983 | 100% |
| NonDemented | 1272 | 1276 | 99.7% |
| VeryMildDemented | 1138 | 1141 | 99.7% |
| **Overall** | **4392** | **4400** | **~100%** |

Only **8 images misclassified** out of 4,400. Precision/Recall/F1 = 1.00 for all classes.

#### Training Code (Key Snippet)

```python
import timm
model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=4)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

# Save best model
torch.save(model.state_dict(), 'model_A.pth')
```

#### Inference Code

```python
model_A = timm.create_model('efficientnet_b3', pretrained=False, num_classes=4)
model_A.load_state_dict(torch.load('models/model_A.pth', map_location=DEVICE))
model_A.eval()

img = Image.open(file).convert('RGB')
tensor = val_transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model_A(tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = probs.argmax().item()
```

---

### Model B — Clinical Risk Prediction (XGBoost)

#### Dataset
- **Source:** Alzheimer's Disease Dataset (Kaggle: `rabieelkharoua/alzheimers-disease-dataset`)
- **Size:** 2,149 patients × 35 columns
- **Target:** `Diagnosis` (0 = No Alzheimer's: 1,389 | 1 = Alzheimer's: 760)
- **Features (32):** Age, Gender, Ethnicity, EducationLevel, BMI, Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality, FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes, Depression, HeadInjury, Hypertension, SystolicBP, DiastolicBP, CholesterolTotal, CholesterolLDL, CholesterolHDL, CholesterolTriglycerides, MMSE, FunctionalAssessment, MemoryComplaints, BehavioralProblems, ADL, Confusion, Disorientation, PersonalityChanges, DifficultyCompletingTasks, Forgetfulness
- **Dropped:** PatientID, DoctorInCharge
- **Split:** 80/20 stratified, Seed=42

#### Architecture & Config

```python
from xgboost import XGBClassifier

model_B = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1389/760,  # handles class imbalance
    random_state=42
)
model_B.fit(X_train, y_train)
```

#### Results

| Metric | No Alzheimer's | Alzheimer's | Overall |
|---|---|---|---|
| Precision | 0.96 | 0.93 | — |
| Recall | 0.96 | 0.92 | — |
| F1 Score | 0.96 | 0.93 | — |
| Support | 278 | 152 | 430 |
| **Accuracy** | | | **94.88%** |
| **AUC** | | | **0.9446** |

**Confusion Matrix:**
```
True Negative:  268  |  False Positive: 10
False Negative:  12  |  True Positive: 140
```

#### Inference Code

```python
import pickle
import numpy as np

with open('models/model_B.pkl', 'rb') as f:
    model_B = pickle.load(f)

X = np.array(patient_features).reshape(1, -1)
risk_prob = float(model_B.predict_proba(X)[0][1]) * 100
```

---

## Dashboard

### Tech Stack
- **Backend:** Flask 3.0 (Python)
- **Frontend:** Vanilla HTML/CSS/JS (Apple-style design)
- **Fonts:** Instrument Serif + DM Sans (Google Fonts)
- **Server:** Gunicorn (production)

### UI Features
- Sticky frosted-glass navigation
- Hero section with live model accuracy stats
- Drag & drop MRI upload with image preview
- Full clinical data form (32 fields, organized by category)
- Toggle-style medical history checkboxes
- Animated loading overlay with step-by-step progress
- MRI result card with probability bars (animated)
- Risk gauge with animated counter
- Unified clinical report in serif typography
- Responsive design (mobile-friendly)

### Flask Backend (`app.py`)

The backend handles:
1. Loading both models at startup
2. `/analyze` POST endpoint:
   - Receives multipart form data (image + clinical fields)
   - Runs Model A inference on the MRI
   - Parses and runs Model B inference on clinical data
   - Calls `generate_report()` for the unified summary
   - Returns JSON with all results

```python
@app.route('/analyze', methods=['POST'])
def analyze():
    # Model A: MRI
    file = request.files['mri']
    img = Image.open(file.stream).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model_A(tensor), dim=1)[0]

    # Model B: Tabular
    row = [float(request.form.get(f, 0)) for f in fields]
    X = np.array(row).reshape(1, -1)
    risk_prob = float(model_B.predict_proba(X)[0][1]) * 100

    return jsonify({ 'mri': {...}, 'risk': {...}, 'report': ... })
```

---

## Setup & Run Locally

### Prerequisites
- Python 3.13
- Git
- VS Code (recommended)

### Step 1 — Clone the repo

```bash
git clone https://github.com/RajanChauhan-07/neuroscan-ai.git
cd neuroscan-ai
```

### Step 2 — Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install flask timm xgboost scikit-learn numpy Pillow gunicorn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 4 — Add model files

Place the following files inside the `models/` folder:
```
models/
├── model_A.pth
├── model_B.pkl
└── class_labels.json
```

### Step 5 — Run

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Key Files

| File | Purpose |
|---|---|
| `app.py` | Flask server, model loading, inference endpoints |
| `templates/index.html` | Full frontend UI (Apple-style) |
| `static/css/style.css` | All styling — typography, layout, animations |
| `static/js/main.js` | Upload handling, form submission, result rendering |
| `models/model_A.pth` | EfficientNetB3 trained weights |
| `models/model_B.pkl` | XGBoost trained model |
| `models/class_labels.json` | `{"0": "MildDemented", "1": "ModerateDemented", ...}` |
| `requirements.txt` | Python dependencies |
| `Procfile` | `web: gunicorn app:app` |

---

## Results Summary

| Model | Architecture | Dataset | Accuracy | AUC |
|---|---|---|---|---|
| Model A — MRI | EfficientNetB3 | 44,000 MRI images (4 classes) | **99.82%** | ~1.00 |
| Model B — Risk | XGBoost | 2,149 patients (32 features) | **94.88%** | **0.9446** |

### Sample Outputs

**High-risk patient (MildDemented MRI):**
```
MRI:  Mild Dementia — 100% confidence
Risk: High Risk — 99.9% progression probability
Report: "The MRI analysis (100.0% confidence) indicates mild dementia.
         Clinical data analysis shows a high risk of progression within
         12 months (probability: 99.9%). Urgent specialist consultation
         required. Consider memory care planning."
```

**Healthy patient (NonDemented MRI):**
```
MRI:  No Dementia — 100% confidence
Risk: Low Risk — 0.5% progression probability
Report: "The MRI analysis (100.0% confidence) indicates no signs of
         dementia. Clinical data analysis shows a low risk of near-term
         progression (probability: 0.5%). Annual cognitive screening is
         sufficient at this stage."
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch 2.3.0, timm 0.9.12 (EfficientNetB3) |
| Machine Learning | XGBoost 2.0.3, scikit-learn |
| Image Processing | torchvision transforms, Pillow |
| Backend | Python 3.12, Flask, Gunicorn 25.1.0 |
| Frontend | HTML5, CSS3, Vanilla JS |
| Reverse Proxy | Nginx 1.24 |
| Process Manager | systemd |
| SSL | Let's Encrypt (Certbot) |
| Cloud | AWS EC2 (t3.medium, Ubuntu 22.04) |
| Static IP | AWS Elastic IP |
| Domain | No-IP Free DDNS |
| Training Platform | Lightning AI (NVIDIA A100 80GB) |
| Version Control | Git, GitHub |

---

## AWS EC2 Deployment

### Step 1 — Launch EC2 Instance

1. Go to **AWS Console → EC2 → Launch Instance**
2. Configure:
   - **Name:** `neuroscan-ai`
   - **AMI:** Ubuntu Server 22.04 LTS
   - **Instance type:** `t3.medium` (2 vCPU, 4GB RAM — minimum for PyTorch)
   - **Key pair:** Create new → `neuroscan-key` → download `.pem`
   - **Storage:** 20GB
   - **Security group — open ports:**
     - SSH (22) — 0.0.0.0/0
     - HTTP (80) — 0.0.0.0/0
     - HTTPS (443) — 0.0.0.0/0

### Step 2 — Connect via SSH

```bash
# Secure your key file
mv ~/Downloads/neuroscan-key.pem ~/.ssh/
chmod 400 ~/.ssh/neuroscan-key.pem

# Connect
ssh -i ~/.ssh/neuroscan-key.pem ubuntu@YOUR_EC2_IP
```

### Step 3 — Install System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv nginx -y
```

### Step 4 — Set Up Python Environment

```bash
mkdir ~/neuroscan && cd ~/neuroscan
python3 -m venv venv
source venv/bin/activate

pip install flask gunicorn numpy Pillow scikit-learn xgboost timm==0.9.12
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

### Step 5 — Upload Project Files

Run from your **local Mac terminal** (not SSH):

```bash
scp -i ~/.ssh/neuroscan-key.pem ~/alzheimer_dashboard/application.py ubuntu@YOUR_IP:~/neuroscan/
scp -i ~/.ssh/neuroscan-key.pem -r ~/alzheimer_dashboard/models ubuntu@YOUR_IP:~/neuroscan/
scp -i ~/.ssh/neuroscan-key.pem -r ~/alzheimer_dashboard/templates ubuntu@YOUR_IP:~/neuroscan/
scp -i ~/.ssh/neuroscan-key.pem -r ~/alzheimer_dashboard/static ubuntu@YOUR_IP:~/neuroscan/
```

### Step 6 — Test the App

```bash
cd ~/neuroscan
source venv/bin/activate
python application.py
# Should show: * Running on http://127.0.0.1:5000
# Press Ctrl+C to stop
```

### Step 7 — Create systemd Service (Auto-restart on reboot)

```bash
sudo nano /etc/systemd/system/neuroscan.service
```

Paste:

```ini
[Unit]
Description=NeuroScan AI Gunicorn
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/neuroscan
Environment="PATH=/home/ubuntu/neuroscan/venv/bin"
ExecStart=/home/ubuntu/neuroscan/venv/bin/gunicorn \
  --workers 2 \
  --bind unix:neuroscan.sock \
  --timeout 120 \
  application:application
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable neuroscan
sudo systemctl start neuroscan
sudo systemctl status neuroscan
# Should show: Active: active (running)
```

### Step 8 — Configure Nginx

```bash
sudo nano /etc/nginx/sites-available/neuroscan
```

Paste:

```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    client_max_body_size 10M;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/ubuntu/neuroscan/neuroscan.sock;
        proxy_read_timeout 120s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/neuroscan /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 9 — Fix Socket Permissions

```bash
sudo usermod -aG ubuntu www-data
sudo chmod 775 /home/ubuntu
sudo chmod 775 /home/ubuntu/neuroscan
sudo systemctl restart neuroscan
sudo systemctl restart nginx
```

### Elastic IP Setup

Elastic IP ensures your server keeps the same IP even after stopping/starting the instance.

1. Go to **EC2 → Elastic IPs → Allocate Elastic IP**
2. Click **Allocate**
3. Select the new IP → **Actions → Associate Elastic IP**
4. Select your `neuroscan-ai` instance → **Associate**

```bash
# Update Nginx with new IP
sudo sed -i 's/OLD_IP/NEW_IP/g' /etc/nginx/sites-available/neuroscan
sudo nginx -t && sudo systemctl restart nginx
```

**Our Elastic IP:** `54.211.52.67`

---

## Domain & HTTPS Setup

### Free Domain (No-IP)

1. Sign up at [noip.com](https://noip.com)
2. Create hostname: `neuroscan.ddns.net`
3. Point to your Elastic IP: `54.211.52.67`
4. Type: A record

### SSL Certificate (Let's Encrypt — Free)

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d neuroscan.ddns.net
# Enter email, agree to terms
# Certificate auto-renews every 90 days
```

### Nginx Config with HTTPS

```bash
sudo nano /etc/nginx/sites-available/neuroscan
```

Replace with:

```nginx
server {
    listen 80;
    server_name neuroscan.ddns.net;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name neuroscan.ddns.net;

    ssl_certificate /etc/letsencrypt/live/neuroscan.ddns.net/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/neuroscan.ddns.net/privkey.pem;

    client_max_body_size 10M;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/ubuntu/neuroscan/neuroscan.sock;
        proxy_read_timeout 120s;
    }
}
```

```bash
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx
```

---

## Managing the Server

### Daily Commands

```bash
# SSH into server
ssh -i ~/.ssh/neuroscan-key.pem ubuntu@54.211.52.67

# Check app status
sudo systemctl status neuroscan

# Restart app
sudo systemctl restart neuroscan

# View live logs
sudo journalctl -u neuroscan -f

# Restart Nginx
sudo systemctl restart nginx
```

### Deploy Code Updates

```bash
# Upload updated file from local machine
scp -i ~/.ssh/neuroscan-key.pem ~/alzheimer_dashboard/application.py ubuntu@54.211.52.67:~/neuroscan/

# SSH in and restart
ssh -i ~/.ssh/neuroscan-key.pem ubuntu@54.211.52.67
sudo systemctl restart neuroscan
```

### Stop/Start Instance

- **Stop:** AWS Console → EC2 → Instance State → Stop
- **Start:** AWS Console → EC2 → Instance State → Start
- Site comes back **automatically** on start (systemd handles it)
- IP stays the same (Elastic IP)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| 502 Bad Gateway | `sudo systemctl restart neuroscan` |
| Nginx default page showing | `sudo rm /etc/nginx/sites-enabled/default && sudo systemctl restart nginx` |
| App not starting | `sudo journalctl -u neuroscan -f` to see errors |
| Permission denied on socket | `sudo chmod 775 /home/ubuntu/neuroscan` |
| SSL not working | Check port 443 is open in EC2 Security Group |
| Site down after reboot | `sudo systemctl enable neuroscan` |

---

## Cost Breakdown

| Resource | Cost |
|---|---|
| EC2 t3.medium (running) | ~$0.04/hr (~$30/month) |
| EC2 t3.medium (stopped) | ~$0/hr |
| 20GB EBS Storage | ~$2/month |
| Elastic IP (attached, running) | Free |
| Elastic IP (instance stopped) | $0.005/hr |
| Let's Encrypt SSL | Free |
| No-IP domain | Free |

> 💡 **Tip:** Stop the instance when not in use to save costs. The site will be down but you only pay ~$2/month for storage.

### Key Deployment Info

| Item | Value |
|---|---|
| Live URL | https://neuroscan.ddns.net |
| Elastic IP | 54.211.52.67 |
| EC2 Instance ID | i-01d908815b489b8f6 |
| Region | us-east-1 (N. Virginia) |
| SSH Key | `~/.ssh/neuroscan-key.pem` |

---

> ⚠️ **Disclaimer:** NeuroScan AI is a clinical research tool only. All diagnoses must be confirmed by a licensed neurologist. Not intended for direct patient care decisions.