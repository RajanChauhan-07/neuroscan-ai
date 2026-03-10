import os, json, pickle, io, base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import timm
import torchvision.transforms as transforms

application = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_A_PATH  = os.path.join(BASE_DIR, 'models', 'model_A.pth')
MODEL_B_PATH  = os.path.join(BASE_DIR, 'models', 'model_B.pkl')
LABELS_PATH   = os.path.join(BASE_DIR, 'models', 'class_labels.json')

# ── Device ────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load Model A (EfficientNetB3) ─────────────────────────
with open(LABELS_PATH) as f:
    CLASS_LABELS = json.load(f)

model_A = timm.create_model('efficientnet_b3', pretrained=False, num_classes=4)
model_A.load_state_dict(torch.load(MODEL_A_PATH, map_location=DEVICE))
model_A.to(DEVICE)
model_A.eval()

# ── Load Model B (XGBoost) ────────────────────────────────
with open(MODEL_B_PATH, 'rb') as f:
    model_B = pickle.load(f)

# ── Image Transform ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Stage metadata ────────────────────────────────────────
STAGE_META = {
    'NonDemented': {
        'label': 'No Dementia',
        'color': '#34C759',
        'risk':  'Low',
        'description': 'No signs of cognitive decline detected in the MRI scan.'
    },
    'VeryMildDemented': {
        'label': 'Very Mild Dementia',
        'color': '#FF9F0A',
        'risk':  'Moderate',
        'description': 'Very early signs present. Close monitoring recommended.'
    },
    'MildDemented': {
        'label': 'Mild Dementia',
        'color': '#FF6B35',
        'risk':  'High',
        'description': 'Mild cognitive impairment detected. Treatment advised.'
    },
    'ModerateDemented': {
        'label': 'Moderate Dementia',
        'color': '#FF3B30',
        'risk':  'Critical',
        'description': 'Moderate dementia confirmed. Immediate care required.'
    }
}

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/analyze', methods=['POST'])
def analyze():
    try:
        # ── Model A: MRI Analysis ─────────────────────────
        file    = request.files['mri']
        img     = Image.open(file.stream).convert('RGB')
        tensor  = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model_A(tensor)
            probs   = torch.softmax(outputs, dim=1)[0]
            pred_idx = probs.argmax().item()

        pred_class = CLASS_LABELS[str(pred_idx)]
        confidence = float(probs[pred_idx]) * 100
        meta       = STAGE_META[pred_class]

        all_probs = {
            CLASS_LABELS[str(i)]: round(float(probs[i]) * 100, 2)
            for i in range(4)
        }

        # ── Model B: Tabular Risk Analysis ────────────────
        fields = [
            'age','gender','ethnicity','education','bmi','smoking',
            'alcohol','physical_activity','diet_quality','sleep_quality',
            'family_history','cardiovascular','diabetes','depression',
            'head_injury','hypertension','systolic_bp','diastolic_bp',
            'cholesterol_total','cholesterol_ldl','cholesterol_hdl',
            'cholesterol_trig','mmse','functional_assessment',
            'memory_complaints','behavioral_problems','adl',
            'confusion','disorientation','personality_changes',
            'difficulty_tasks','forgetfulness'
        ]

        row = []
        for f_name in fields:
            val = request.form.get(f_name, 0)
            row.append(float(val) if val != '' else 0.0)

        X = np.array(row).reshape(1, -1)
        risk_prob    = float(model_B.predict_proba(X)[0][1]) * 100
        risk_label   = 'High' if risk_prob >= 60 else ('Moderate' if risk_prob >= 35 else 'Low')
        risk_color   = '#FF3B30' if risk_label == 'High' else ('#FF9F0A' if risk_label == 'Moderate' else '#34C759')

        # ── Unified Report ────────────────────────────────
        report = generate_report(pred_class, confidence, risk_prob, risk_label)

        return jsonify({
            'success': True,
            'mri': {
                'stage':       meta['label'],
                'stage_key':   pred_class,
                'confidence':  round(confidence, 1),
                'color':       meta['color'],
                'description': meta['description'],
                'all_probs':   all_probs
            },
            'risk': {
                'probability': round(risk_prob, 1),
                'label':       risk_label,
                'color':       risk_color
            },
            'report': report
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def generate_report(stage, confidence, risk_prob, risk_label):
    stage_map = {
        'NonDemented':       'no signs of dementia',
        'VeryMildDemented':  'very mild dementia indicators',
        'MildDemented':      'mild dementia',
        'ModerateDemented':  'moderate dementia'
    }
    progression_map = {
        'High':     'a high risk of progression within 12 months',
        'Moderate': 'a moderate risk of progression within 12 months',
        'Low':      'a low risk of near-term progression'
    }
    recommendation_map = {
        ('NonDemented',      'Low'):      'Annual cognitive screening is sufficient at this stage.',
        ('NonDemented',      'Moderate'): 'Despite a clean MRI, clinical risk factors warrant a 6-month follow-up.',
        ('NonDemented',      'High'):     'MRI appears normal, but elevated clinical risk factors require immediate neurological evaluation.',
        ('VeryMildDemented', 'Low'):      'Early monitoring and lifestyle interventions are recommended.',
        ('VeryMildDemented', 'Moderate'): 'Cognitive therapy and 6-month neuroimaging follow-up advised.',
        ('VeryMildDemented', 'High'):     'Prompt specialist referral and treatment planning recommended.',
        ('MildDemented',     'Low'):      'Begin cognitive support therapy and 6-month check-ins.',
        ('MildDemented',     'Moderate'): 'Pharmacological and non-pharmacological interventions should begin immediately.',
        ('MildDemented',     'High'):     'Urgent specialist consultation required. Consider memory care planning.',
        ('ModerateDemented', 'Low'):      'Full care support and specialist management required.',
        ('ModerateDemented', 'Moderate'): 'Comprehensive care plan and family counselling strongly advised.',
        ('ModerateDemented', 'High'):     'Immediate specialist intervention and full-time care assessment required.',
    }

    key = (stage, risk_label)
    recommendation = recommendation_map.get(key, 'Please consult a specialist for a comprehensive evaluation.')

    return (
        f"The MRI analysis ({confidence:.1f}% confidence) indicates {stage_map[stage]}. "
        f"Clinical data analysis shows {progression_map[risk_label]} (probability: {risk_prob:.1f}%). "
        f"{recommendation}"
    )


if __name__ == '__main__':
    application.run(debug=True, port=5000)