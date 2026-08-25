import pandas as pd
import torch
import os
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import re
import numpy as np
import math
import time
from sklearn.model_selection import KFold


def count_parameters(model, print_detail=False):
    total_params = 0
    trainable_params = 0
    param_dict = {}

    for name, param in model.named_parameters():
        param_num = param.numel()
        total_params += param_num
        if param.requires_grad:
            trainable_params += param_num
        if print_detail:
            param_dict[name] = f"{param_num / 1e6:.4f}M"

    total_m = total_params / 1e6
    trainable_m = trainable_params / 1e6

    if print_detail:
        print("=" * 50)
        print(f"Model Parameter Summary (M):")
        print(f"Total: {total_m:.2f} | Trainable: {trainable_m:.2f}")
        print("=" * 50)
    return total_m, trainable_m, param_dict


def measure_inference_time(model, data_loader, repeat=10, warmup=3, device="cuda"):
    model.eval()
    model.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device)
                _ = model(x_batch)
                break

    times = []
    with torch.no_grad():
        for _ in range(repeat):
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device)
                start = time.time()
                _ = model(x_batch)
                end = time.time()
                infer_time = (end - start) * 1000
                times.append(infer_time)
                break

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n[推理时间统计] 平均耗时：{avg_time:.2f}ms | 标准差：{std_time:.2f}ms (n={repeat})")
    print(f"✓ 满足临床实时要求（＜100ms）：{avg_time < 100}")
    return avg_time, std_time

df = pd.read_csv(r'.\param_embeddings\node_embeddings_final_kstr_128.csv')
embedding_dict = {}
for _, row in df.iterrows():
    node_type = row['node_type']
    node_id = row['node_id']
    emb = torch.tensor([float(x) for x in row['embedding'].split()])
    embedding_dict.setdefault(node_type, {})[node_id] = emb

symptom_map_df = pd.read_csv(r".\STP_Subset\entity_224\symptomID_723.csv", dtype=str)
herb_map_df = pd.read_csv(r".\STP_Subset\entity_224\herbID_723.csv", dtype=str)
syndrome_map_df = pd.read_csv(r".\STP_Subset\entity_224\syndromeID_723.csv", dtype=str)
tongue_map_df = pd.read_csv(r".\STP_Subset\entity_224\tongueID_723.csv", dtype=str)
pulse_map_df = pd.read_csv(r".\STP_Subset\entity_224\pulseID_723.csv", dtype=str)

symptom_name_to_id = dict(zip(symptom_map_df['symptom_name'].str.strip(), symptom_map_df['ID'].str.strip()))
herb_name_to_id = dict(zip(herb_map_df['herb_name'].str.strip(), herb_map_df['ID'].str.strip()))
syndrome_name_to_id = dict(zip(syndrome_map_df['syndrome_name'].str.strip(), syndrome_map_df['ID'].str.strip()))
tongue_name_to_id = dict(zip(tongue_map_df['tongue_name'].str.strip(), tongue_map_df['ID'].str.strip()))
pulse_name_to_id = dict(zip(pulse_map_df['pulse_name'].str.strip(), pulse_map_df['ID'].str.strip()))

train_df = pd.read_excel(r".\STP_Subset\stp_sub.xlsx", dtype=str)

def safe_parse_string(s):
    if pd.isna(s):
        return ""
    return str(s).strip()


def parse_text_field(text):
    items = [item.strip() for item in text.split('，') if item.strip()]
    return [item for item in items if not re.search(r'\d+剂', item)]


def extract_clean_herb_name(s):
    match = re.match(r'^([\u4e00-\u9fa5]+)', s.strip())
    if match:
        return match.group(1)
    return s.strip()


def extract_clean_herb_names(herb_text):
    herbs = []
    for med in parse_text_field(herb_text):
        name = extract_clean_herb_name(med)
        if name and not re.search(r'\d+剂', name):
            herbs.append(name)
    return herbs


samples = []
total_rows = len(train_df)
empty_field_rows = []
symptom_miss_rows = []
herb_miss_rows = []

unmatched_symptom_names = Counter()
unmatched_herb_names = Counter()
unmatched_zheng_names = Counter()
unmatched_tongue_names = Counter()
unmatched_pulse_names = Counter()

for idx, row in train_df.iterrows():
    symptom_text = safe_parse_string(row.get('其余症状', ''))
    herb_text = safe_parse_string(row.get('处方', ''))
    syndrome_text = safe_parse_string(row.get('证候', ''))
    tongue_text = safe_parse_string(row.get('舌像', ''))
    pulse_text = safe_parse_string(row.get('脉象', ''))

    if not symptom_text or not herb_text or (not tongue_text and not pulse_text):
        empty_field_rows.append(idx)
        continue

    symptom_names = parse_text_field(symptom_text)
    herb_names = extract_clean_herb_names(herb_text)
    syndrome_names = parse_text_field(syndrome_text) if syndrome_text else []
    tongue_names = parse_text_field(tongue_text) if tongue_text else []
    pulse_names = parse_text_field(pulse_text) if pulse_text else []

    symptom_ids = []
    for name in symptom_names:
        _id = symptom_name_to_id.get(name)
        if _id is None:
            unmatched_symptom_names[name] += 1
        symptom_ids.append(_id)

    herb_ids = []
    for name in herb_names:
        _id = herb_name_to_id.get(name)
        if _id is None:
            unmatched_herb_names[name] += 1
        herb_ids.append(_id)

    syndrome_ids = [syndrome_name_to_id.get(name, None) for name in syndrome_names]
    tongue_ids = [tongue_name_to_id.get(name, None) for name in tongue_names]
    pulse_ids = [pulse_name_to_id.get(name, None) for name in pulse_names]

    if any(x is None for x in symptom_ids):
        symptom_miss_rows.append(idx)
        continue
    if any(x is None for x in herb_ids):
        herb_miss_rows.append(idx)
        continue

    samples.append({
        'symptoms': [x for x in symptom_ids if x is not None],
        'tongue': [x for x in tongue_ids if x is not None],
        'pulse': [x for x in pulse_ids if x is not None],
        'syndrome': [x for x in syndrome_ids if x is not None],
        'herbs': [x for x in herb_ids if x is not None]
    })

train_all, val_test = train_test_split(samples, test_size=0.2, random_state=42)
val_samples, test_samples = train_test_split(val_test, test_size=0.5, random_state=42)

herb_freq_counter = Counter()
for sample in train_all:
    herb_freq_counter.update(sample['herbs'])

LOW_FREQ_THRESHOLD = 3
low_freq_herbs = {herb_id: count for herb_id, count in herb_freq_counter.items() if count < LOW_FREQ_THRESHOLD}

low_freq_samples = []
normal_samples = []
for sample in samples:
    has_low_freq = any(herb in low_freq_herbs for herb in sample['herbs'])
    if has_low_freq:
        low_freq_samples.append(sample)
    else:
        normal_samples.append(sample)

train_low_freq = [s for s in train_all if any(herb in low_freq_herbs for herb in s['herbs'])]
train_normal = [s for s in train_all if not any(herb in low_freq_herbs for herb in s['herbs'])]

REPEAT_FACTOR = 3

augmented_train_low_freq = train_low_freq * REPEAT_FACTOR
final_train_samples = train_normal + augmented_train_low_freq
random.shuffle(final_train_samples)

node_to_idx = defaultdict(dict)
embedding_matrices = {}
for node_type in embedding_dict:
    id_list = list(embedding_dict[node_type].keys())
    node_to_idx[node_type] = {nid: i for i, nid in enumerate(id_list)}
    embedding_matrices[node_type] = torch.stack([embedding_dict[node_type][nid] for nid in id_list])

missing = {'symptom': set(), 'herb': set(), 'syndrome': set(), 'tongue': set(), 'pulse': set()}
for s in samples:
    for sid in s['symptoms']:
        if sid not in node_to_idx['symptom']:
            missing['symptom'].add(sid)
    for hid in s['herbs']:
        if hid not in node_to_idx['herb']:
            missing['herb'].add(hid)
    for synd in s['syndrome']:
        if synd not in node_to_idx['syndrome']:
            missing['syndrome'].add(synd)
    for tid in s['tongue']:
        if tid not in node_to_idx['tongue']:
            missing['tongue'].add(tid)
    for pid in s['pulse']:
        if pid not in node_to_idx['pulse']:
            missing['pulse'].add(pid)

class HerbDataset(Dataset):
    def __init__(self, samples, node_to_idx, embedding_dict, embedding_dim,
                 encoder=None, syndrome_model=None, use_pred_syndrome=False, topk_syndrome=3):
        self.samples = samples
        self.node_to_idx = node_to_idx
        self.embedding_dict = embedding_dict
        self.embedding_dim = embedding_dim
        self.encoder = encoder
        self.syndrome_model = syndrome_model
        self.use_pred_syndrome = use_pred_syndrome
        self.topk_syndrome = topk_syndrome
        self.idx_to_syndrome = {v: k for k, v in node_to_idx['syndrome'].items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        x_symptom = torch.stack([
            self.embedding_dict['symptom'][sid]
            for sid in sample['symptoms']
        ]).mean(dim=0)

        if sample['tongue']:
            tongue_emb = torch.stack([
                self.embedding_dict['tongue'][tid]
                for tid in sample['tongue']
            ]).mean(dim=0)
        else:
            tongue_emb = torch.zeros(self.embedding_dim)

        if sample['pulse']:
            pulse_emb = torch.stack([
                self.embedding_dict['pulse'][pid]
                for pid in sample['pulse']
            ]).mean(dim=0)
        else:
            pulse_emb = torch.zeros(self.embedding_dim)

        if self.use_pred_syndrome:
            with torch.no_grad():
                input_feat = torch.cat([x_symptom, tongue_emb, pulse_emb], dim=-1)
                input_feat = input_feat.unsqueeze(0)
                if input_feat.dim() == 1:
                    input_feat = input_feat.unsqueeze(0)
                logits = self.syndrome_model(input_feat)
                topk = torch.topk(logits, k=self.topk_syndrome).indices
                s_embs = []
                for idx in topk:
                    sid = self.idx_to_syndrome.get(idx.item())
                    if sid in self.embedding_dict['syndrome']:
                        s_embs.append(self.embedding_dict['syndrome'][sid])
                x_syndrome = torch.stack(s_embs).mean(dim=0) if s_embs else torch.zeros(self.embedding_dim)
        else:
            s_embs = [self.embedding_dict['syndrome'][sid] for sid in sample['syndrome']
                      if sid in self.embedding_dict['syndrome']]
            x_syndrome = torch.stack(s_embs).mean(dim=0) if s_embs else torch.zeros(self.embedding_dim)

        x_raw = torch.cat([x_symptom, tongue_emb, pulse_emb, x_syndrome], dim=-1)
        x_input = x_raw

        y = torch.zeros(len(self.node_to_idx['herb']))
        for hid in sample['herbs']:
            hidx = self.node_to_idx['herb'].get(hid)
            if hidx is not None:
                y[hidx] = 1

        return x_input, y

class MultiModalContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, main_model_hidden_dim, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.main_model_hidden_dim = main_model_hidden_dim

        self.symptom_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        self.tongue_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        self.pulse_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )

        self.modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=2,
            batch_first=True
        )

        self.fusion_encoder = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, 3 * input_dim)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(3 * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        B = x.shape[0]
        symptom_feat = x[:, :self.input_dim]
        tongue_feat = x[:, self.input_dim:2 * self.input_dim]
        pulse_feat = x[:, 2 * self.input_dim:]

        s = self.symptom_encoder(symptom_feat)
        t = self.tongue_encoder(tongue_feat)
        p = self.pulse_encoder(pulse_feat)

        modal_feats = torch.stack([s, t, p], dim=1)
        attn_output, _ = self.modal_attention(modal_feats, modal_feats, modal_feats)
        s_attn, t_attn, p_attn = attn_output[:, 0], attn_output[:, 1], attn_output[:, 2]

        fused = torch.cat([s_attn, t_attn, p_attn], dim=-1)
        fused = self.fusion_encoder(fused)
        z = self.projection_head(fused)
        return F.normalize(z, dim=-1)


class ContrastivePairDataset(Dataset):
    def __init__(self, samples, node_to_idx, embedding_dict, embedding_dim,
                 num_pairs=10000, hard_neg_ratio=0.3, alpha=3, beta=1):
        self.embedding_dict = embedding_dict
        self.embedding_dim = embedding_dim
        self.samples = samples
        self.hard_neg_ratio = hard_neg_ratio
        self.alpha = alpha
        self.beta = beta
        self.herb_to_samples = self._build_herb_index()
        self.pairs = self._create_pairs(num_pairs)

    def _build_herb_index(self):
        herb_index = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            for herb in sample['herbs']:
                herb_index[herb].append(idx)
        return herb_index

    def _sample_embedding(self, sample):

        if sample['symptoms']:
            symptom_embs = [self.embedding_dict['symptom'][sid] for sid in sample['symptoms']]
            symptom_vec = torch.stack(symptom_embs).mean(dim=0)
        else:
            symptom_vec = torch.zeros(self.embedding_dim)

        if sample['tongue']:
            tongue_embs = [self.embedding_dict['tongue'][tid] for tid in sample['tongue']]
            tongue_vec = torch.stack(tongue_embs).mean(dim=0)
        else:
            tongue_vec = torch.zeros(self.embedding_dim)

        if sample['pulse']:
            pulse_embs = [self.embedding_dict['pulse'][pid] for pid in sample['pulse']]
            pulse_vec = torch.stack(pulse_embs).mean(dim=0)
        else:
            pulse_vec = torch.zeros(self.embedding_dim)

        combined = torch.cat([symptom_vec, tongue_vec, pulse_vec], dim=-1)
        return F.normalize(combined, dim=-1)

    def _get_positive_sample(self, sample):
        sample_herbs = set(sample['herbs'])
        sample_syndrome = set(sample['syndrome'])
        candidates = set()

        for herb in sample_herbs:
            for idx in self.herb_to_samples.get(herb, []):
                candidate = self.samples[idx]
                if candidate is sample:
                    continue
                herb_overlap = len(sample_herbs & set(candidate['herbs']))
                if herb_overlap >= self.alpha:
                    candidates.add(idx)

        if sample_syndrome:
            for idx, candidate in enumerate(self.samples):
                if candidate is sample:
                    continue
                if sample_syndrome & set(candidate['syndrome']):
                    candidates.add(idx)

        if candidates:
            return self.samples[random.choice(list(candidates))]
        return None

    def _get_hard_negative(self, sample):
        sample_herbs = set(sample['herbs'])
        sample_syndrome = set(sample['syndrome'])
        sample_symptoms = set(sample['symptoms'])
        sample_tongue = set(sample['tongue'])
        sample_pulse = set(sample['pulse'])

        candidates = []
        for candidate in self.samples:
            if candidate is sample:
                continue

            candidate_syndrome = set(candidate['syndrome'])
            syndrome_overlap = len(sample_syndrome & candidate_syndrome)
            herb_overlap = len(sample_herbs & set(candidate['herbs']))

            if herb_overlap >= self.alpha or syndrome_overlap > 0:
                continue

            diagnostic_overlap = (
                len(sample_symptoms & set(candidate['symptoms']))
                + len(sample_tongue & set(candidate['tongue']))
                + len(sample_pulse & set(candidate['pulse']))
            )

            if diagnostic_overlap >= self.beta and syndrome_overlap == 0:
                candidates.append(candidate)

        if candidates:
            return random.choice(candidates)
        return None

    def _create_pairs(self, num_pairs):

        pairs = []
        num_hard_neg = int(num_pairs * self.hard_neg_ratio)
        num_random_neg = num_pairs - num_hard_neg

        for _ in range(num_pairs):
            while True:
                s1 = random.choice(self.samples)
                s2 = self._get_positive_sample(s1)
                if s2 is not None:
                    break
            x1 = self._sample_embedding(s1)
            x2 = self._sample_embedding(s2)
            pairs.append((x1, x2, 1))

        for _ in range(num_hard_neg):
            while True:
                s1 = random.choice(self.samples)
                s2 = self._get_hard_negative(s1)
                if s2 is not None:
                    break
            x1 = self._sample_embedding(s1)
            x2 = self._sample_embedding(s2)
            pairs.append((x1, x2, 0))

        for _ in range(num_random_neg):
            while True:
                s1 = random.choice(self.samples)
                s2 = random.choice(self.samples)
                if s2 is s1:
                    continue

                herb_overlap = len(set(s1['herbs']) & set(s2['herbs']))
                syndrome_overlap = len(set(s1['syndrome']) & set(s2['syndrome']))

                is_positive = (herb_overlap >= self.alpha) or (syndrome_overlap > 0)
                if is_positive:
                    continue

                diagnostic_overlap = (
                    len(set(s1['symptoms']) & set(s2['symptoms']))
                    + len(set(s1['tongue']) & set(s2['tongue']))
                    + len(set(s1['pulse']) & set(s2['pulse']))
                )
                is_hard_negative = (diagnostic_overlap >= self.beta) and (syndrome_overlap == 0)

                if not is_hard_negative:
                    break

            x1 = self._sample_embedding(s1)
            x2 = self._sample_embedding(s2)
            pairs.append((x1, x2, 0))

        random.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

herb_counts = defaultdict(int)
for sample in samples:
    for herb in sample['herbs']:
        herb_counts[herb] += 1
herb_counts = dict(herb_counts)

class_weights = torch.ones(len(node_to_idx['herb']))
for herb_id, count in herb_counts.items():
    idx = node_to_idx['herb'].get(herb_id)
    if idx is not None and idx < len(class_weights):
        class_weights[idx] = torch.log(torch.tensor(count + 1.0))
    else:
        print(f"herb_id={herb_id} 映射到无效 idx={idx}，跳过")

class_weights_tensor = class_weights


class LabelAwareAttention(nn.Module):
    def __init__(self, hidden_dim, num_labels):
        super(LabelAwareAttention, self).__init__()
        self.label_queries = nn.Parameter(torch.randn(num_labels, hidden_dim))
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        keys = self.linear_k(x).unsqueeze(1)
        values = self.linear_v(x).unsqueeze(1)
        queries = self.label_queries.unsqueeze(0)

        scores = torch.matmul(queries, keys.transpose(-1, -2)) / (x.size(-1) ** 0.5)
        attn_weights = torch.softmax(scores, dim=1)
        output = (attn_weights * values).squeeze(2)
        return output


class RecommendationModelWithTransformerAndLabelAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=2, temperature=1.0):
        super(RecommendationModelWithTransformerAndLabelAttention, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention = LabelAwareAttention(hidden_dim, output_dim)

        self.dropout = nn.Dropout(0.3)
        self.temperature = temperature

        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        x = self.embedding(x).unsqueeze(1)
        x = self.encoder(x).squeeze(1)

        label_reps = self.attention(x)
        logits = self.output_proj(self.dropout(label_reps)).squeeze(-1)

        return logits / self.temperature


class TemperatureScheduler:
    def __init__(self, initial_temperature, decay_rate=0.99):
        self.temperature = initial_temperature
        self.decay_rate = decay_rate

    def step(self):
        self.temperature *= self.decay_rate

    def get_temperature(self):
        return self.temperature


embedding_dim = next(iter(embedding_dict['syndrome'].values())).shape[0]
dataset = HerbDataset(samples, node_to_idx, embedding_dict, embedding_dim)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets, gamma):
        probs = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t) ** gamma
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_factor * focal_factor * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def dynamic_gamma2(epoch, total_epochs, gamma_start=3.0, gamma_end=1.0):
    ratio = epoch / (total_epochs - 1)
    return gamma_start + (gamma_end - gamma_start) * ratio


class PrescriptionLoss(nn.Module):
    def __init__(self, alpha=0.25, lambda_coeff=0.05, reduction='mean', eps=1e-8):

        super(PrescriptionLoss, self).__init__()
        self.alpha = alpha
        self.lambda_coeff = lambda_coeff
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):

        probs = torch.sigmoid(inputs)

        Pe = probs * targets + (1 - probs) * (1 - targets)
        Pe = torch.clamp(Pe, self.eps, 1.0)

        log_Pe = torch.log(Pe)
        loss = self.alpha * (-log_Pe + self.lambda_coeff * Pe * log_Pe)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SyndromePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_syndromes,
                 symptom_dim, tongue_dim, pulse_dim,
                 dropout_rate=0.3):
        super().__init__()
        self.symptom_dim = symptom_dim
        self.tongue_dim = tongue_dim
        self.pulse_dim = pulse_dim

        assert symptom_dim + tongue_dim + pulse_dim == input_dim, \
            "各模态维度之和必须等于总输入维度"

        self.symptom_encoder = nn.Sequential(
            nn.Linear(symptom_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2, track_running_stats=True),
            nn.GELU()
        )

        self.tongue_encoder = nn.Sequential(
            nn.Linear(tongue_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, track_running_stats=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, track_running_stats=True),
            nn.GELU()
        )

        self.pulse_encoder = nn.Sequential(
            nn.Linear(pulse_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, track_running_stats=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, track_running_stats=True),
            nn.GELU()
        )

        self.modality_attention = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )

        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2, track_running_stats=True),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.feature_extractor = nn.Sequential(
            ResidualBlock(hidden_dim * 2, hidden_dim * 2, track_running_stats=True),
            nn.Dropout(dropout_rate),
            ResidualBlock(hidden_dim * 2, hidden_dim, track_running_stats=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2, track_running_stats=True),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_syndromes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_input):

        if x_input.dim() == 1:
            x_input = x_input.unsqueeze(0)

        training_mode = self.training
        if self.training and x_input.size(0) == 1:
            self.eval()

        symptom_feat = x_input[:, :self.symptom_dim]
        tongue_feat = x_input[:, self.symptom_dim:self.symptom_dim + self.tongue_dim]
        pulse_feat = x_input[:, self.symptom_dim + self.tongue_dim:]

        s_encoded = self.symptom_encoder(symptom_feat)
        t_encoded = self.tongue_encoder(tongue_feat)
        p_encoded = self.pulse_encoder(pulse_feat)

        ref_feat = torch.mean(torch.cat([t_encoded.unsqueeze(1), p_encoded.unsqueeze(1)], dim=1), dim=1)
        modality_weights = self.modality_attention(ref_feat)

        s_weighted = s_encoded * modality_weights[:, 0].unsqueeze(1)
        t_weighted = t_encoded * modality_weights[:, 1].unsqueeze(1)
        p_weighted = p_encoded * modality_weights[:, 2].unsqueeze(1)

        fused = torch.cat([s_weighted, t_weighted, p_weighted], dim=1)
        fused = self.cross_modal_fusion(fused)

        x = self.feature_extractor(fused)

        logits = self.classifier(x)

        if training_mode and x_input.size(0) == 1:
            self.train()

        if x_input.size(0) == 1:
            logits = logits.squeeze(0)

        return logits


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, track_running_stats=True):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)

        self.bn1 = nn.BatchNorm1d(out_dim, track_running_stats=track_running_stats)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim, track_running_stats=track_running_stats)

        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim, track_running_stats=track_running_stats)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.bn2(x)

        x += residual
        return self.activation(x)



class SyndromeSupervisedDataset(Dataset):
    def __init__(self, samples, embedding_dict, node_to_idx, embedding_dim):
        self.samples = samples
        self.embedding_dict = embedding_dict
        self.node_to_idx = node_to_idx
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x_symptom = torch.stack([
            self.embedding_dict['symptom'][sid]
            for sid in sample['symptoms']
        ]).mean(dim=0)
        if sample['tongue']:
            x_tongue = torch.stack([
                self.embedding_dict['tongue'][tid]
                for tid in sample['tongue']
            ]).mean(dim=0)
        else:
            x_tongue = torch.zeros(self.embedding_dim)

        if sample['pulse']:
            x_pulse = torch.stack([
                self.embedding_dict['pulse'][pid]
                for pid in sample['pulse']
            ]).mean(dim=0)
        else:
            x_pulse = torch.zeros(self.embedding_dim)

        x_input = torch.cat([x_symptom, x_tongue, x_pulse], dim=-1)

        y = torch.zeros(len(self.node_to_idx['syndrome']))
        for sid in sample['syndrome']:
            s_idx = self.node_to_idx['syndrome'].get(sid)
            if s_idx is not None:
                y[s_idx] = 1
        return x_input, y



def evaluate_precision_recall_f1(model, loader, k_list=[5, 10, 15]):
    model.eval()
    results = {k: {"precision": [], "recall": []} for k in k_list}
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            probs = torch.sigmoid(logits)
            for probs_row, true_row in zip(probs, y_batch):
                true_indices = set((true_row > 0).nonzero(as_tuple=True)[0].tolist())
                if not true_indices:
                    continue
                for k in k_list:
                    topk_indices = set(torch.topk(probs_row, k).indices.tolist())
                    hits = len(topk_indices & true_indices)
                    precision = hits / k
                    recall = hits / len(true_indices)
                    results[k]["precision"].append(precision)
                    results[k]["recall"].append(recall)

    for k in k_list:
        p = sum(results[k]["precision"]) / len(results[k]["precision"])
        r = sum(results[k]["recall"]) / len(results[k]["recall"])
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        print(f"Top-{k} → Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")


def collect_pt_distribution(model, dataloader):
    model.eval()
    pt_list = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            logits = model(x_batch)
            probs = torch.sigmoid(logits)
            p_t = probs * y_batch + (1 - probs) * (1 - y_batch)
            pt_list.append(p_t)

    all_pt = torch.cat(pt_list, dim=0).cpu().flatten().numpy()
    return all_pt

def enhanced_nt_xent_loss(z1, z2, labels, temperature=0.5, lambda_modal=0.1):

    B = z1.size(0)
    positive_mask = labels.bool()

    sim_matrix = torch.matmul(z1, z2.T) / temperature
    targets = torch.arange(B, device=z1.device)

    if positive_mask.any():
        loss_12 = F.cross_entropy(
            sim_matrix[positive_mask],
            targets[positive_mask],
            reduction='none'
        )
        loss_21 = F.cross_entropy(
            sim_matrix.T[positive_mask],
            targets[positive_mask],
            reduction='none'
        )
        info_nce = 0.5 * (loss_12 + loss_21)

        pair_sim = F.cosine_similarity(z1, z2, dim=1)
        hard_weight = 1 - torch.sigmoid(pair_sim[positive_mask] / temperature)
        ce_loss = (info_nce * hard_weight).sum() / hard_weight.sum().clamp(min=1e-8)

        modal_consistency = 1 - pair_sim[positive_mask].mean()
    else:
        ce_loss = z1.sum() * 0.0
        modal_consistency = z1.sum() * 0.0
    total_loss = ce_loss + lambda_modal * modal_consistency
    return total_loss


main_model_hidden_dim = 64
encoder_contrastive = MultiModalContrastiveEncoder(input_dim=3 * embedding_dim,
                                                   main_model_hidden_dim=main_model_hidden_dim)
encoder_herb = MultiModalContrastiveEncoder(input_dim=4 * embedding_dim, main_model_hidden_dim=main_model_hidden_dim)
encoder = MultiModalContrastiveEncoder(input_dim=4 * embedding_dim, main_model_hidden_dim=main_model_hidden_dim)
contrastive_dataset = ContrastivePairDataset(final_train_samples, node_to_idx, embedding_dict, embedding_dim)
contrastive_loader = DataLoader(contrastive_dataset, batch_size=64, shuffle=True)
optimizer = torch.optim.Adam(encoder_contrastive.parameters(), lr=1e-3)


def train_contrastive_encoder(encoder, train_loader, val_loader, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(epochs):
        encoder.train()
        train_loss = 0.0
        for x1, x2, labels in train_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

            z1 = encoder(x1)
            z2 = encoder(x2)
            loss = enhanced_nt_xent_loss(z1, z2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x1.size(0)

        encoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x1, x2, labels in val_loader:
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
                z1 = encoder(x1)
                z2 = encoder(x2)
                loss = enhanced_nt_xent_loss(z1, z2, labels)
                val_loss += loss.item() * x1.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        scheduler.step()

        print(f"[Contrastive] Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), "../github_op/best_contrastive_encoder.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"早停于Epoch {epoch + 1}")
                break

    encoder.load_state_dict(torch.load("../open_1/best_contrastive_encoder.pth"))
    return encoder

def transfer_contrastive_weights(contrastive_encoder, target_linear_layer, embedding_dim):

    with torch.no_grad():
        fusion_linear = contrastive_encoder.fusion_encoder[-1]
        fusion_weight = fusion_linear.weight
        fusion_bias = fusion_linear.bias
        fusion_weight_t = fusion_weight.T
        target_weight = target_linear_layer.weight
        target_out_dim, target_in_dim = target_weight.shape
        transfer_dim = 3 * embedding_dim
        fusion_weight_t_cropped = fusion_weight_t[:target_out_dim, :]
        transfer_ratio = transfer_dim / target_in_dim
        target_weight[:, :transfer_dim] = fusion_weight_t_cropped[:, :transfer_dim] * transfer_ratio
        target_bias = target_linear_layer.bias
        target_bias.copy_(fusion_bias[:target_out_dim])

    return target_linear_layer



contrastive_train_samples, contrastive_val_samples = train_test_split(
    train_all,
    test_size=0.1,
    random_state=42
)


contrastive_train_dataset = ContrastivePairDataset(
    samples=contrastive_train_samples,
    node_to_idx=node_to_idx,
    embedding_dict=embedding_dict,
    embedding_dim=embedding_dim,
    num_pairs=10000,
    hard_neg_ratio=0.3
)

contrastive_val_dataset = ContrastivePairDataset(
    samples=contrastive_val_samples,
    node_to_idx=node_to_idx,
    embedding_dict=embedding_dict,
    embedding_dim=embedding_dim,
    num_pairs=2000,
    hard_neg_ratio=0.3
)


contrastive_train_loader = DataLoader(contrastive_train_dataset, batch_size=64, shuffle=True)
contrastive_val_loader = DataLoader(contrastive_val_dataset, batch_size=64, shuffle=False)
# === [Stage 1] ===
encoder_contrastive = MultiModalContrastiveEncoder(
    input_dim=embedding_dim,
    main_model_hidden_dim=main_model_hidden_dim
)

encoder_contrastive = train_contrastive_encoder(
    encoder=encoder_contrastive,
    train_loader=contrastive_train_loader,
    val_loader=contrastive_val_loader,
    epochs=20,
    lr=1e-3
)

# === [Stage 2] ===
syndrome_dataset = SyndromeSupervisedDataset(final_train_samples, embedding_dict, node_to_idx, embedding_dim)
syndrome_loader = DataLoader(syndrome_dataset, batch_size=32, shuffle=True)

num_syndromes = len(node_to_idx['syndrome'])
syndrome_model = SyndromePredictor(input_dim=3 * embedding_dim, hidden_dim=128, num_syndromes=num_syndromes,
                                   symptom_dim=embedding_dim, tongue_dim=embedding_dim, pulse_dim=embedding_dim)
optimizer_syndrome = torch.optim.Adam(syndrome_model.parameters(), lr=1e-3)
loss_fn_syndrome = nn.BCEWithLogitsLoss()

print("\n[Stage 2] ")
for epoch in range(20):
    syndrome_model.train()
    total_loss = 0
    for x, y in syndrome_loader:
        logits = syndrome_model(x)
        loss = loss_fn_syndrome(logits, y)
        optimizer_syndrome.zero_grad()
        loss.backward()
        optimizer_syndrome.step()
        total_loss += loss.item()
    print(f"[SyndromeModel] Epoch {epoch + 1}, Loss: {total_loss / len(syndrome_loader):.4f}")

train_dataset = HerbDataset(final_train_samples, node_to_idx, embedding_dict, embedding_dim,
                            encoder=encoder, syndrome_model=syndrome_model,
                            use_pred_syndrome=True, topk_syndrome=3)
val_dataset = HerbDataset(val_samples, node_to_idx, embedding_dict, embedding_dim,
                          encoder=encoder, syndrome_model=syndrome_model,
                          use_pred_syndrome=True, topk_syndrome=3)
test_dataset = HerbDataset(test_samples, node_to_idx, embedding_dict, embedding_dim,
                           encoder=encoder, syndrome_model=syndrome_model,
                           use_pred_syndrome=True, topk_syndrome=3)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

model = RecommendationModelWithTransformerAndLabelAttention(
    input_dim=4 * embedding_dim, hidden_dim=64, output_dim=len(node_to_idx['herb'])
)


class BaseRecommendationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(x)
        return self.fc(x)

base_model = BaseRecommendationModel(4*embedding_dim, 64, len(node_to_idx['herb']))
base_total, _, _ = count_parameters(base_model)

model.embedding = transfer_contrastive_weights(
    contrastive_encoder=encoder_contrastive,
    target_linear_layer=model.embedding,
    embedding_dim=embedding_dim
)
scheduler = TemperatureScheduler(initial_temperature=1.5, decay_rate=0.995)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = FocalLoss(alpha=0.25, reduction='mean')

total_epochs = 150

train_loss_curve = []
val_metrics_curve = {k: {'precision': [], 'recall': [], 'f1': []} for k in [5, 10, 15, 20]}

best_val_f1 = 0.0
best_model_path = "../github_op/output/best_model.pth"

for epoch in range(total_epochs):
    gamma = dynamic_gamma2(epoch, total_epochs, gamma_start=2.0, gamma_end=1.0)
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch, gamma=gamma)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_loss_curve.append(avg_loss)

    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, γ={gamma:.3f}")

    scheduler.step()
    model.temperature = scheduler.get_temperature()

    model.eval()
    val_precisions = {k: [] for k in [5, 10, 15, 20]}
    val_recalls = {k: [] for k in [5, 10, 15, 20]}
    val_f1s = {k: [] for k in [5, 10, 15, 20]}
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            logits = model(x_batch)
            probs = torch.sigmoid(logits)
            for probs_row, true_row in zip(probs, y_batch):
                true_indices = set((true_row > 0).nonzero(as_tuple=True)[0].tolist())
                if not true_indices:
                    continue
                for k in [5, 10, 15, 20]:
                    topk_indices = set(torch.topk(probs_row, k).indices.tolist())
                    hits = len(topk_indices & true_indices)
                    precision = hits / k
                    recall = hits / len(true_indices)
                    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    val_precisions[k].append(precision)
                    val_recalls[k].append(recall)
                    val_f1s[k].append(f1)

    current_val_f1 = sum(val_f1s[10]) / len(val_f1s[10]) if val_f1s[10] else 0


    for k in [5, 10, 15, 20]:
        p = sum(val_precisions[k]) / len(val_precisions[k]) if val_precisions[k] else 0
        r = sum(val_recalls[k]) / len(val_recalls[k]) if val_recalls[k] else 0
        f1 = sum(val_f1s[k]) / len(val_f1s[k]) if val_f1s[k] else 0
        val_metrics_curve[k]['precision'].append(p)
        val_metrics_curve[k]['recall'].append(r)
        val_metrics_curve[k]['f1'].append(f1)
        print(f"Top-{k} → Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")


    if current_val_f1 > best_val_f1:
        best_val_f1 = current_val_f1
        torch.save(model.state_dict(), best_model_path)
        print(f" 保存新的最优模型（Top-10 F1: {best_val_f1:.4f}）")

print("\n测试集评估：")

model.load_state_dict(torch.load(best_model_path))
model.eval()

print("\n===== 推理时间统计 =====")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
avg_infer_time, std_infer_time = measure_inference_time(
    model=model,
    data_loader=test_loader,
    repeat=10,
    warmup=3,
    device=device
)

model = model.cpu()

evaluate_precision_recall_f1(model, test_loader, k_list=[5, 10, 15, 20])

torch.save(model.state_dict(), "../github_op/output/model_supervised.pth")

epochs = range(1, total_epochs + 1)

plt.figure(figsize=(8, 4))
plt.plot(epochs, train_loss_curve, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("./output/train_loss_curve.png")
plt.show()


for metric in ['precision', 'recall', 'f1']:
    plt.figure(figsize=(8, 4))
    for k in [5, 10, 15, 20]:
        plt.plot(epochs, val_metrics_curve[k][metric], label=f"Top-{k}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"Validation {metric.capitalize()} Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./output/val_{metric}_curve.png")
    plt.show()

def analyze_batch_predictions(
        model,
        loader,
        herb_vocab,
        id2text=None,
        top_k=15,
        save_path="herb_prediction_report.csv",
        filter_recall_threshold=0
):
    model.eval()
    records = []
    sample_id = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            probs = torch.sigmoid(logits)

            for i in range(len(x_batch)):
                pred_probs = probs[i]
                true_labels = y_batch[i]
                topk_indices = torch.topk(pred_probs, top_k).indices.tolist()
                true_indices = (true_labels > 0).nonzero(as_tuple=True)[0].tolist()

                pred_herbs = [herb_vocab.get(idx, "Unknown Herb") for idx in topk_indices]
                true_herbs = [herb_vocab.get(idx, "Unknown Herb") for idx in true_indices]
                pred_set = set(pred_herbs)
                true_set = set(true_herbs)
                hit_herbs = list(pred_set & true_set)
                missed_herbs = list(true_set - pred_set)
                recall = len(hit_herbs) / len(true_set) if true_set else 0
                precision = len(hit_herbs) / top_k if top_k else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                input_text = id2text[sample_id] if id2text else f"样本 {sample_id}"

                if recall < filter_recall_threshold:
                    records.append({
                        "样本ID": sample_id,
                        "命中草药": "、".join(hit_herbs),
                        "未命中草药": "、".join(missed_herbs),
                    })
                sample_id += 1

    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"分析完成")
    return df


herb_id_to_name = {v: k for k, v in herb_name_to_id.items()}

idx_to_id = {v: k for k, v in node_to_idx['herb'].items()}

herb_vocab = {
    idx: herb_id_to_name.get(hid, "Unknown Herb")
    for idx, hid in idx_to_id.items()
}

model.eval()
df_report = analyze_batch_predictions(
    model=model,
    loader=test_loader,
    herb_vocab=herb_vocab,
    id2text=None,
    top_k=15,
    save_path="../github_op/output/herb_prediction_report.csv",
    filter_recall_threshold=1
)


os.makedirs("../github_op/output/key_metrics", exist_ok=True)

import json
metrics_dict = {
    "inference_time": {
        "avg_ms": float(avg_infer_time),
        "std_ms": float(std_infer_time),
        "meet_clinical_requirement": bool(avg_infer_time < 100)
    }
}

with open("../github_op/output/key_metrics/tradeoff_analysis.json", "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, ensure_ascii=False, indent=4)

print("\n 所有关键指标已保存到：output/key_metrics/tradeoff_analysis.json")
