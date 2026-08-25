import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import math
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index
import numpy as np
import time


def count_model_parameters(model, print_detail=True):

    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        param_num = param.numel()
        total_params += param_num
        if param.requires_grad:
            trainable_params += param_num
    total_m = total_params / 1e6
    trainable_m = trainable_params / 1e6
    if print_detail:
        print("="*80)
        print(f"Model Parameter Summary (M):")
        print(f"Total: {total_m:.4f} | Trainable: {trainable_m:.4f}")
        print("="*80)
    return {"total_m": total_m, "trainable_m": trainable_m}

def measure_inference_time(model, data, device="cuda"):
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(data.x_dict, data.edge_index_dict)
    start = time.time()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
    full_time = time.time() - start
    single_times = {}
    for ntype in ['symptom', 'herb', 'syndrome']:
        if ntype not in out: continue
        test_num = min(100, out[ntype].shape[0])
        start = time.time()
        with torch.no_grad():
            _ = out[ntype][:test_num]
        single_times[ntype] = (time.time() - start)/test_num * 1000
    return {"full_graph_s": full_time, "single_entity_ms": single_times}

def calculate_embedding_quality(out_dict, data, isolated_map=None):
    quality = {}
    for ntype in ['symptom', 'herb', 'syndrome']:
        if ntype not in out_dict: continue
        emb = out_dict[ntype].detach().cpu().numpy()
        emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        sim = np.dot(emb_norm, emb_norm.T)
        sim = sim[~np.eye(len(sim), dtype=bool)]
        quality[f"{ntype}_cos_sim_mean"] = np.mean(sim)
    if isolated_map:
        for ntype, iso_ids in isolated_map.items():
            if not iso_ids or ntype not in out_dict: continue
            iso_emb = out_dict[ntype][iso_ids].detach().cpu().numpy()
            non_iso_ids = [i for i in range(data[ntype].num_nodes) if i not in iso_ids]
            if not non_iso_ids: continue
            non_iso_emb = out_dict[ntype][non_iso_ids].detach().cpu().numpy()
            iso_norm = iso_emb / np.linalg.norm(iso_emb, axis=1, keepdims=True)
            non_iso_norm = non_iso_emb / np.linalg.norm(non_iso_emb, axis=1, keepdims=True)
            quality[f"{ntype}_iso_cross_sim"] = np.mean(np.dot(iso_norm, non_iso_norm.T))
    return quality

def save_metrics(all_metrics, save_path="./top3theta03/hgt_metrics.json"):
    import json
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

TCM_CORE_DIMENSIONS = [
    'cold', 'hot', 'deficiency', 'excess', 'yin', 'yang', 'interior', 'exterior',
    'wind', 'damp', 'phlegm', 'stasis', 'liver', 'spleen', 'lung', 'kidney'
]
CORE_DIM_SIZE = len(TCM_CORE_DIMENSIONS)

NODE_TYPE_SEMANTIC_WEIGHTS = {
    'symptom': [0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02],
    'syndrome': [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01],
    'herb': [0.18, 0.18, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.15, 0.15, 0.1, 0.1],
    'tongue': [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02, 0.0, 0.0],
    'pulse': [0.1, 0.1, 0.2, 0.2, 0.15, 0.15, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02, 0.0, 0.0],
    'cause-disease': [0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.02, 0.02, 0.02, 0.02],
    'disease-location': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25],
    'concoct': [0.25, 0.25, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.02, 0.02, 0.02, 0.02],
    'effect': [0.2, 0.2, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05],
    'flavor': [0.3, 0.3, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05],
    'meridians-tropism': [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2],
    'nature': [0.35, 0.35, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05],
    'taboo': [0.28, 0.28, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.04, 0.04, 0.04, 0.04],
    'toxicity': [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0]
}

def auto_init_tcm_embeddings(node_type, num_nodes, embed_dim=128, random_seed=42):

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    type_weights = np.array(NODE_TYPE_SEMANTIC_WEIGHTS[node_type])
    semantic_features = []
    for i in range(CORE_DIM_SIZE):
        mean = 0.5
        std = 0.1 / (type_weights[i] + 1e-6)
        dim_vals = np.random.normal(mean, std, num_nodes)
        dim_vals = np.clip(dim_vals, 0.0, 1.0)
        semantic_features.append(dim_vals)
    semantic_features = np.array(semantic_features).T

    bias = np.zeros((num_nodes, CORE_DIM_SIZE))
    if node_type == 'herb':
        bias[:, 12:16] = 0.1
    elif node_type == 'syndrome':
        bias[:, 0:8] = 0.1
    elif node_type == 'cause-disease':
        bias[:, 8:12] = 0.1
    elif node_type == 'concoct':
        bias[:, 0:2] = 0.12
    elif node_type == 'effect':
        bias[:, 0:2] = 0.15
    elif node_type == 'flavor':
        bias[:, 0:6] = 0.18
    elif node_type == 'meridians-tropism':
        bias[:, 12:16] = 0.22
    elif node_type == 'nature':
        bias[:, 0:2] = 0.25
    elif node_type == 'taboo':
        bias[:, 0:2] = 0.22
    elif node_type == 'toxicity':
        bias[:, 8:12] = 0.28
    semantic_features = np.clip(semantic_features + bias, 0.0, 1.0)

    projection = nn.Linear(CORE_DIM_SIZE, embed_dim)

    nn.init.xavier_uniform_(projection.weight, gain=nn.init.calculate_gain('tanh'))
    nn.init.constant_(projection.bias, 0.0)

    semantic_tensor = torch.tensor(semantic_features, dtype=torch.float32)
    with torch.no_grad():
        init_embeds = projection(semantic_tensor)

    init_embeds = F.normalize(init_embeds, p=2, dim=1)

    return init_embeds

def sparsemax(input: Tensor, dim: int = -1) -> Tensor:
    number_of_logits = input.size(dim)
    input_sorted, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_sorted.cumsum(dim) - 1
    arange = torch.arange(1, number_of_logits + 1, device=input.device, dtype=input.dtype)
    for i in range(len(input.shape)):
        if i != dim:
            arange = arange.unsqueeze(i)
    k_selected = 1 + arange * input_sorted > input_cumsum
    k_max = (k_selected * arange).max(dim=dim, keepdim=True)[0]
    tau = (input_sorted.gather(dim, (k_max - 1).long()) + (
                1 - input_sorted.cumsum(dim).gather(dim, (k_max - 1).long())) / k_max)
    output = torch.clamp(input - tau, min=0)
    return output

class CrossModalHGTConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Dict[str, int]],
            out_channels: int,
            metadata: Metadata,
            heads: int = 1,
            cross_modal_weight: float = 1.0,
            **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(metadata[1])
        }
        self.cross_modal_weight = cross_modal_weight
        self.modal_groups = {
            'symptom': 0,
            'syndrome': 0,
            'herb': 1,
            'tongue': 2,
            'pulse': 2,
            'cause-disease': 3,
            'disease-location': 3
        }
        self.num_modal_groups = len(set(self.modal_groups.values()))

        self.dst_node_types = set([key[-1] for key in self.edge_types])
        self.kqv_lin = HeteroDictLinear(self.in_channels, self.out_channels * 3)
        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels, types=self.node_types)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)

        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False, is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False, is_sorted=True)

        self.modal_interaction = Parameter(torch.Tensor(self.num_modal_groups, self.num_modal_groups))

        self.type_to_modal = nn.Embedding(len(self.node_types), self.num_modal_groups)

        self.skip = ParameterDict({
            node_type: Parameter(torch.empty(1))
            for node_type in self.node_types
        })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_key = '__'.join(edge_type)
            self.p_rel[edge_key] = Parameter(torch.empty(1, heads))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)
        nn.init.xavier_uniform_(self.modal_interaction)
        self.type_to_modal.reset_parameters()

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
            self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
            edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads
        ks, vs, type_list, offset = [], [], [], {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N
            edge_type_offset = self.edge_types_map[edge_type]
            device = next(iter(self.parameters())).device
            if k_dict: device = next(iter(k_dict.values())).device
            type_vec = torch.arange(H, dtype=torch.long, device=device).view(-1, 1).repeat(1,
                                                                                           N) * num_edge_types + edge_type_offset
            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()
        k = self.k_rel(ks, type_vec).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec).view(H, -1, D).transpose(0, 1)
        return k, v, offset

    def _get_modal_attention(self, src_type: str, dst_type: str) -> Tensor:
        src_modal = self.modal_groups.get(src_type, -1)
        dst_modal = self.modal_groups.get(dst_type, -1)
        if src_modal == -1 or dst_modal == -1:
            return torch.tensor(1.0, device=self.modal_interaction.device)
        base_attn = self.modal_interaction[src_modal, dst_modal]
        src_type_idx = self.node_types.index(src_type)
        dst_type_idx = self.node_types.index(dst_type)
        src_emb = self.type_to_modal(torch.tensor(src_type_idx, device=self.modal_interaction.device))
        dst_emb = self.type_to_modal(torch.tensor(dst_type_idx, device=self.modal_interaction.device))
        type_attn = F.cosine_similarity(src_emb, dst_emb, dim=0)
        return torch.sigmoid((base_attn + type_attn) / 2.0)

    def forward(self, x_dict: Dict[NodeType, Tensor], edge_index_dict: Dict[EdgeType, Adj]) -> Dict[
        NodeType, Optional[Tensor]]:
        H, D = self.heads, self.out_channels // self.heads
        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key], q_dict[key], v_dict[key] = k.view(-1, H, D), q.view(-1, H, D), v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(k_dict, v_dict, edge_index_dict)

        cross_modal_attrs = {}
        for edge_type in edge_index_dict:
            modal_attn = self._get_modal_attention(edge_type[0], edge_type[-1])
            edge_key = '__'.join(edge_type)
            device = self.modal_interaction.device
            p_val = self.p_rel[edge_key] if edge_key in self.p_rel else torch.tensor(1.0, device=device)
            cross_modal_attrs[edge_type] = p_val * (1.0 + self.cross_modal_weight * modal_attn)

        edge_index, edge_attr = construct_bipartite_edge_index(edge_index_dict, src_offset, dst_offset,
                                                               edge_attr_dict=cross_modal_attrs, num_nodes=k.size(0))

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)

        for node_type, start_offset in dst_offset.items():
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:start_offset + q_dict[node_type].size(0)]

        a_dict = self.out_lin({k: F.gelu(v) for k, v in out_dict.items() if v is not None})

        for node_type, out in out_dict.items():
            alpha = self.skip[node_type].sigmoid()
            out_dict[node_type] = alpha * a_dict[node_type] + (1 - alpha) * x_dict[node_type]

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor, index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))

        if ptr is not None:

            alpha = softmax(alpha, index, ptr, size_i)
        else:

            alpha = sparsemax(alpha, dim=0)

        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

ENTITY_DIR = './STP_Subset/entity'
RELATION_DIR = './STP_Subset/relation'
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 128
CROSS_MODAL_WEIGHT = 0.5

data = HeteroData()
id_maps = {}


print("Loading entity data with TCM semantic initialization...")
data = HeteroData()
id_maps = {}

for file in os.listdir(ENTITY_DIR):
    if not file.endswith('.csv'):
        continue
    df = pd.read_csv(os.path.join(ENTITY_DIR, file))
    label = file.split('ID')[0].strip('_').lower()
    num_nodes = len(df)
    ids = df['ID'].values
    idx_map = {id_: i for i, id_ in enumerate(ids)}
    id_maps[label] = idx_map

    init_embeds = auto_init_tcm_embeddings(
        node_type=label,
        num_nodes=num_nodes,
        embed_dim=HIDDEN_CHANNELS,
        random_seed=42
    )
    data[label].x = init_embeds
    data[label].node_id = torch.arange(num_nodes)

    if 'name' in df.columns:
        data[label].node_name = df['name'].values

print("Entities loaded:", list(data.node_types))

def find_isolated_nodes(data, node_type):

    all_ids = set(range(data[node_type].num_nodes))
    connected_ids = set()

    for etype in data.edge_types:
        if etype[0] == node_type:
            connected_ids.update(data[etype].edge_index[0].tolist())
        if etype[2] == node_type:
            connected_ids.update(data[etype].edge_index[1].tolist())

    isolated = list(all_ids - connected_ids)
    print(f"[Analysis] Node Type: {node_type} | Total: {len(all_ids)} | Isolated: {len(isolated)}")
    return isolated


# ---------- STEP 2: LOAD RELATION CSVs & TRAINING & KSTR ----------

print("Loading relation data...")
for file in os.listdir(RELATION_DIR):
    if not file.endswith('.csv'): continue
    df = pd.read_csv(os.path.join(RELATION_DIR, file))

    parts = file.replace('.csv', '').split('_')
    head_type = parts[0].lower()
    tail_type = parts[1].lower()
    rel_type = '_'.join(parts[2:]).lower()

    try:
        head_map = id_maps[head_type]
        tail_map = id_maps[tail_type]
        src = [head_map[i] for i in df[':START_ID'] if i in head_map]
        dst = [tail_map[i] for i in df[':END_ID'] if i in tail_map]
    except KeyError:
        print(f"[WARN] Skipping relation {file} due to missing node type.")
        continue

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data[(head_type, rel_type, tail_type)].edge_index = edge_index

    rev_rel_type = f'rev_{rel_type}'
    data[(tail_type, rev_rel_type, head_type)].edge_index = edge_index[[1, 0]]

print("Relations loaded:", list(data.edge_types))

def train_with_early_stopping(model, data, optimizer, max_epochs=100, patience=5):
    best_loss = float('inf')
    best_state = None
    counter = 0
    model.train()

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)

        loss_all = 0
        for edge_type, edge_index in data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src_idx, dst_idx = edge_index[0], edge_index[1]
            if src_idx.numel() == 0: continue

            neg_dst_idx = torch.randint(0, out[dst_type].size(0), (dst_idx.size(0),), device=dst_idx.device)

            pos_score = (out[src_type][src_idx] * out[dst_type][dst_idx]).sum(dim=1)
            neg_score = (out[src_type][src_idx] * out[dst_type][neg_dst_idx]).sum(dim=1)

            score = torch.cat([pos_score, neg_score])
            label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])

            loss = F.binary_cross_entropy_with_logits(score, label)
            loss_all += loss

        loss_all.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Loss: {loss_all.item():.4f}")

        if loss_all.item() < best_loss - 1e-4:
            best_loss = loss_all.item()
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"[Early Stop] Best Loss: {best_loss:.4f}")
                break

    model.load_state_dict(best_state)
    return model

def add_multi_target_pseudo_edges(data, out, src_type, dst_type_list, id_maps, topk=3, score_threshold=0.2):

    TCM_LOGIC_CONSTRAINTS = {
        'symptom': ['herb', 'syndrome'],
        'syndrome': ['symptom', 'cause-disease', 'disease-location', 'tongue', 'pulse'],
        'tongue': ['syndrome'],
        'pulse': ['syndrome'],
        'herb': ['symptom', 'syndrome'],
        'cause-disease': ['syndrome'],
        'disease-location': ['syndrome']
    }

    if src_type in TCM_LOGIC_CONSTRAINTS:
        valid_dst_list = [dst for dst in dst_type_list if dst in TCM_LOGIC_CONSTRAINTS[src_type]]
        if not valid_dst_list:
            print(f"[WARN] No valid dst type for {src_type} (constrained by TCM logic)")
            return []
        dst_type_list = valid_dst_list


    device = out[src_type].device
    isolated_ids = find_isolated_nodes(data, src_type)
    all_new_edges = []

    src_emb_norm = F.normalize(out[src_type], p=2, dim=1)

    src_valid_mask = torch.isfinite(src_emb_norm).all(dim=1) & (src_emb_norm.norm(dim=1) > 1e-6)
    valid_isolated_ids = [sid for sid in isolated_ids if src_valid_mask[sid]]
    print(f"[INFO] {src_type}: {len(isolated_ids)} isolated nodes, {len(valid_isolated_ids)} valid (after mask)")


    for dst_type in dst_type_list:
        edge_name = f'pseudo_{src_type}_{dst_type}'
        new_edges = []
        dst_emb = out[dst_type]
        dst_emb_norm = F.normalize(dst_emb, p=2, dim=1)

        dst_valid_mask = torch.isfinite(dst_emb_norm).all(dim=1) & (dst_emb_norm.norm(dim=1) > 1e-6)
        valid_dst_indices = torch.where(dst_valid_mask)[0]
        if len(valid_dst_indices) == 0:
            print(f"[WARN] No valid dst nodes for {src_type} → {dst_type}")
            continue

        valid_dst_emb = dst_emb_norm[valid_dst_indices]

        for sid in valid_isolated_ids:
            src_emb = src_emb_norm[sid]
            scores = torch.matmul(valid_dst_emb, src_emb.unsqueeze(1)).squeeze()
            topk_scores, topk_indices = torch.topk(scores, min(topk, len(scores)))

            for score, idx in zip(topk_scores.tolist(), topk_indices.tolist()):
                if score >= score_threshold:
                    did = valid_dst_indices[idx].item()
                    new_edges.append((sid, did))

        if not new_edges:
            print(f"[WARN] No pseudo edges for {src_type} → {dst_type} (threshold={score_threshold})")
            continue

        edge_tensor = torch.tensor(new_edges, dtype=torch.long).T.to(device)
        data[(src_type, edge_name, dst_type)].edge_index = edge_tensor
        data[(dst_type, f'rev_{edge_name}', src_type)].edge_index = torch.stack([edge_tensor[1], edge_tensor[0]])

        print(f"[INFO] Added {len(new_edges)} pseudo edges for {src_type} → {dst_type} (threshold={score_threshold})")
        all_new_edges.extend(new_edges)

    return isolated_ids
# ---------- STEP 3: MODEL DEFINITION -----------

class HeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_dim=128, out_dim=128, num_layers=2,
                 dropout=0.1, use_output_proj=True, cross_modal_weight=0.5):
        super().__init__()
        self.use_output_proj = use_output_proj
        self.cross_modal_weight = cross_modal_weight

        self.proj = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            ) for node_type in metadata[0]
        })

        self.hgt_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            self.hgt_layers.append(
                CrossModalHGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim if i < num_layers - 1 else out_dim,
                    metadata=metadata,
                    heads=4,
                    cross_modal_weight=cross_modal_weight
                )
            )
            self.norms.append(nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim if i < num_layers - 1 else out_dim)
                for node_type in metadata[0]
            }))

        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

        if self.use_output_proj:
            self.output_proj = nn.ModuleDict({
                node_type: nn.Sequential(
                    nn.Linear(out_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.Tanh()
                ) for node_type in metadata[0]
            })

    def forward(self, x_dict, edge_index_dict):

        x_dict = {k: self.proj[k](v) for k, v in x_dict.items()}

        for layer, norm in zip(self.hgt_layers, self.norms):

            x_dict_next = layer(x_dict, edge_index_dict)

            new_x_dict = {}
            for k in x_dict_next.keys():
                h = self.activation(norm[k](x_dict_next[k]))
                h = self.dropout(h)

                if h.size(-1) == x_dict[k].size(-1):
                    new_x_dict[k] = h + x_dict[k]
                else:
                    new_x_dict[k] = h
            x_dict = new_x_dict

        if self.use_output_proj:
            x_dict = {k: self.output_proj[k](v) for k, v in x_dict.items()}

        return x_dict


# ---------- STEP 4: MODEL TRAINING & FINE-TUNING ----------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

print("Stage 1: Training base embeddings with cross-modal attention...")
model = HeteroGNN(
    data.metadata(),
    hidden_dim=HIDDEN_CHANNELS,
    out_dim=OUT_CHANNELS,
    cross_modal_weight=CROSS_MODAL_WEIGHT
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
model = train_with_early_stopping(model, data, optimizer, max_epochs=100, patience=5)

def freeze_model_parameters(model, mode='fine-tune', freeze=None):

    if freeze is not None:
        mode = 'all' if freeze else 'fine-tune'

    for name, param in model.named_parameters():
        if mode == 'all':
            param.requires_grad = False
        else:
            if 'hgt_layers' in name or 'modal_interaction' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    print(f"[INFO] Model layers frozen for {mode} (freeze={freeze}).")

def retrain_with_multiple_pseudo_edges(data, model, isolated_map, max_epochs=50, lr=0.001, patience=5):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    best_loss = float('inf')
    best_state = model.state_dict()
    counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = 0

        for etype in data.edge_index_dict:
            if 'pseudo' not in etype[1]:
                continue
            src_type, _, dst_type = etype
            if src_type not in isolated_map:
                continue

            edge_index = data[etype].edge_index
            src, dst = edge_index[0], edge_index[1]

            mask = torch.tensor([i.item() in isolated_map[src_type] for i in src], device=device)
            if mask.sum() == 0:
                continue
            src = src[mask]
            dst = dst[mask]
            neg_dst = torch.randint(0, out[dst_type].size(0), (dst.size(0),), device=device)

            pos_score = (out[src_type][src] * out[dst_type][dst]).sum(dim=1)
            neg_score = (out[src_type][src] * out[dst_type][neg_dst]).sum(dim=1)
            score = torch.cat([pos_score, neg_score])
            label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
            l = F.binary_cross_entropy_with_logits(score, label)
            loss += l

        if loss == 0:
            print("[WARN] No pseudo edges to train, skip retrain.")
            break

        loss.backward()
        optimizer.step()
        print(f"[Joint Pseudo Epoch {epoch}] Loss: {loss.item():.4f}")

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"[Joint Early Stop] Best Loss: {best_loss:.4f} after {epoch} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("[INFO] Keep original model weights (no better loss found).")
    return model


# ---------- STEP 6: PSEUDO-EDGE REFINEMENT FOR ISOLATED NODES ----------
print("\n Adding multi-target pseudo edges for isolated symptom nodes")

node_types, edge_types = data.metadata()

pseudo_edge_types = [
    ('symptom', 'pseudo_symptom_herb', 'herb'),
    ('herb', 'rev_pseudo_symptom_herb', 'symptom'),

    ('syndrome', 'pseudo_syndrome_symptom', 'symptom'),
    ('symptom', 'rev_pseudo_syndrome_symptom', 'syndrome'),

    ('syndrome', 'pseudo_syndrome_tongue', 'tongue'),
    ('tongue', 'rev_pseudo_syndrome_tongue', 'syndrome'),

    ('syndrome', 'pseudo_syndrome_pulse', 'pulse'),
    ('pulse', 'rev_pseudo_syndrome_pulse', 'syndrome'),

    ('pulse', 'pseudo_pulse_syndrome', 'syndrome'),
    ('syndrome', 'rev_pseudo_pulse_syndrome', 'pulse'),

    ('tongue', 'pseudo_tongue_syndrome', 'syndrome'),
    ('syndrome', 'rev_pseudo_tongue_syndrome', 'tongue'),

    ('syndrome', 'pseudo_syndrome_cause-disease', 'cause-disease'),
    ('cause-disease', 'rev_pseudo_syndrome_cause-disease', 'syndrome'),

    ('syndrome', 'pseudo_syndrome_disease-location', 'disease-location'),
    ('disease-location', 'rev_pseudo_syndrome_disease-location', 'syndrome'),

    ('cause-disease', 'pseudo_cause-disease_syndrome', 'syndrome'),
    ('syndrome', 'rev_pseudo_cause-disease_syndrome', 'cause-disease'),

    ('disease-location', 'pseudo_disease-location_syndrome', 'syndrome'),
    ('syndrome', 'rev_pseudo_disease-location_syndrome', 'disease-location'),
]
for et in pseudo_edge_types:
    if et not in edge_types:
        edge_types.append(et)

metadata = (node_types, edge_types)

model.metadata = metadata
model.hgt_layers = nn.ModuleList()
model.norms = nn.ModuleList()
num_layers = 2
for i in range(num_layers):
    model.hgt_layers.append(
        CrossModalHGTConv(
            in_channels=HIDDEN_CHANNELS,
            out_channels=HIDDEN_CHANNELS if i < num_layers - 1 else OUT_CHANNELS,
            metadata=metadata,
            heads=2,
            cross_modal_weight=CROSS_MODAL_WEIGHT
        )
    )
    model.norms.append(nn.ModuleDict({
        node_type: nn.LayerNorm(HIDDEN_CHANNELS if i < num_layers - 1 else OUT_CHANNELS)
        for node_type in metadata[0]
    }))
model = model.to(device)

model.eval()
with torch.no_grad():
    out = model(data.x_dict, data.edge_index_dict)

all_isolated_map = {}
pseudo_tasks = [
    ('symptom', ['herb']),
    ('syndrome', ['symptom', 'cause-disease', 'disease-location', 'tongue', 'pulse']),
    ('pulse', ['syndrome']),
    ('tongue', ['syndrome']),
    ('cause-disease', ['syndrome']),
    ('disease-location', ['syndrome']),
]
for src_type, dst_list in pseudo_tasks:

    iso_ids = add_multi_target_pseudo_edges(
        data, out,
        src_type=src_type,
        dst_type_list=dst_list,
        id_maps=id_maps,
        topk=3,
        score_threshold=0.2
    )
    all_isolated_map[src_type] = iso_ids

freeze_model_parameters(model, mode='fine-tune')

model = retrain_with_multiple_pseudo_edges(
    data, model,
    isolated_map=all_isolated_map,
    max_epochs=50,
    lr=0.001,
    patience=5
)

print("\n[Final] Exporting refined embeddings and diagnostic metadata...")

def save_refined_embeddings(model, data, id_maps, filename):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)

    records = []
    for ntype in data.node_types:
        node_ids = data[ntype].node_id.tolist()
        embs = out[ntype].detach().cpu().numpy()
        reverse_map = {v: k for k, v in id_maps[ntype].items()}

        names = getattr(data[ntype], 'node_name', [None] * len(node_ids))

        for i in range(len(node_ids)):
            idx = node_ids[i]
            original_id = reverse_map.get(idx, f"unknown_{idx}")
            name = names[i] if names[i] is not None else "N/A"
            emb_str = ' '.join([f'{v:.6f}' for v in embs[idx]])
            records.append({
                'node_type': ntype,
                'node_id': original_id,
                'node_name': name,
                'embedding': emb_str
            })

    df_out = pd.DataFrame(records)
    df_out.to_csv(filename, index=False)
    print(f"Success: Refined embeddings saved to {filename}")

save_refined_embeddings(model, data, id_maps,
                        './param_embeddings/node_embeddings_final_kstr_128.csv')

with open('./param_embeddings/kstr_isolated_metadata.pkl', 'wb') as f:
    pickle.dump({
        'isolated_map': all_isolated_map,
        'config': {
            'theta': 0.2,
            'cross_modal_weight': CROSS_MODAL_WEIGHT
        }
    }, f)
print("KSTR workflow completed successfully.")