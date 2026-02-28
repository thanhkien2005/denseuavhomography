"""
scripts/sanity_check_shapes.py
──────────────────────────────
Verify expected tensor shapes end-to-end WITHOUT importing any model code.
Run from repo root:
    python scripts/sanity_check_shapes.py

All shapes are derived from:
    img_size   = 512
    patch_size = 16
    embed_dim  = 384  (ViT-S)
    B          = 2    (dummy batch)
"""

import torch

# ──────────────────────────────────────────────
#  Config (mirrors configs/denseuav_v1.yaml)
# ──────────────────────────────────────────────
B          = 2       # batch size
C_img      = 3       # RGB
IMG        = 512     # spatial size
PATCH      = 16      # ViT patch size
GRID       = IMG // PATCH   # 32
N_PATCHES  = GRID * GRID    # 1024
C_EMB      = 384            # ViT-S embed dim
N_CLASSES  = 2256           # training classes

print("=" * 56)
print("  DenseUAV-Homo  —  expected tensor shapes")
print("=" * 56)

# ── 1. Raw inputs ─────────────────────────────
uav_img = torch.zeros(B, C_img, IMG, IMG)
sat_img = torch.zeros(B, C_img, IMG, IMG)
print(f"[input]  uav_img : {tuple(uav_img.shape)}")   # (B,3,512,512)
print(f"[input]  sat_img : {tuple(sat_img.shape)}")   # (B,3,512,512)
assert uav_img.shape == (B, 3, IMG, IMG)
assert sat_img.shape == (B, 3, IMG, IMG)

# ── 2. ViT token output ───────────────────────
#    timm ViT returns (B, N_PATCHES+1, C_EMB)  when global_pool=''
tokens_uav = torch.zeros(B, N_PATCHES + 1, C_EMB)
tokens_sat = torch.zeros(B, N_PATCHES + 1, C_EMB)
print(f"[tokens] uav tokens : {tuple(tokens_uav.shape)}")  # (B,1025,384)
print(f"[tokens] sat tokens : {tuple(tokens_sat.shape)}")  # (B,1025,384)
assert tokens_uav.shape == (B, N_PATCHES + 1, C_EMB)

# ── 3. Patch tokens → feature map ─────────────
#    drop cls token → (B, N_PATCHES, C_EMB) → reshape → (B,C_EMB,GRID,GRID)
patch_tokens_uav = tokens_uav[:, 1:, :]                  # (B,1024,384)
feat_map_uav     = patch_tokens_uav.transpose(1, 2)       # (B,384,1024)
feat_map_uav     = feat_map_uav.reshape(B, C_EMB, GRID, GRID)
feat_map_sat     = torch.zeros(B, C_EMB, GRID, GRID)
print(f"[fmap]   uav feat map : {tuple(feat_map_uav.shape)}")  # (B,384,32,32)
print(f"[fmap]   sat feat map : {tuple(feat_map_sat.shape)}")  # (B,384,32,32)
assert feat_map_uav.shape == (B, C_EMB, GRID, GRID)

# ── 4. HomographyNet output ───────────────────
H_u2s = torch.zeros(B, 3, 3)
print(f"[homo]   H_u2s        : {tuple(H_u2s.shape)}")   # (B,3,3)
assert H_u2s.shape == (B, 3, 3)

# ── 5. Warped + gated feature map ────────────
feat_map_uav_warp  = torch.zeros(B, C_EMB, GRID, GRID)   # after warp_perspective
gate               = torch.zeros(B, 1, 1, 1)              # scalar gate per sample
feat_map_uav_blend = gate * feat_map_uav_warp + (1 - gate) * feat_map_uav
print(f"[warp]   uav warp     : {tuple(feat_map_uav_warp.shape)}")  # (B,384,32,32)
print(f"[gate]   gate         : {tuple(gate.shape)}")               # (B,1,1,1)
print(f"[blend]  uav blend    : {tuple(feat_map_uav_blend.shape)}") # (B,384,32,32)
assert feat_map_uav_blend.shape == (B, C_EMB, GRID, GRID)

# ── 6. GeM pooling → embedding ────────────────
#    GeM: (B,C,H,W) → (B,C,1,1) → flatten → (B,C) → L2-norm → (B,C)
emb_uav = torch.zeros(B, C_EMB)
emb_sat = torch.zeros(B, C_EMB)
print(f"[emb]    uav emb      : {tuple(emb_uav.shape)}")  # (B,384)
print(f"[emb]    sat emb      : {tuple(emb_sat.shape)}")  # (B,384)
assert emb_uav.shape == (B, C_EMB)

# ── 7. Classifier logits ──────────────────────
logits_uav = torch.zeros(B, N_CLASSES)
logits_sat = torch.zeros(B, N_CLASSES)
print(f"[logits] uav logits   : {tuple(logits_uav.shape)}")  # (B,2256)
print(f"[logits] sat logits   : {tuple(logits_sat.shape)}")  # (B,2256)
assert logits_uav.shape == (B, N_CLASSES)

# ── 8. Labels ─────────────────────────────────
labels = torch.zeros(B, dtype=torch.long)
print(f"[label]  labels       : {tuple(labels.shape)}")    # (B,)
assert labels.shape == (B,)

print("=" * 56)
print("  All pipeline shape assertions PASSED.")
print("=" * 56)

# ══════════════════════════════════════════════════════
# SECTION 2 — Metric functions (random dummy embeddings)
# ══════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.recall import recall_at_k
from metrics.sdm    import haversine_distance, sdm_at_k, sdm_at_k_multi

print()
print("=" * 56)
print("  Metric function shape / correctness checks")
print("=" * 56)

Q = 16    # number of queries
G = 64    # number of gallery items
D = C_EMB  # embedding dim = 384
K_LIST = [1, 5, 10]
S = 5e3   # distance scale metres

torch.manual_seed(0)

# ── Recall@K inputs ───────────────────────────────────
# L2-normalise to satisfy cosine-similarity assumption
q_emb  = torch.randn(Q, D)
q_emb  = q_emb  / q_emb.norm(dim=1, keepdim=True)    # (Q, D)
g_emb  = torch.randn(G, D)
g_emb  = g_emb  / g_emb.norm(dim=1, keepdim=True)    # (G, D)

# Assign unique labels 0..Q-1 to queries; repeat them in gallery
# so that gallery[i] is the *correct* match for query[i % Q]
q_labels = torch.arange(Q)                            # (Q,)  labels 0-15
g_labels = torch.arange(G) % Q                        # (G,)  labels repeat 0-15
print(f"[recall] q_emb   : {tuple(q_emb.shape)}")    # (16, 384)
print(f"[recall] g_emb   : {tuple(g_emb.shape)}")    # (64, 384)
print(f"[recall] q_labels: {tuple(q_labels.shape)}") # (16,)
print(f"[recall] g_labels: {tuple(g_labels.shape)}") # (64,)

recall = recall_at_k(q_emb, g_emb, q_labels, g_labels, k_list=K_LIST)
for k, v in recall.items():
    print(f"  Recall@{k:<3} = {v:.4f}  (random baseline; should be ~K/G = {K_LIST[K_LIST.index(k)]/G:.3f})")
assert set(recall.keys()) == set(K_LIST), "recall_at_k must return all requested K"
assert all(0.0 <= v <= 1.0 for v in recall.values()), "recall values must be in [0,1]"

# Perfect retrieval check: make query[0] identical to gallery[0]
q_perfect   = g_emb[:Q].clone()    # queries are copies of first Q gallery items
g_labels_p  = torch.arange(G) % Q
recall_perf = recall_at_k(q_perfect, g_emb, q_labels, g_labels_p, k_list=[1])
print(f"  Recall@1 (perfect emb, 4 matches per query) = {recall_perf[1]:.4f}  (expected 1.0)")
assert recall_perf[1] == 1.0, "Perfect embeddings must give Recall@1 = 1.0"

print("[recall] All assertions PASSED.")

# ── Haversine distance ─────────────────────────────────
# Check: same point → distance 0
lon = torch.tensor([[120.387]])
lat = torch.tensor([[30.324]])
d_zero = haversine_distance(lon, lat, lon, lat)
print(f"[haversine] same-point distance : {d_zero.item():.6f} m  (expected 0.0)")
assert d_zero.item() < 1e-3, f"Same-point Haversine must be ~0, got {d_zero.item()}"

# Check: London → Paris ~341 km
lon1 = torch.tensor([[-0.1276]])
lat1 = torch.tensor([[51.5074]])
lon2 = torch.tensor([[2.3522]])
lat2 = torch.tensor([[48.8566]])
d_lp = haversine_distance(lon1, lat1, lon2, lat2)
print(f"[haversine] London->Paris : {d_lp.item()/1000:.1f} km  (expected ~341 km)")
assert 330_000 < d_lp.item() < 360_000, f"London-Paris distance unexpected: {d_lp.item()}"
print("[haversine] All assertions PASSED.")

# ── SDM@K inputs ──────────────────────────────────────
# GPS: small area around the DenseUAV campus (lon≈120.38, lat≈30.32)
q_gps  = torch.zeros(Q, 2)
q_gps[:, 0] = 120.38 + torch.rand(Q) * 0.01   # lon  (Q, 2)[:,0]
q_gps[:, 1] =  30.32 + torch.rand(Q) * 0.01   # lat  (Q, 2)[:,1]
g_gps  = torch.zeros(G, 2)
g_gps[:, 0] = 120.38 + torch.rand(G) * 0.01   # (G, 2)[:,0]
g_gps[:, 1] =  30.32 + torch.rand(G) * 0.01   # (G, 2)[:,1]
print(f"[sdm]  q_gps : {tuple(q_gps.shape)}")  # (16, 2)
print(f"[sdm]  g_gps : {tuple(g_gps.shape)}")  # (64, 2)

K_MAX = max(K_LIST)
sim_mat      = q_emb @ g_emb.T                          # (Q, G)
topk_indices = sim_mat.topk(K_MAX, dim=1).indices        # (Q, K_MAX)
print(f"[sdm]  topk_indices : {tuple(topk_indices.shape)}")  # (16, 10)
assert topk_indices.shape == (Q, K_MAX)

for k in K_LIST:
    score = sdm_at_k(topk_indices, q_gps, g_gps, K=k, s=S)
    print(f"  SDM@{k:<3} = {score:.4f}  (random; expected > 0 if GPS are close)")
    assert 0.0 <= score <= 1.0, f"SDM@{k} out of range: {score}"

# Perfect retrieval + co-located GPS → SDM@1 = 1.0
#   query i's correct gallery item is at index i (same GPS)
topk_perfect = torch.arange(Q).unsqueeze(1)             # (Q, 1)  rank-1 = exact match
g_gps_perf   = q_gps[:Q].clone()                        # gallery[i] at same GPS as query[i]
# pad to K_MAX columns
topk_perfect = topk_perfect.expand(Q, K_MAX)             # (Q, K_MAX)
sdm_perf     = sdm_at_k(topk_perfect, q_gps, g_gps_perf, K=1, s=S)
print(f"  SDM@1 (perfect+co-located) = {sdm_perf:.4f}  (expected 1.0)")
assert abs(sdm_perf - 1.0) < 1e-5, f"Perfect SDM@1 must be 1.0, got {sdm_perf}"

# Multi-K convenience wrapper
sdm_multi = sdm_at_k_multi(q_emb, g_emb, q_gps, g_gps, k_list=K_LIST, s=S)
assert set(sdm_multi.keys()) == set(K_LIST), "sdm_at_k_multi must return all K"
print(f"  sdm_at_k_multi keys: {sorted(sdm_multi.keys())}  OK")
print("[sdm] All assertions PASSED.")

print()
print("=" * 56)
print("  All metric assertions PASSED.")
print("=" * 56)

# ══════════════════════════════════════════════════════
# SECTION 3 — SiameseViT model forward shapes
# ══════════════════════════════════════════════════════
from models.vit_siamese import SiameseViT

print()
print("=" * 56)
print("  SiameseViT model shape check (pretrained=False)")
print("=" * 56)

N_CLS_DUMMY = 10   # small number of classes for speed

model = SiameseViT(
    num_classes   = N_CLS_DUMMY,
    embed_dim     = C_EMB,    # 384
    img_size      = IMG,      # 512
    patch_size    = PATCH,    # 16
    gem_p         = 3.0,
    gem_learnable = True,
    pretrained    = False,    # no download in sanity check
)
model.eval()
print(f"  {model}")
print()

with torch.no_grad():
    uav_dummy = torch.randn(B, 3, IMG, IMG)   # (2, 3, 512, 512)
    sat_dummy = torch.randn(B, 3, IMG, IMG)   # (2, 3, 512, 512)

    # ── forward_features ───────────────────────────────────
    tokens_uav = model.forward_features(uav_dummy)
    print(f"[model] forward_features(uav) : {tuple(tokens_uav.shape)}")
    # expected: (B, 1025, 384)
    assert tokens_uav.shape == (B, N_PATCHES + 1, C_EMB), (
        f"Expected ({B},{N_PATCHES+1},{C_EMB}), got {tuple(tokens_uav.shape)}"
    )

    # ── reshape_patch_tokens ────────────────────────────────
    fmap_uav = model.reshape_patch_tokens(tokens_uav)
    print(f"[model] reshape_patch_tokens  : {tuple(fmap_uav.shape)}")
    # expected: (B, 384, 32, 32)
    assert fmap_uav.shape == (B, C_EMB, GRID, GRID), (
        f"Expected ({B},{C_EMB},{GRID},{GRID}), got {tuple(fmap_uav.shape)}"
    )

    # ── full forward ────────────────────────────────────────
    out = model(uav_dummy, sat_dummy)

    emb_uav    = out["emb_uav"]     # (B, 384)
    emb_sat    = out["emb_sat"]     # (B, 384)
    logit_uav  = out["logit_uav"]  # (B, N_CLS_DUMMY)
    logit_sat  = out["logit_sat"]  # (B, N_CLS_DUMMY)
    gate_logit = out["gate_logit"] # (B, 1)
    delta      = out["delta"]      # (B, 8)

    rows = [
        ("emb_uav",    emb_uav,    (B, C_EMB),       torch.float32),
        ("emb_sat",    emb_sat,    (B, C_EMB),       torch.float32),
        ("logit_uav",  logit_uav,  (B, N_CLS_DUMMY), torch.float32),
        ("logit_sat",  logit_sat,  (B, N_CLS_DUMMY), torch.float32),
        ("gate_logit", gate_logit, (B, 1),            torch.float32),
        ("delta",      delta,      (B, 8),            torch.float32),
    ]
    for name, t, exp_shape, exp_dtype in rows:
        ok = (tuple(t.shape) == exp_shape) and (t.dtype == exp_dtype)
        print(f"  {name:<12} {str(tuple(t.shape)):<20} {str(exp_shape):<20}"
              f" {'OK' if ok else 'FAIL'}")
        assert tuple(t.shape) == exp_shape, f"{name} shape mismatch"
        assert t.dtype == exp_dtype,        f"{name} dtype mismatch"

    # ── Gate stats ──────────────────────────────────────────
    # gate = sigmoid(gate_logit); head_gate is zero-weight + bias=-2 at init
    # → gate_logit ≈ -2.0  →  g ≈ sigmoid(-2) ≈ 0.119
    gate = torch.sigmoid(gate_logit)   # (B, 1)
    g_mean = gate.mean().item()
    g_min  = gate.min().item()
    g_max  = gate.max().item()
    print(f"  gate (g)       : mean={g_mean:.4f}  min={g_min:.4f}  max={g_max:.4f}")
    print(f"  gate_logit     : {gate_logit.squeeze().tolist()}")
    # At init, head_gate.weight=0, bias=-2  → gate_logit=-2 → g≈0.119
    assert all(0.0 < v < 1.0 for v in gate.flatten().tolist()), \
        "gate values must be strictly in (0, 1)"
    print("  gate range     : OK (all g in (0,1))")

    # ── L2 normalisation check ──────────────────────────────
    norms_uav = emb_uav.norm(p=2, dim=1)   # (B,) — must be all 1.0
    norms_sat = emb_sat.norm(p=2, dim=1)   # (B,) — must be all 1.0
    ones      = torch.ones(B)
    print(f"  emb_uav norms  : {norms_uav.tolist()}")   # [1.0, 1.0]
    print(f"  emb_sat norms  : {norms_sat.tolist()}")   # [1.0, 1.0]
    assert torch.allclose(norms_uav, ones, atol=1e-5), (
        f"UAV embeddings not L2-normalised: {norms_uav.tolist()}"
    )
    assert torch.allclose(norms_sat, ones, atol=1e-5), (
        f"SAT embeddings not L2-normalised: {norms_sat.tolist()}"
    )
    print("  L2-norm check  : OK (all norms = 1.0)")

    # ── pos_embed size check ────────────────────────────────
    pos_len = model.backbone.pos_embed.shape[1]   # must be 1025
    print(f"  pos_embed len  : {pos_len}  (expected {N_PATCHES + 1})")
    assert pos_len == N_PATCHES + 1, (
        f"pos_embed has {pos_len} tokens; expected {N_PATCHES+1}"
    )

print()
print("=" * 56)
print("  All model assertions PASSED.")
print("=" * 56)
