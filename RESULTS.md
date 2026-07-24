# PPO+GNN Training Results

```text
✓  GPU detected: Tesla T4 (15.6 GB)
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
✓  model.get_action patched → auto-cast to cuda
✓  model.forward_batch patched → auto-cast to cuda

────────────────────────────────────────────────────────────
  Training GNNEncoder · use-case: DEFENCE · season: MONSOON
  Device       : cuda
  Model params : 136,866
  Node feat dim: 9  |  Edge feat dim: 5
  GPU memory   : 10.1 MB allocated
────────────────────────────────────────────────────────────
Training PPO+GNN [BATCHED-GNN, PERSISTENT-OPT]  |  200 epochs × 144 steps
Use-case: defence  |  Clip ε=0.3  |  LR=0.0008
Mini-batch: 32  |  Update iters: 8
Entropy: 0.15 → 0.03 over 80 epochs
GAE λ=0.95
Device: cuda
————————————————————————————————————————————————————————————
Epoch  10/200 | R= +0.959 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.1365 | LR=0.00080 | StepTime=0.0546s
Epoch  20/200 | R= +0.918 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.1215 | LR=0.00079 | StepTime=0.0581s
Epoch  30/200 | R= +0.892 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.1065 | LR=0.00078 | StepTime=0.0569s
Epoch  40/200 | R= +0.952 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0915 | LR=0.00075 | StepTime=0.0542s
Epoch  50/200 | R= +0.896 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0765 | LR=0.00072 | StepTime=0.0588s

——————————————————————————————————————————————————————
  Mid-train eval @ epoch 50/200 (3 runs × 144 steps, no early termination)
——————————————————————————————————————————————————————
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
  Random         : -13.995
  Shortest-Len   : +52.204
  Dijkstra-Hop   : +47.696
  Max-SKR        : +146.552
  PPO+GNN        : +151.590 ◄ PPO
——————————————————————————————————————————————————————

Epoch  60/200 | R= +0.957 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0615 | LR=0.00067 | StepTime=0.0537s
Epoch  70/200 | R= +0.938 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0465 | LR=0.00062 | StepTime=0.0521s
Epoch  80/200 | R= +0.902 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0315 | LR=0.00056 | StepTime=0.0522s
Epoch  90/200 | R= +0.926 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0300 | LR=0.00050 | StepTime=0.0503s
Epoch 100/200 | R= +0.946 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0300 | LR=0.00044 | StepTime=0.0549s

——————————————————————————————————————————————————————
  Mid-train eval @ epoch 100/200 (3 runs × 144 steps, no early termination)
——————————————————————————————————————————————————————
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
  Random         : -5.725
  Shortest-Len   : +52.204
  Dijkstra-Hop   : +47.696
  Max-SKR        : +146.552
  PPO+GNN        : +137.374 ◄ PPO
——————————————————————————————————————————————————————

Epoch 110/200 | R= +0.946 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0300 | LR=0.00037 | StepTime=0.0530s
Epoch 120/200 | R= +0.913 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0300 | LR=0.00031 | StepTime=0.0538s
Epoch 130/200 | R= +0.959 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0300 | LR=0.00024 | StepTime=0.0563s
Epoch 140/200 | R= +0.936 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0300 | LR=0.00019 | StepTime=0.0551s
Epoch 150/200 | R= +0.920 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0300 | LR=0.00014 | StepTime=0.0521s

——————————————————————————————————————————————————————
  Mid-train eval @ epoch 150/200 (3 runs × 144 steps, no early termination)
——————————————————————————————————————————————————————
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
  Random         : +13.995
  Shortest-Len   : +52.204
  Dijkstra-Hop   : +47.696
  Max-SKR        : +136.552
  PPO+GNN        : +165.117 ◄ PPO
——————————————————————————————————————————————————————

Epoch 160/200 | R= +1.010 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0300 | LR=0.00009 | StepTime=0.0487s
Epoch 170/200 | R= +0.885 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0300 | LR=0.00006 | StepTime=0.0511s
Epoch 180/200 | R= +0.938 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0300 | LR=0.00003 | StepTime=0.0530s
Epoch 190/200 | R= +0.914 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0300 | LR=0.00001 | StepTime=0.0523s
Epoch 200/200 | R= +0.896 | w(q)=0.0000 | SKR=1.200 | Pool=0.160 | ent=0.0300 | LR=0.00001 | StepTime=0.0495s

——————————————————————————————————————————————————————
  Mid-train eval @ epoch 200/200 (3 runs × 144 steps, no early termination)
——————————————————————————————————————————————————————
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
DEBUG: QKDRoutingEnv initialized with source='Delhi', dest='Chennai')
DEBUG: Network nodes (first 5): ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
  Random         : +33.912
  Shortest-Len   : +52.204
  Dijkstra-Hop   : +47.696
  Max-SKR        : +146.552
  PPO+GNN        : +152.349 ◄ PPO
——————————————————————————————————————————————————————


════════════════════════════════════════════════════════════
  DIAGNOSTICS  (DEFENCE - MONSOON)
────────────────────────────────────────────────────────────
  Device             : cuda
  Wall time          : 77.7 min  (4664.6s)
  Peak GPU memory    : 306.6 MB
  env.step() — n=28800
    mean : 54.0 ms
    p50  : 46.8 ms
    p95  : 86.7 ms
    max  : 562.2 ms
    sum  : 1555.6 s  (33% of wall)
  ⚠  mean step > 30ms — check physics recompute frequency
════════════════════════════════════════════════════════════
✓  GNNEncoder (defence - monsoon) — checkpoint saved → checkpoint_defence_monsoon.pt

```
