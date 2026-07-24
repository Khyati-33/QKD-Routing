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
Epoch  10/200 | R= +0.959 | w(q)=0.0580 | SKR=1.847 | Pool=0.372 | ent=0.1365 | LR=0.00080 | StepTime=0.0202s
Epoch  20/200 | R= +0.918 | w(q)=0.0568 | SKR=1.861 | Pool=0.367 | ent=0.1215 | LR=0.00079 | StepTime=0.0217s
Epoch  30/200 | R= +0.892 | w(q)=0.0588 | SKR=1.861 | Pool=0.367 | ent=0.1065 | LR=0.00078 | StepTime=0.0214s
Epoch  40/200 | R= +0.952 | w(q)=0.0576 | SKR=1.889 | Pool=0.358 | ent=0.0915 | LR=0.00075 | StepTime=0.0219s
Epoch  50/200 | R= +0.896 | w(q)=0.0604 | SKR=1.847 | Pool=0.372 | ent=0.0765 | LR=0.00072 | StepTime=0.0213s

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

Epoch  60/200 | R= +0.957 | w(q)=0.0593 | SKR=1.861 | Pool=0.367 | ent=0.0615 | LR=0.00067 | StepTime=0.0199s
Epoch  70/200 | R= +0.938 | w(q)=0.0590 | SKR=1.861 | Pool=0.367 | ent=0.0465 | LR=0.00062 | StepTime=0.0205s
Epoch  80/200 | R= +0.902 | w(q)=0.0578 | SKR=1.861 | Pool=0.367 | ent=0.0315 | LR=0.00056 | StepTime=0.0210s
Epoch  90/200 | R= +0.926 | w(q)=0.0578 | SKR=1.861 | Pool=0.367 | ent=0.0300 | LR=0.00050 | StepTime=0.0190s
Epoch 100/200 | R= +0.946 | w(q)=0.0578 | SKR=1.861 | Pool=0.367 | ent=0.0300 | LR=0.00044 | StepTime=0.0211s

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

Epoch 110/200 | R= +0.946 | w(q)=0.0601 | SKR=1.847 | Pool=0.372 | ent=0.0300 | LR=0.00037 | StepTime=0.0205s
Epoch 120/200 | R= +0.913 | w(q)=0.0579 | SKR=1.847 | Pool=0.372 | ent=0.0300 | LR=0.00031 | StepTime=0.0212s
Epoch 130/200 | R= +0.959 | w(q)=0.0599 | SKR=1.847 | Pool=0.372 | ent=0.0300 | LR=0.00024 | StepTime=0.0207s
Epoch 140/200 | R= +0.936 | w(q)=0.0617 | SKR=1.833 | Pool=0.377 | ent=0.0300 | LR=0.00019 | StepTime=0.0208s
Epoch 150/200 | R= +0.920 | w(q)=0.0577 | SKR=1.861 | Pool=0.367 | ent=0.0300 | LR=0.00014 | StepTime=0.0201s

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

Epoch 160/200 | R= +1.010 | w(q)=0.0570 | SKR=1.847 | Pool=0.372 | ent=0.0300 | LR=0.00009 | StepTime=0.0192s
Epoch 170/200 | R= +0.885 | w(q)=0.0587 | SKR=1.861 | Pool=0.367 | ent=0.0300 | LR=0.00006 | StepTime=0.0210s
Epoch 180/200 | R= +0.938 | w(q)=0.0590 | SKR=1.861 | Pool=0.367 | ent=0.0300 | LR=0.00003 | StepTime=0.0190s
Epoch 190/200 | R= +0.914 | w(q)=0.0586 | SKR=1.875 | Pool=0.363 | ent=0.0300 | LR=0.00001 | StepTime=0.0207s
Epoch 200/200 | R= +0.896 | w(q)=0.0586 | SKR=1.847 | Pool=0.372 | ent=0.0300 | LR=0.00001 | StepTime=0.0209s

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


✓  GNNEncoder (defence - monsoon) — checkpoint saved → checkpoint_defence_monsoon.pt

```
