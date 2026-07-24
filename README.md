# Physics-Informed PPO+GNN Routing for Hybrid Fiber/FSO QKD Networks with Simplified Trusted Nodes
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-green.svg)](https://gymnasium.farama.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.5+-orange.svg)](https://pytorch-geometric.readthedocs.io/)
This project implements a discrete-time, finite-horizon Markov Decision Process (MDP) and a Proximal Policy Optimization (PPO) agent with a graph neural network (GNN) encoder to route quantum keys over a hybrid Fiber/Free-Space Optics (FSO) network. The physical and security model builds on the practical framework of **Simplified Trusted Nodes (STNs)** in the finite-key setting, but the routing problem itself is solved by a hop-by-hop, graph-structured learned policy rather than a fixed-candidate-path value-based agent.

---

## System Architecture & How It Works

Conventional Trusted Node (TN) networks distill an end-to-end key hop by hop, running a full cryptographic cycle at every intermediate relay. An STN network removes this bottleneck. Intermediate nodes measure independent raw keys with their immediate neighbors and broadcast the resulting parity bits over an authenticated classical channel, so the heavy error-correction and privacy-amplification work happens only once, at the two endpoints.

### The Operational Flow
1. **Quantum State Transmission:** Quantum states are streamed across each link along a chosen route.
2. **Local Key Matching:** Each intermediate STN measures independent raw keys ($L^i$ and $R^i$) with its left and right neighbors.
3. **Public Parity Broadcast:** Each STN computes a parity bit ($p^i = L^i \oplus R^i$) from its two raw keys and broadcasts it over an authenticated classical channel, consuming a small amount of its local authenticated key pool to secure that broadcast.
4. **Endpoint Key Extraction:** The two endpoints collect every parity along the route and run finite-key EC and PA themselves, distilling a single secure end-to-end key.

Because intermediate relays never correct errors locally, per-link noise is not filtered out hop by hop. It compounds across the whole path, and this compounding, not any single link's instantaneous quality, is what determines whether a route remains secure.

---

## Why the Physics Feeds the Network & RL Agent

In this setting, **the physical environment directly dictates network survival and routing feasibility**. Because intermediate relays do not run error correction step-by-step, **noise is not filtered out at each hop; it accumulates across the path.** The physical layer is therefore not a background simulation parameter, it is the primary driver of the MDP's state transitions and the routing policy itself.

### 1. Atmospheric and Medium-Induced Noise
* **Free-Space Optics (FSO) Vulnerabilities:** FSO channels are highly dynamic. The physics engine tracks the **Rytov variance** (modelling Gamma-Gamma atmospheric turbulence and scintillation), **sun elevation angle** (modelling background ambient noise at the receiver), and **pointing/geometric errors** (modelling physical misalignment of optical beams).
* **Fiber Thermal Drift:** Fiber links experience minor physical shifts due to diurnal temperature changes, influencing baseline channel transmittance and Quantum Bit Error Rate (QBER).
* **The Compound Risk:** A single link degraded by midday turbulence or thermal drift will corrupt the parity sequence, causing the accumulated path noise ($e^*_{\mathrm{path}}$) to exceed the absolute physical distillation boundary of $0.11$.

### 2. The Authentication Key Consumption Paradox
Every parity broadcast an STN makes must be cryptographically authenticated. This requires the nodes to consume a small portion of their **local authenticated key pool** ($K_{\mathrm{local}}$).
* Routing heavy traffic ($D$) repeatedly through the same intermediate nodes drains their pools even though those nodes never distill the end-to-end key themselves.
* Once a node's local pool hits zero, it can no longer broadcast parities securely, rendering that entire path dead.

### 3. The Dynamic "Refresh" State
To counter pool depletion, links must occasionally enter a **background local refresh mode** to run localized link-level post-processing and restock their authenticated reserves. A link undergoing background refresh cannot simultaneously support active end-to-end user routing, so pool management is a genuine scheduling trade-off, not a side constraint.

---

## The Indian Fiber/FSO Corridor Topology

Training and evaluation run over a **162-node** hybrid network built around eight major Indian cities (Delhi, Mumbai, Bangalore, Chennai, Kolkata, Hyderabad, Ahmedabad, Pune), fixed as endpoint anchors and connected along ten designated national backbone corridors. Direct point-to-point links along these corridors would span hundreds of kilometers, which is not physically realistic for a single quantum channel, so each corridor is densified with intermediate STN relay nodes under a maximum single-hop spacing of **80 km**, the same relay-spacing convention used in the base STN routing-policy comparison this project follows.

The 162-node target size was found empirically rather than assumed. Reducing the relay density from an initial 300-node graph under the same 80 km spacing and nearest-neighbor connection rule, the topology stopped being fully connected at 161 nodes, at least one corridor could no longer be spanned within the hop-length limit. At 162 nodes, connectivity is restored across every corridor, making this the minimum relay density at which the topology remains a single connected graph under the spacing constraint.

---

## The Reinforcement Learning Formulation

The routing agent is a centralized controller that observes the live physical and resource state of the entire network and selects the next hop at every step, rather than selecting an index into a path list computed offline.

### Graph-Structured State and Hop-by-Hop Actions
Unlike a fixed-candidate-path formulation, the state here is the network's full node and edge feature graph at the current timestep: per-node features covering local pool health, node type, diurnal phase, and progress toward the destination, and per-edge features covering link medium, QBER, normalized secret key rate, and a single-hop chain-noise correction term. The action is the choice of next-hop neighbor, masked to the current node's actual adjacency, so the action space stays bounded by local degree rather than growing with network size or with the number of candidate paths considered.

### GNN Encoder and Destination-Aware Attention Actor
A message-passing GNN encoder produces node embeddings from this graph, allowing the policy to generalize across topology rather than memorize per-node behavior. A destination-aware, multi-head attention actor then scores each neighbor of the current node relative to both the current node and the destination, with a hard-threshold gate that masks out any neighbor whose link currently violates the security or throughput floor before a next hop is ever sampled. A PPO training loop, using Generalized Advantage Estimation and a clipped surrogate objective, updates the shared encoder and actor-critic heads jointly.

### Reward Function Constraints
The reward structure uses a **hard early-exit constraint** backed by finite-key security proofs. If the compounded end-to-end path QBER, or the finite-key chain-noise correction over the path's STN hops, spikes above $0.11$, the agent receives a catastrophic penalty ($-C_{\mathrm{sec}}$) and no other term is evaluated.

Otherwise, the policy optimizes a use-case-conditioned composite reward surface:
* **Security Margin:** Rewards distance from the QBER limit, with a smooth penalty ramp active in a warning band below it.
* **Throughput Maximization ($r_{\mathrm{SKR}}$):** Rewards the path's weakest-link secret key rate under finite-size block restrictions ($n = 10^6$).
* **Inventory Preservation ($r_{\mathrm{pool}}, r_{\mathrm{dep}}$):** Rewards the path's minimum key-pool ratio and its trend, and penalizes the agent heavily if it drives intermediate node reserves into critical zones.
* **Congestion Penalty:** Penalizes the squared rate of pool depletion across the path.
* **Proactive Failovers ($r_{\mathrm{switch}}$):** Includes a route-switching penalty to prevent constant jitter. Crucially, this penalty is **conditionally waived** if the active path's QBER enters a pre-emptive warning zone ($e_w = 0.08$), training the agent to actively route traffic *around* a weakening FSO link right before atmospheric conditions destroy its security margin.

The relative weight of each term is conditioned on a declared deployment use-case, defence, commercial, or research, so the same environment and reward structure can express very different operational priorities without redesigning the reward by hand for each scenario.

---

## Baselines and Ablations

The trained PPO+GNN policy is evaluated against a random policy, an unweighted shortest-hop policy, three reactive Dijkstra variants reweighted by instantaneous latency, secret key rate, or QBER, a resource-aware Dijkstra variant that removes insecure or depleted links before reweighting by key-pool emptiness, and an integer-linear-programming oracle that selects the max-min pool-ratio path among the shortest candidate routes on the currently secure subgraph. A PPO+LSTM ablation, identical in every respect except that the GNN encoder is replaced by a sequence-only LSTM over the node features, isolates the contribution of relational, graph-structured inductive bias from the contribution of learning and the attention actor alone.

---

## 📜 Citation

If you use this environment or codebase in your academic work, please consider citing the underlying security framework:

```bibtex
@inproceedings{krawec2024finite,
  title={Finite Key Security of Simplified Trusted Node Networks},
  author={Krawec, Walter O. and Wang, Bing and Brown, Ryan},
  booktitle={2024 IEEE International Conference on Quantum Computing and Engineering (QCE)},
  year={2024},
  organization={IEEE}
}
```
