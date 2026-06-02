# Device-Aware Dueling DQN Routing for Hybrid Fiber/FSO QKD Networks with Simplified Trusted Nodes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-green.svg)](https://gymnasium.farama.org/)

This project implements a discrete-time, finite-horizon Markov Decision Process (MDP) and a Dueling Deep Q-Network (Dueling DQN) agent to optimize quantum key routing over hybrid Fiber/Free-Space Optics (FSO) networks. The network architecture is built upon the practical framework of **Simplified Trusted Nodes (STNs)** in the finite-key setting.

---

## System Architecture & How It Works

Traditional Trusted Node (TN) networks require every intermediate relay node to perform a computationally intensive cryptographic cycle: error correction (EC) and privacy amplification (PA) for every link, on every session. 

An **STN network** removes this bottleneck by shifting the heavy quantum post-processing calculations away from intermediate relays and isolating them *strictly* at the terminal endpoints (Alice and Bob). 

### The Operational Flow
1. **Quantum State Transmission:** Quantum states are streamed across adjacent links along a path.
2. **Local Key Matching:** Intermediate STNs perform measurements to establish independent raw keys ($L^i$ and $R^i$) with their immediate left and right neighbors.
3. **Public Parity Broadcast:** Instead of decrypting and re-encrypting user data, the intermediate STN computes raw parity bits ($p^i = L^i \oplus R^i$) and broadcasts them over standard classical channels.
4. **Endpoint Key Extraction:** Alice and Bob collect all public parities along the selected route. Using this structural parity chain, they run the finite-key EC and PA protocols exclusively at the endpoints to distill a secure, direct end-to-end secret key.

---

## Why the Physics Feeds the Network & RL Agent

In an STN infrastructure, **the physical environment directly dictates network survival and routing feasibility**. Because intermediate relays do not run error correction step-by-step, **noise is not filtered out at each hop; it accumulates exponentially across the path.** Therefore, the physical layer changes from a background simulation parameter into the primary driver of the MDP state transition and the RL policy.

### 1. Atmospheric and Medium-Induced Noise
* **Free-Space Optics (FSO) Vulnerabilities:** FSO channels are highly dynamic. The physics engine tracks the **Rytov variance** (modelling Gamma-Gamma atmospheric turbulence and scintillation), **sun elevation angle** (modelling background ambient noise blinding detectors), and **pointing/geometric errors** (modelling physical misalignment of optical beams).
* **Fiber Thermal Drift:** Fiber lines experience minor physical shifts due to diurnal temperature changes, influencing baseline channel transmittance and Quantum Bit Error Rate (QBER).
* **The Compound Risk:** A single link degraded by early morning fog or solar alignment will corrupt the parity sequence, causing the accumulated path noise ($e^*_{\mathrm{path}}$) to exceed the absolute physical distillation boundary of $0.11$.

### 2. The Authentication Key Consumption Paradox
To keep the network "simplified," STNs must use public classical channels to announce their parities. However, to guarantee information-theoretic security, **these classical announcements must be cryptographically authenticated** (e.g., via Wegman-Carter polynomial hashing).
* This authentication requires the nodes to consume a small portion of their **local authenticated key pool** ($K_{\mathrm{local}}$).
* If an RL agent routes high-throughput user traffic ($D$) through a path repeatedly, the intermediate nodes will exhaust their local authenticated pools just trying to sign the parity broadcasts.
* Once a node's local pool hits zero, it can no longer broadcast parities securely, rendering that entire path dead. 

### 3. The Dynamic "Refresh" State
To counter pool depletion, links must occasionally enter a **background local refresh mode** to run localized link-level post-processing and restock their authenticated reserves. However, a link undergoing a background refresh cannot support active end-to-end user routing.

---

## The Reinforcement Learning Formulation

The RL agent acts as a centralized network controller that views these physical properties and makes real-time scheduling decisions.

### How State Features Shape Policy
The agent receives an augmented feature vector containing the physical parameters of every link, alongside a smooth $\sin$/$\cos$ time-of-day encoding. 
* **Predicting Diurnal Weather Patterns:** By observing the sun elevation angle, time encoding, and Rytov variance, the agent learns to *anticipate* when FSO links will degrade (e.g., at dawn or during mid-day thermal turbulence).
* **Managing Inventory Transience:** By observing $K_{\mathrm{local}}$ and the active refresh flags ($\mathbb{1}_{\mathrm{refresh}}$), the agent learns to balance instantaneous network throughput against future resource depletion.

### Reward Function Constraints
The reward structure uses a **hard early-exit constraint** backed by finite-key security proofs. If the calculated end-to-end path QBER spikes above the cryptographic limit, the agent receives a catastrophic penalty ($-C_{\mathrm{sec}}$).

Otherwise, the policy optimizes a composite reward surface:
* **Throughput Maximization ($r_{\mathrm{SKR}}$):** Normalizes the extracted path key rate under finite-size block restrictions ($n = 10^6$).
* **Inventory Preservation ($r_{\mathrm{pool}}, r_{\mathrm{dep}}$):** Penalizes the agent heavily if it drives intermediate node reserves into critical zones.
* **Proactive Failovers ($r_{\mathrm{switch}}$):** Includes a route-switching penalty to prevent constant jitter. Crucially, this penalty is **conditionally waived** if the active path's QBER enters a pre-emptive warning zone ($e_w = 0.08$), training the agent to actively route traffic *around* an FSO link right before atmospheric conditions destroy its security margin.

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
