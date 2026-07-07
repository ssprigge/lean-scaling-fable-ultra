"""leanscale: empirical scaling laws of coding-LLM predictability by language.

Implements the measurement design of Gwern's "Lean Software Scaling Laws"
(https://gwern.net/lean-scaling): measure per-position perplexity of a frozen
LLM over increasingly long source-code contexts, normalize to bits-per-byte,
fit per-language power laws, and extrapolate crossovers; with bug-injection
and type-signature-ablation cross-checks.
"""

__version__ = "0.1.0"
