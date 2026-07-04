# Blind Pairwise Packet v1

Updated: 2026-06-08

This packet prepares a blind A/B preference bundle for Distinct5-WikiArt without exposing method names to the evaluator.

## Contents

- [blind_pairwise_manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/blind_pairwise_v1/blind_pairwise_manifest.csv)
- [blind_pairwise_manifest.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/blind_pairwise_v1/blind_pairwise_manifest.json)
- [blind_pairwise_rubric.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/blind_pairwise_v1/blind_pairwise_rubric.md)
- [exploratory_blind_audit_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/blind_pairwise_v1/exploratory_blind_audit_summary.csv)
- panel directory:
  - `aaai2027/blind_pairwise_v1/panels/`

## Comparisons included

- `LBM-Knee` vs `SaMST`
- `LBM-Knee` vs `Seedream`
- `LBM-PS-v2` vs `SaMST`
- `LBM-K` vs `IDT`

## Panel layout

Each panel shows:

1. `Source`
2. `Candidate A`
3. `Candidate B`
4. `Target ref`

Candidate A/B order is randomized per panel and stored only in the manifest.

## Intended blind questions

For each panel, score:

1. Which candidate better matches the target style?
2. Which candidate better preserves the source content and structure?
3. Which candidate has fewer visible artifacts?

Answer format:

- `A better`
- `B better`
- `Tie`

## Status

- packet generation: `done`
- blinded manifest: `done`
- evaluator backend:
  - local VLM backend not yet wired in the current workspace
  - human or external VLM scoring still pending

## Generation script

- [scripts_gen_blind_pairwise_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/scripts_gen_blind_pairwise_packet.py)
