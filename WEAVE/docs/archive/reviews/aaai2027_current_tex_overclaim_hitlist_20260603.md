# AAAI 2027 Current TeX Overclaim Hit List

Date: 2026-06-03  
Lane: `adversarial_review`  
Scope: only the most urgent current overclaim-risk lines in `paper_aaai2026.tex`

## 1. Abstract

- section:
  - `Abstract`
- quote snippet:
  - `the main remaining bottleneck is not tokenizer size, but faithful style execution through the latent renderer`
- why evidence is not enough now:
  - current tokenizer evidence is still concentrated in the present Distinct5 family and one renderer setting;
  - this is stronger than what the ablation table and probes actually close.
- safer wording:
  - `within the current Distinct5 tokenizer family, the main observed bottleneck is faithful style execution through the latent renderer rather than raw tokenizer size alone`

## 2. Contributions

- section:
  - `Our contributions`
- quote snippet:
  - `A terminal-matching design used in the current mainline (SA-SWD)`
- why evidence is not enough now:
  - this is much safer than before, but Gate B is still open;
  - the matched semantic-vs-random control is not closed, so even contribution-list placement can still read stronger than the evidence.
- safer wording:
  - `A semantic-aligned terminal-matching design used in the current mainline, with semantic-axis necessity evaluated separately by an ongoing matched control`

## 3. Method / Tokenizer

- section:
  - `Style tokenizer as executable control`
- quote snippet:
  - `increasing tokenizer capacity alone did not break the style ceiling`
- why evidence is not enough now:
  - this is currently supported only by the tested tokenizer variants under the present Distinct5 setup;
  - it risks reading like a broader tokenizer theorem.
- safer wording:
  - `within the tested Distinct5 tokenizer variants, increasing tokenizer capacity alone did not break the style ceiling`

## 4. Method / Tokenizer conclusion

- section:
  - `Style tokenizer as executable control`
- quote snippet:
  - `We therefore treat tokenizer design as an executable representation problem`
- why evidence is not enough now:
  - the framing is plausible, but the `therefore` makes it sound fully established rather than a bounded interpretation of current probes.
- safer wording:
  - `We therefore interpret the current results as evidence that tokenizer design should be treated as an executable representation problem in this setup`

## 5. Historical cost paragraph

- section:
  - `Historical strict-750 comparison`, paragraph under `Table~\ref{tab:cost}`
- quote snippet:
  - `the proposed transport objective remains inexpensive to retrain and straightforward to evaluate at scale under the present protocol`
- why evidence is not enough now:
  - Gate C is still open;
  - even after softening, `inexpensive to retrain` is still a comparative efficiency conclusion without normalized time-to-parity.
- safer wording:
  - `the reproduced operating-point records indicate a manageable practical footprint under the present protocol, without constituting normalized time-to-parity evidence`

## 6. Distinct5 subsection

- section:
  - `Distinct5-512 stress benchmark`
- quote snippet:
  - `LBM occupies the strongest measured content-preserving frontier among the currently reproduced points`
- why evidence is not enough now:
  - mostly safe, but still vulnerable if read as a general frontier claim rather than a strict `full 5x5 / 750 / Distinct5` statement;
  - this section already does a lot of scope work, so one more pin is worth it.
- safer wording:
  - `under the current Distinct5-512 full 5x5/750 protocol, LBM occupies the strongest measured content-preserving frontier among the currently reproduced points`

## 7. Tokenizer ablation discussion

- section:
  - `Tokenizer and representation ablations`
- quote snippet:
  - `The next tokenizer should therefore expose two factors rather than one larger code`
- why evidence is not enough now:
  - this is design-direction advice, not yet a directly validated necessity;
  - current results support it as a leading hypothesis, not a closed requirement.
- safer wording:
  - `The current ablations suggest that the next tokenizer should likely expose two factors rather than one larger code`

## 8. Discussion

- section:
  - `Discussion and Limitations`
- quote snippet:
  - `the remaining weakness is faithful style execution rather than style separability in the tokenizer alone`
- why evidence is not enough now:
  - this is close to safe, but still reads broader than the actually tested family;
  - it should stay pinned to current reproduced protocols and tokenizer probes.
- safer wording:
  - `within the current reproduced protocols, the remaining weakness appears to be faithful style execution rather than tokenizer separability alone`

## 9. Conclusion

- section:
  - `Conclusion`
- quote snippet:
  - `The bottleneck is not raw token capacity, but executing a separable style carrier through a content-conditioned renderer without paying an LPIPS penalty`
- why evidence is not enough now:
  - same issue as the abstract and discussion;
  - too categorical relative to the present scope of tokenizer experiments.
- safer wording:
  - `The current tokenizer evidence suggests that the bottleneck is not raw token capacity alone, but executing a separable style carrier through a content-conditioned renderer without paying an LPIPS penalty`

## Immediate priority order

1. Fix the two tokenizer-bottleneck sentences in `Abstract` and `Conclusion`.
2. Soften the cost-paragraph tail sentence until Gate C closes.
3. Narrow the tokenizer-method and tokenizer-ablation `therefore` claims to `within the tested/current setup`.
4. Keep the SA-SWD contribution bullet explicitly subordinate to the still-open Gate B control.
