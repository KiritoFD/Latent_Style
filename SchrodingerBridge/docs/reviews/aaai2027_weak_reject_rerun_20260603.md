# AAAI 2027 Weak-Reject Rerun Memo

Date: 2026-06-03  
Lane: `adversarial_review`

## 1) Top 5 current reject risks

1. **SA-SWD mechanism novelty is still under-isolated.**  
   The manuscript still depends on a semantic terminal-matching story whose semantic-vs-random control is not yet fully closed.

2. **Efficiency evidence is still not a fair normalized parity result.**  
   The new Distinct5 timing packet is much safer, but it is still a timing-context artifact rather than a matched time-to-threshold or time-to-parity proof.

3. **Tokenizer-to-renderer diagnosis is still only partially closed.**  
   The current paper safely says executed style survival is the sharper question, but it still lacks a direct closure experiment that isolates code-to-execution alignment as the next mechanism bottleneck.

4. **Distinct5 can still be attacked as a narrow metric-stress split.**  
   The manuscript is much better here, but any drift from `metric-stress benchmark` toward broader AST superiority would reopen a reviewer attack quickly.

5. **Perceptual validity is still proxy-based.**  
   Artifact-sensitive metrics help, but there is still no human study or rubric-style preference closure, which keeps some quality claims vulnerable if phrased too broadly.

## 2) Which are wording-only vs evidence-only vs experiment-required

- **Experiment-required**
  - Risk 1: SA-SWD mechanism novelty under-isolated
  - Risk 2: efficiency not yet normalized parity
  - Risk 3: tokenizer-to-renderer mechanism only partially closed

- **Wording-only**
  - Risk 4: Distinct5 scope drift beyond a metric-stress benchmark

- **Evidence-only**
  - Risk 5: perceptual validity remains proxy-based

## 3) One strict priority order for the next 3 closures

1. **Close Gate B first:** matched semantic-vs-random SA-SWD comparison, so the paper can either keep or demote the semantic-axis contribution cleanly.
2. **Close the next mechanism probe second:** direct tokenizer code-to-execution alignment evidence, so the renderer-side diagnosis becomes more than a bounded interpretation.
3. **Close Gate C third if stronger efficiency language is desired:** either produce a real matched parity rule with comparable curves, or freeze the paper permanently at `same-scope timing context` and stop short of parity claims.

## 4) What the paper still cannot claim

- It still cannot claim that **semantic projection-axis selection is proven necessary**.
- It still cannot claim a **fair comparative training-speed win** or a **normalized time-to-parity advantage**.
- It still cannot claim that **tokenizer size is generally not the bottleneck** beyond the tested Distinct5 tokenizer family and current setting.
- It still cannot claim a **universal latent-space metric theorem** such as `MSE is wrong everywhere in latent space`.
- It still cannot claim **broad AST superiority** beyond the current reproduced protocols, reproduced baselines, and the Distinct5 metric-stress split.
