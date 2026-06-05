# One-Week Model Improvement Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Spend the next week maximizing acceptance odds by improving our own model, repeating the right experiments, and tightening design-validity evidence on the remote `RTX 3060` WSL surface.

**Architecture:** The main line is no longer a baseline-first plan. We use the existing reviewer ledger to prioritize: improve the executor-side/mainline model, run a narrow stability-focused hyperparameter sweep, repeat only the paper-facing wins, and add only the cheapest extra controls that answer concrete reviewer questions. `latent SaMam` stays as a bounded side quest, not the core schedule.

**Tech Stack:** SchrodingerBridge training/eval stack, remote `RTX 3060` WSL, existing Distinct5-512 evaluator contract, reviewed config packets, git with frequent small commits, experiment docs in `SchrodingerBridge/docs/experiments/`.

---

## Priority Reset

This plan supersedes the earlier two-day latent-SaMam-first plan.

Primary objective order for the next week:

1. improve our own model frontier
2. add repeat / robustness evidence for our design
3. answer specific reviewer mechanism and implementation questions
4. keep baseline expansion bounded and opportunistic

What is already closed and should not consume more formal budget:

- endpoint-only metric closure: negative and done
- semantic-vs-random axis closure: negative and done

What is still high-value:

- executor-side improvement on the paper-facing family
- same-family stability / path evidence packaging
- repeatability of the best point
- cheap controls for pairing cache / projection sensitivity / implementation clarity

## Fixed Constraints

- formal GPU runs must use the remote `3060` WSL surface
- no local GPU usage
- every experiment lane must write a dated note under `SchrodingerBridge/docs/experiments/`
- every meaningful code / config / result milestone gets its own small commit and push
- `latent SaMam` must not eat more than about half a day unless it becomes obviously cheap and competitive

## Definition Of Done For The Week

By the end of the week, at least these should be true:

1. one improved paper-facing LBM variant has been trained and fully evaluated on Distinct5-512
2. that variant has at least one repeat or robustness packet, not just a single lucky run
3. we have at least one concise supplementary control packet answering a concrete reviewer question about design validity
4. the experiment ledger clearly states what is positive evidence, what is negative closure, and what is only bounded context
5. `latent SaMam` is either cheaply ruled out or left as a side note with explicit stop reasons

## Main Track A: Improve The Model

### A1. Executor-side promotion on the paper-facing family

Reason:

- the landed tokenizer-localization packet says executor-only refresh is stronger than style-branch refresh on the matched `L e1` surface
- that is the strongest current hint for where model improvement should come from

Deliverables:

- one remote packet that ports the executor-side insight onto the current paper-facing family
- one short note explaining the exact transfer of the idea and whether it improves `delta_idt`, `LPIPS`, or `ArtFID`

Concrete target:

- create one `H`-family or current mainline successor variant that reuses the executor-focused refresh idea instead of opening a large tokenizer redesign

Stop rule:

- if the first clean packet is clearly worse than the current mainline on both `delta_idt` and `LPIPS`, do not spawn a whole branch family from it

### A2. Narrow stability / force-balance sweep

Reason:

- older black-dot mitigation evidence already narrowed a useful range:
  - `terminal_swd_weight = 10.0 ~ 11.0`
  - `w_kinetic = 0.42 ~ 0.48`
  - `w_cycle = 0.15 ~ 0.20`
  - `semantic_attn_temperature = 0.11 ~ 0.13`
- this is a credible path to a better visual and LPIPS frontier without reopening giant search space

Deliverables:

- a compact 3-4 run remote sweep on Distinct5-512
- one sweep note with a small table and explicit keep/drop decision

Success condition:

- at least one point is visibly cleaner or better on `LPIPS` without giving back all no-op-adjusted style movement

Stop rule:

- do not exceed 4 formal configs before choosing a keep candidate

### A3. Promote one new candidate only

Once `A1` and `A2` produce candidates:

- choose one best candidate
- run full eval
- put it into the same comparison ledger as current `F/H/K` references

Do not keep multiple equally half-baked candidates alive in the paper.

## Main Track B: Repeat And Prove

### B1. Repeat the best candidate

Reason:

- one-off wins are weak reviewer evidence
- current review pressure is on rigor and stability, not only raw idea novelty

Deliverables:

- one additional seed or repeat packet for the chosen best candidate
- machine-readable summary with mean / spread for the paper-facing metrics

Minimum metrics to report:

- full `clip_style`
- transfer `clip_style`
- `delta_idt_full`
- `delta_idt_transfer`
- `content_lpips`
- targetwise `ArtFID` if the full packet is retained

### B2. Same-budget retrace for our own variant

Reason:

- same-cost discussion is useful only if our own improved point has a clean timing anchor

Deliverables:

- one timing-backed same-budget point for the improved LBM variant
- update the timing ledger and same-cost inventory

Important boundary:

- this is mainly to place our improved model on our own frontier, not to reopen giant baseline-time rhetoric

## Main Track C: Cheap Reviewer Controls

These must be cheap and directly tied to review questions.

### C1. Pairing-cache sensitivity

Reviewer question:

- does the prototype-aware pairing cache matter, or would naive pairing do the same thing?

Deliverable:

- one short-budget control comparing current pairing cache vs random/simple pairing

Desired outcome:

- either positive support for the cache or a bounded statement that it is a mild helper rather than a central theorem

### C2. Projection-count / projection-source sensitivity

Reviewer question:

- are projection choices arbitrary or brittle?

Deliverable:

- one cheap sensitivity packet on projection count or simplified projection routing

Important boundary:

- do not reopen semantic-vs-random superiority claims
- this control is only for sensitivity / implementation hygiene

### C3. Implementation clarity packet

Reviewer question:

- VAE details, latent size consistency, and data roots are still easy to attack

Deliverable:

- one doc note that records exact latent shape, VAE path, scaling, decode contract, train root, test root, and pairing cache path for the current paper-facing runs

This is documentation work, but it directly reduces review risk.

## Side Track D: Baseline / Latent Work

### D1. `latent SaMam` smoke only

Budget:

- at most half a day unless it becomes trivially cheap and promising

Goal:

- determine feasibility, not build a whole parallel story

Acceptable outcomes:

1. a tiny viable packet and one sanity output
2. a clean negative closure note saying why it should not absorb more time now

### D2. Other baseline reproduction only if main queue is blocked

If the remote queue would otherwise go idle overnight:

- allow one bounded auxiliary baseline task

Otherwise:

- do not let baseline curiosity displace model-improvement work

## Suggested Day Split

### Day 1

- refresh the living experiment ledger
- define exact `A1` executor-side promotion packet
- define exact `A2` narrow sweep configs
- commit config/docs packet before launch

### Day 2

- run `A1`
- launch first half of `A2`
- write the first result note

### Day 3

- finish `A2`
- pick the single promoted candidate
- update comparison ledger and timing note

### Day 4

- run `B1` repeat packet
- if queue space remains, run `C1` pairing-cache control

### Day 5

- run `C2` projection sensitivity
- write `C3` implementation clarity note

### Day 6

- use the best open slot for either:
  - `latent SaMam` smoke
  - one more repeat if the improved model looks promising but variance is unclear

### Day 7

- consolidate:
  - experiment notes
  - timing ledger
  - claim boundaries
  - paper-facing keep/drop decisions

## Required Living Docs

Maintain or create these during the week:

- one weekly top-level tracker note under `SchrodingerBridge/docs/experiments/`
- one dated note per packet
- timing updates under `SchrodingerBridge/docs/timing/`
- same-cost updates under `SchrodingerBridge/docs/experiments/2026-06-04-distinct5_same_cost_inventory.csv`

## Commit Rhythm

Minimum commit points:

1. plan correction landed
2. config packet for `A1/A2` landed
3. first improved-model result landed
4. promoted candidate and full eval landed
5. repeat/control packets landed
6. weekly consolidation note landed

Every one of these should be pushed immediately.

## What Should Explicitly Stop

- no more formal endpoint-only reruns
- no more formal semantic-vs-random reruns
- no big baseline-first branch unless our main queue is blocked
- no making `latent SaMam` the headline task
- no broad speed rhetoric experiments unless we explicitly decide to reopen strong efficiency claims

## End-Of-Week Decision Rule

At the end of the week, the paper-facing model story should be one of:

1. **Improved model promoted**
   - we found a better and repeated LBM variant

2. **Current model defended**
   - no better variant survived, but we added stronger repeatability and design-validity evidence

3. **Hybrid**
   - a modestly improved variant plus stronger validity controls

All three are acceptable. What is not acceptable is spending the week mostly on side baselines while our own model remains under-justified.
