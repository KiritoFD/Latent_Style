# CarrierGate Next Review Chain

Date: 2026-06-09

Current state:

- the current remote lane is:
  - `aaai2027_inmortal_hold4mid_e8_carriergate_injection_seed42_b8a2`
- its post-train watcher is:
  - [run_inmortal_posttrain_eval_latest_epochs_when_done.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_inmortal_posttrain_eval_latest_epochs_when_done.py)
- the watcher is configured to rerun clean eval only on the fresh epoch subset inferred from the latest training CSV

Updated status:

- the remote `carrier_gate injection` training process has now exited
- only the fresh-eval watcher remains alive
- once the watcher observes the process fully gone, it should launch:
  - `full_eval_fresh_localreview`
  - for the inferred fresh epoch subset only

Why this matters:

- the run root contains older checkpoints and older `full_eval` summaries
- the fresh-epoch watcher avoids mixing:
  - old `epoch_0007..0012`
  - with the current fresh continuation

Next local heavy-review chain:

1. wait for `full_eval_fresh_localreview/epoch_0001..0006` summaries
2. pull the resulting images / metrics to local if needed
   - helper:
     - [pull_remote_carriergate_fresh_eval.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/pull_remote_carriergate_fresh_eval.py)
3. run:
   - local `IntroStyle`
   - local `DINO`
   - local `Qwen xopqwen36v35b` panel review
4. compare against:
   - `LBM-Knee e13`
   - `Seedream-4.5`

Decision threshold:

- if the line cannot beat `LBM-Knee` on `IntroStyle`
- and also cannot stay left of `LBM-PS-v2` on `DINO`
- then it should be closed and not consume further main GPU time
