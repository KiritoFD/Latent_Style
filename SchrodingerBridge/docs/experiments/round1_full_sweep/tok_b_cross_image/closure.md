# tok_b_cross_image Closure

- Status: `recalibration_needed`
- Current read:
  - this family did enter a real strict formal lane on its first tokenizer-tail retry:
    - `batch=8`
    - representative early live read about `9793MiB`
  - but the same run later drifted down within `epoch_0001` and was killed by the under-band guard:
    - about `8279MiB`
    - before the first retained checkpoint landed
  - the followup `batch=9` and `batch=10` retries did not yield a clean second bracket point:
    - both runs reached the launcher / trainer initialization path
    - but then hit reproducible `OSError: [Errno 5] Input/output error` on the run-root training log path before a trustworthy formal read could be taken
- Closure consequence:
  - `tok_b_cross_image` is now a mixed tokenizer-tail recalibration case:
    - first useful strict point says `batch=8` trends under-band late
    - later retries are confounded by run-root I/O instability
  - keep it in `recalibration_needed`
  - do not count the `batch=9/10` retries as clean formal evidence
