# Late-Stokes Family Summary

Date: 2026-06-07

This note summarizes what the `Stokes` mechanism now means on the `Distinct5-512`
mainline after both from-scratch and late-fine-tune packets have landed.

## Family read

The family is no longer ambiguous.

What the evidence now says is:

- `Stokes` from scratch is mostly a **style-for-LPIPS tradeoff** mechanism.
- `Stokes` applied **late**, after the `P_attn` geometry is already formed, is a
  **useful repair family**.
- inside the late-fine-tune regime, `w_stokes_viscous` behaves like a
  continuous tradeoff knob, not a binary good/bad switch.

## Useful operating points

Reference anchor before late `Stokes`:

- `P_attn` continuation `e11`
  - transfer `0.7289 / 0.6211`

Late-fine-tune points now established:

- `0.05`
  - transfer `0.7274 / 0.6033`
  - read: better LPIPS-balanced near-frontier point

- `0.02`
  - transfer `0.7307 / 0.6183`
  - read: better raw-style point, but gives back part of the LPIPS gain

So the family currently exposes two paper-useful anchors:

- `0.05` for balance
- `0.02` for style ceiling

## Why the next probe is `0.03`

The current curve is monotone in the way we would expect:

- weakening `Stokes` from `0.05` to `0.02` recovered style
- but also lost LPIPS

That makes `0.03` the highest-value next probe:

- it stays inside the same mechanism family
- it targets the middle of a now-observed one-dimensional tradeoff curve
- and it has a real chance to dominate both endpoints if the curve is smooth

Current status:

- `0.03` packet is authored and committed
- a remote idle-wait watcher is live
- launch is blocked only by host-side Windows graphics memory, not by WSL jobs
