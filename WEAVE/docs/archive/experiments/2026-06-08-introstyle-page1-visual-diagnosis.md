# IntroStyle Page-1 Visual Diagnosis

Date: 2026-06-08

Artifacts:

- visual packet:
  - [introstyle_page1_visual_packet.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/introstyle_page1/visual_packet/introstyle_page1_visual_packet.png)
  - [introstyle_page1_visual_packet.pdf](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/introstyle_page1/visual_packet/introstyle_page1_visual_packet.pdf)
- numeric shortlist:
  - [page1_shortlist_comparison.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/introstyle_page1/page1_shortlist_comparison.csv)

Scope:

- selected shared case manifest:
  - [multi_source_cases.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/introstyle_page1/multi_source_cases.csv)
- current packet spans multiple source families:
  - `Early_Renaissance -> Impressionism`
  - `Early_Renaissance -> Minimalism`
  - `Impressionism -> Minimalism`
  - `Minimalism -> Rococo`
  - `Rococo -> Ukiyo_e`
  - `Ukiyo_e -> Early_Renaissance`
- compared columns:
  - `IDT`
  - `SaMAM-2250`
  - `SaMST e15`
  - `Lat SaMAM`
  - `Lat SaMST`
  - `LBM-Knee`
  - `LBM-PS-v2`
  - `Seedream-4.5`
  - `Target ref`

## Visual read

### 1. Latent baselines are weak across multiple source families, not just one case

- `Lat SaMAM`
  - behaves close to a no-op
  - preserves the source layout and palette almost too literally
  - explains why its `IntroStyle delta-IDT` is effectively zero
- `Lat SaMST`
  - collapses into a blurry low-structure haze
  - it is not a subtle content-preserving stylization failure
  - it is a genuine representational collapse
  - this matches its strongly negative `IntroStyle delta-IDT`

### 2. Pixel baselines are not equivalent, and their failure modes are stable across content types

- `SaMAM-2250`
  - mostly applies conservative contrast / color / texture darkening
  - can move away from pure `IDT`, but usually does not achieve a strong target-specific look
  - this is consistent with weak positive `IntroStyle delta-IDT` but poor margin
- `SaMST e15`
  - can produce visibly target-like patterning
  - especially in `Impressionism` and `Minimalism`
  - but often damages structure and introduces heavy style takeover
  - this is still true when the source changes from:
    - narrative renaissance scene
    - landscape
    - flat minimal field
    - portrait
    - ukiyo-e mountain composition
  - this explains why it beats `SaMAM` on smoke `IntroStyle delta-IDT` while still sitting in a bad LPIPS regime

### 3. The current LBM family splits into two distinct failure modes that now look source-agnostic

- `LBM-Knee`
  - keeps scene geometry and object placement unusually well
  - but the stylization often looks pale, washed, and under-committed
  - the model seems to trace structure while under-injecting visible target-style texture hierarchy
  - this now repeats on:
    - complex figure scenes
    - line-structured landscapes
    - portraits
    - already-stylized ukiyo-e sources
- `LBM-PS-v2`
  - stylization is more aggressive than `LBM-Knee`
  - but it often becomes a generic foggy painterly layer
  - the result can score well on `CLIP` while looking less target-specific than desired
  - on the expanded packet it frequently behaves like a domain-level `painting filter`
    rather than a clean target-style attribution mechanism
  - visually this supports the concern that `CLIP-friendly stylization` is not the same as true style attribution

### 4. Seedream remains strongest on obvious style injection, but its advantage is now more specific

- `Seedream-4.5`
  - is the most visually explicit on target-style cues across these four cases
  - it is especially clear in `Minimalism`, where it makes a categorical visual move instead of just light painterly drift
  - it also stays more target-specific than `LBM-PS-v2` on the expanded packet
  - however, it also preserves some source shapes less faithfully than `LBM-Knee`

### 5. The hard mechanism target is now sharper

The expanded packet suggests the missing ingredient is not simply:

- stronger color shift
- or more painterly blur

Instead, the missing ingredient looks like:

- target-style-specific mid/high-frequency organization
- with source-aware spatial placement
- especially around:
  - contour neighborhoods
  - repetitive texture fields
  - portrait/background separation
  - flat-region decorative fill

## Mechanism implication

The present gap is now visually clearer:

- `LBM-Knee` already has the geometry anchor
- the next mechanism should not aim only for more scalar style score
- it should specifically try to add:
  - stronger target-style texture hierarchy
  - more style-specific spatial statistics
  - without collapsing into:
    - `SaMST`-style structure damage
    - or `LBM-PS-v2`-style generic painterly fog

In other words, the next mechanism question is:

- can we reopen style from the `LBM-Knee` geometry basin
- while making the style visually more explicit and target-specific
- not merely more painterly in a generic way

The expanded packet adds one more constraint:

- the new mechanism should be tested against multiple source families immediately
- otherwise a gain on one narrative scene can still hide a generic filter solution
