# Mainline inference crop-region figure provenance

**Date:** 2026-08-14

These figures are deterministic documentation views of real VStarBench source
images and archived policy actions. They are not generated or retouched model
illustrations.

Each JPEG panel contains:

- the complete source image, resized only for display;
- the exact Crop clean-final source-pixel bbox in red;
- the exact Atomic Crop+TGVF encoder-input source-pixel bbox in cyan;
- the two source-pixel crops, resized only to fit the right-hand panels.

All boxes use half-open `xyxy` coordinates: the right and bottom boundaries are
exclusive. The JPEG panels are visual aids rather than pixel-identity artifacts.
The archived decoded-RGB SHA remains authoritative for Crop clean-final.

## Coordinate provenance

Crop clean-final source bboxes and decoded-RGB SHA256 values are copied directly
from the formal trajectory audit. Re-cropping each immutable source image at the
recorded half-open bbox and hashing the decoded RGB bytes reproduces all three
archived SHA256 values exactly.

Atomic inference JSONL records the requested Qwen3 bbox but not the derived
source bbox. Its source bbox is deterministically reconstructed using the exact
production conversion `qwen3-relative-1000-floor-v1`:

```text
source bbox = [
  x1 * source_width  // 1000,
  y1 * source_height // 1000,
  x2 * source_width  // 1000,
  y2 * source_height // 1000,
]
```

The Atomic crop shown in a panel is the RGB input to Qwen vision. Its crop
pre-merge main/DeepStack features are then passed through frozen RP67 to produce
latent `D` and D-DeepStack. The policy receives those final latent tensors, not
the crop RGB directly.

## Figure identities

| figure | source image | source SHA256 | Crop model → source bbox | Crop RGB SHA256 | Atomic model → source bbox | panel SHA256 |
|---|---|---|---|---|---|---|
| `images/vstar_000000_crop_regions.jpg` | `VStarBench/0000_0.jpg`, 2000x1500 | `7fbbc4f28e251181d6e90e20e7f022dfaadf652f8b03419c936d7efee5caaaf7` | `[266,67,380,181]` → `[532,100,760,271]` | `14f4d3100a97580681d1b48dbac418feef8068b8a91487c3901c236070eb2ffd` | `[133,0,382,463]` → `[266,0,764,694]` | `ee6be4bcfd2cec593ffcf7f96757d39ede06a6e67a52e6e5217abd835ae7f90d` |
| `images/vstar_000148_crop_regions.jpg` | `VStarBench/0148_0.jpg`, 2250x1500 | `9021d99227c04b45680a66392a62f5083a4ede83b5b497ec1dbd2232937c1ea6` | `[65,569,459,851]` → `[146,853,1032,1276]` | `46892fba39c70aef0430fc4dbcf5e7eaa399b3276cd5c9bc87bdd69ba75bb5c8` | `[0,243,768,884]` → `[0,364,1728,1326]` | `99ca03c0143f161027b3e4fc0b2e6cea1b8b8aa4ad856b99c24120411a8f0956` |
| `images/vstar_000044_crop_regions.jpg` | `VStarBench/0044_0.jpg`, 3000x1500 | `7253f9bcdef974ea271249d87f2369e7637a964bda7f619d76933ed8574ae9ce` | `[432,587,500,706]` → `[1296,880,1500,1059]` | `17173b80e834f329821d5c51bc54a43ef91742bcfbc9fe38644f98130f9bd4bd` | `[454,568,550,725]` → `[1362,852,1650,1087]` | `a5dd67f910533577efc595400411e7e5019270d3a1100d014d76e516277b0339` |

## Source artifacts

```text
artifacts/evaluation/
  PRL14-A-CoreDev2511-cleanfinal-step0-step8-step16-v1/
    step8/inference/rank-*.jsonl

artifacts/policy/
  PRL-20-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-8step-ws8/
    evaluation/
      PRL20-R0-FROZEN-RP67-TFREE-CROP-TGVF-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/
        step8/inference/rank-*.jsonl
```
