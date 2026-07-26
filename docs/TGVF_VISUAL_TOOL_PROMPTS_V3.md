# TGVF Visual Tool Prompts v3

Prompt version: `tgvf-visual-tool-prompts-v3`.

Version 3 preserves the v2 shared user prompt and successful tool responses.
It repairs the Crop coordinate contract for the primary Qwen3-VL policy:

- `bbox_2d` is `[x1,y1,x2,y2]` on Qwen3-VL's original-image-relative
  `0..1000` grid;
- all coordinates remain within `0..1000`, with `x2 > x1` and `y2 > y1`;
- both Crop system prompts and both Crop tool schemas state that convention;
- the runtime converts the sampled model box to immutable-source pixels and
  records both boxes under `CROP-COORDINATES-20260724`.

This is a Qwen3-VL prompt/schema contract. Qwen2.5-VL requires a separately
identified prompt/schema plus the actual processor-resized geometry; the
runtime adapter fails closed when that geometry is missing.

The shared human-readable and native text template hashes remain:

- `e44a55bbf2f35a8b34cab1462af499ee4741f19e0561d27f130b8f2fd2316c60`;
- `8ccbdaa73d2b470afa7cd087e87ed42e2556e6bb3cf6c51fd414d7ae9eaedb6e`.

| Profile | Tool schema SHA-256 | Prompt bundle SHA-256 |
|---|---|---|
| TGVF only (unchanged v2 identity) | `f33f61d48bc4341f88077e90afca941819769b6209eb54893a9ed6b44856aba5` | `b44d8a461709bcf73e1447468b69ccfc93409aad514d7a3fcd8e40e849011c8a` |
| Crop only | `db46b434b97bea551038dd30990847eb605c40409c4220762229f692cd21a3c0` | `0df7f1b89c875c5cc2f3fe47a70e9814f2d40563199d4c20672bbd7175bc1359` |
| TGVF + Crop | `0f73b2e8c06a88d3fc08857843d153fb7138c4a3f66d64b4e6dd2c6dfef1ca39` | `6a9d7bb4cd5ee1ef08cb38d8e5c20b651e94756a73968f9e8bd8358bd45ffe04` |

The tokenizer/chat-template golden fixture is
`tests/fixtures/qwen3_visual_tool_prompts_v3.json`. The v1 and v2 files remain
historical provenance and are not identities for a repaired Crop run.
