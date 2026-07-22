# TGVF Visual Tool Prompts v2

Prompt version: `tgvf-visual-tool-prompts-v2`. Tool schemas, profile-specific
system prompts, successful tool responses, and the four-call cap are unchanged
from v1. Version 2 changes only the shared user prompt so the final answer is
deterministically extractable without introducing a custom answer tag or a new
tokenizer token.

```text
<image>
{question}

Use the available visual tool if additional visual evidence is needed.

After completing your reasoning, give only the final answer without explanation:
- For multiple-choice questions, give only the option letter.
- For mathematics questions, give only the final value or expression.
- For other questions, give only a concise answer.
```

The shared human-readable template SHA-256 is
`e44a55bbf2f35a8b34cab1462af499ee4741f19e0561d27f130b8f2fd2316c60`;
the native text-item template SHA-256 is
`8ccbdaa73d2b470afa7cd087e87ed42e2556e6bb3cf6c51fd414d7ae9eaedb6e`.

| Profile | Prompt bundle SHA-256 |
|---|---|
| TGVF only | `b44d8a461709bcf73e1447468b69ccfc93409aad514d7a3fcd8e40e849011c8a` |
| Crop only | `4bc9d8e2a0ec2772d608b18281e6afe5cf02662851e58826e0202a5ac342dab0` |
| TGVF + Crop | `c860c0fadaf9a175105a74e3db3881005c4f0875a2e107412a9ca18910aba150` |

Historical direct benchmark baselines retain their original VLMEvalKit prompt.
Tool-enabled Policy evaluation uses v2 and supplies only the post-reasoning
final answer to VLMEvalKit while retaining the complete raw trajectory for
audit.
