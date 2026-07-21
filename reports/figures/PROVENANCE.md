# Figure provenance

The report figures are bounded copies of three source images already selected
by the audited representation internal-evaluation manifest. They are included
only to explain the diagnostic construction; they are not additional training
or evaluation samples.

| Local asset | Dataset/source ID | Source file | SHA256 | Report role |
|---|---|---|---|---|
| `baseball_2410492.jpg` | Visual Genome `2410492` | `/home/dredvpn009/Flash_Storage/datasets/visual_genome/VG_100K_2/2410492.jpg` | `541c4d1390152604e90f796c20c36646f82841673e88e376e26297ce32b91a86` | Same-image positive/negative target-presence example |
| `chart_2017.png` | ChartQA `train_016650` | `/home/dredvpn009/Flash_Storage/datasets/chartqa/images/train/train_016650.png` | `f1c256201cd86661826e80444f0b4e69053174be2d14f9fa8e4299580864e3f8` | Cross-image value-flip example, value 2017 |
| `chart_2019.png` | ChartQA `train_014325` | `/home/dredvpn009/Flash_Storage/datasets/chartqa/images/train/train_014325.png` | `5df18cd8fa4ebc43857731606f8de473e2b67d6f22cc93394b2d916c52f669db` | Cross-image value-flip example, value 2019 |

Audited manifest:
`configs/representation/internal_evaluation/qwen3_v4_clean_imend_audited_grounding_v1.json`,
file SHA256
`a65aa6e6038ada1436302b60440136cc98b388552a7782b48ec95ed4324938c0`.
