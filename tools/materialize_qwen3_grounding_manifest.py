#!/usr/bin/env python3
"""Materialize the manually audited v4 Qwen3 native-D grounding manifest."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

from tgvf_rl.representation.training.data import load_retained_representation_jsonl
from tgvf_rl.representation.training.qwen3_grounding import (
    TARGET_PRESENCE_QUESTION,
)


SOURCE_PATH = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/revisit_vlm/data/tgvf_teacher/generated/"
    "runs/tgvf_v4_teacher_50k_clean_imend/splits/"
    "tgvf_v4_teacher_stage1_protocol_c_focus.test.jsonl"
)
SOURCE_SHA256 = "de61c731eb961825a77df587cd76c00eabfea75b5c6003096f3cc7f1a51dd82d"
RETAINED_MANIFEST_SHA256 = (
    "534f5b1e648d0bca2b1ea2ff02f81e1fb7abbb456f16faacbb118ca94f7306b0"
)
OUTPUT_PATH = Path(
    "/nvmesv/dredvpn009/projects/r-vlm/tgvf-e2e-rl/configs/representation/"
    "internal_evaluation/qwen3_v4_clean_imend_audited_grounding_v1.json"
)


# These rows were visually inspected against the exact source image bytes.
# Cross-image rows additionally share the Qwen3 max-pixels=262144 visual grid.
CROSS_IMAGE_AUDIT = (
    (
        "chart-tallest-2017-vs-2019",
        "tgvf_v4_teacher_50k:chartqa:train_016650:0::focus1",
        "tgvf_v4_teacher_50k:chartqa:train_014325:0::focus1",
        "Which year has the tallest bar in the chart?",
        "the tallest bar together with its year label and top edge",
        "2017",
        "2019",
        "The rightmost and tallest bar is labeled 2017.",
        "The tallest stacked bar is labeled 2019.",
    ),
    (
        "chart-tallest-2010-vs-2009",
        "tgvf_v4_teacher_50k:chartqa:train_011253:0::focus1",
        "tgvf_v4_teacher_50k:chartqa:train_023113:0::focus1",
        "Which year has the tallest bar in the chart?",
        "the tallest bar together with its year label and top edge",
        "2010",
        "2009",
        "The tallest bar is the 2010 bar labeled 26.3.",
        "The tallest bar is the 2009 bar labeled 49.6%.",
    ),
    (
        "chart-tallest-2018-vs-2017",
        "tgvf_v4_teacher_50k:chartqa:train_014100:0::focus1",
        "tgvf_v4_teacher_50k:chartqa:train_015867:0::focus1",
        "Which year has the tallest bar in the chart?",
        "the tallest bar together with its year label and top edge",
        "2018",
        "2017",
        "The highest late-year bar is labeled 2018.",
        "The highest blue bar is labeled 2017.",
    ),
    (
        "motor-vehicle-body-blue-vs-red",
        "tgvf_v4_teacher_50k:visual_genome:2322845:0::focus1",
        "tgvf_v4_teacher_50k:visual_genome:2393223:0::focus1",
        "What is the dominant color of the foreground two-wheeled motor vehicle's main painted body panels?",
        "main painted body panels of the foreground two-wheeled motor vehicle, excluding wheels, seat, and trim",
        "blue",
        "red",
        "The parked scooter's visible painted body panels are dark blue.",
        "The motorcycle's tank and side body panels are predominantly red.",
    ),
    (
        "foreground-clothing-white-vs-black",
        "tgvf_v4_teacher_50k:visual_genome:2410492:0::focus1",
        "tgvf_v4_teacher_50k:visual_genome:2320766:0::focus1",
        "What is the dominant color of the largest foreground person's outer clothing?",
        "largest foreground person's torso and leg clothing surfaces, excluding skin and accessories",
        "white",
        "black",
        "The foreground baseball player's uniform shirt and pants are white.",
        "The largest foreground beach figure wears predominantly black outer clothing.",
    ),
    (
        "subject-accessory-bow-vs-bridle",
        "tgvf_v4_teacher_50k:visual_genome:2325101:3::focus1",
        "tgvf_v4_teacher_50k:visual_genome:2383628:0::focus1",
        "What prominent accessory is attached around the central subject's neck or head?",
        "prominent accessory attached around the central subject's neck or head",
        "bow",
        "bridle",
        "A red bow is visibly tied around the teddy bear's neck.",
        "A bridle or halter strap is visibly wrapped around the horse's head.",
    ),
    (
        "printed-surface-blue-vs-white",
        "tgvf_v4_teacher_50k:textocr:616dbff1d697c919:0::focus1",
        "tgvf_v4_teacher_50k:textvqa:ae31a3dfe7ef32a5:0::focus1",
        "What is the dominant color of the broad rectangular surface carrying most of the visible printed content?",
        "broad rectangular surface carrying most of the visible printed content",
        "blue",
        "white",
        "Most printed text lies on the broad dark-blue cover surface.",
        "Most printed content lies on the broad white workbook pages.",
    ),
    (
        "document-surface-white-vs-gray",
        "tgvf_v4_teacher_50k:docvqa:sthm0227_1:0::focus1",
        "tgvf_v4_teacher_50k:textocr:4f0d3773f7edbe0e:0::focus1",
        "What is the dominant color of the large central paper or cover surface?",
        "large central paper or cover surface inside its outer boundary",
        "white",
        "gray",
        "The handwritten note is on a predominantly white sheet.",
        "The central book-cover field is predominantly gray, excluding the blue outer background.",
    ),
)


TARGET_PRESENCE_AUDIT = (
    (
        "baseball-pants-not-locomotive",
        "tgvf_v4_teacher_50k:visual_genome:2410492:0::focus1",
        "large fabric area of both pant legs with nearby belt and shoes",
        "side panels and front face of a locomotive body with surrounding trim",
        "The baseball player's white pant legs are clearly visible.",
        "No train or locomotive appears anywhere in the baseball-field image.",
    ),
    (
        "train-body-not-polar-bear",
        "tgvf_v4_teacher_50k:visual_genome:2380160:0::focus1",
        "side panels and front face of the locomotive body with surrounding trim",
        "polar bear figure standing on top of a decorated cake",
        "The locomotive body and trim are clearly visible beside the platform.",
        "No cake or polar-bear figure appears in the railway image.",
    ),
    (
        "scooter-body-not-cake-icing",
        "tgvf_v4_teacher_50k:visual_genome:2322845:0::focus1",
        "painted side and rear body panels of the scooter with nearby seat and wheel context",
        "oval patch on top of a cake surrounded by white icing",
        "The parked scooter and its body panels are visible beside the fence.",
        "No cake or icing appears in the outdoor scooter image.",
    ),
    (
        "cake-bears-not-hydrant",
        "tgvf_v4_teacher_50k:visual_genome:2372630:1::focus1",
        "entire top of the cake containing all polar bear figures and nearby decorations",
        "painted metal body of a street fire hydrant",
        "Two polar-bear figures are visibly placed on the cake.",
        "No street or fire hydrant appears in the cake image.",
    ),
    (
        "kitchen-cabinets-not-horse",
        "tgvf_v4_teacher_50k:visual_genome:2389036:0::focus1",
        "cabinet door fronts and drawer faces with their painted surfaces and edges",
        "horse face and muzzle with straps wrapping around its head",
        "The kitchen cabinets and drawers are clearly visible.",
        "No horse appears in the indoor kitchen image.",
    ),
    (
        "note-salutation-not-motorcycle",
        "tgvf_v4_teacher_50k:docvqa:sthm0227_1:0::focus1",
        "top handwritten salutation line with the underlined name and nearby margin",
        "painted motorcycle fuel tank with colored stripe graphics",
        "The underlined handwritten salutation is visible near the page top.",
        "No motorcycle or fuel tank appears on the handwritten page.",
    ),
    (
        "hydrant-not-wine-label",
        "tgvf_v4_teacher_50k:visual_genome:2381146:0::focus1",
        "painted metal surface across the hydrant cap and side with surrounding sidewalk",
        "printed front label on an olive-green wine bottle",
        "The yellow fire hydrant is clearly visible on the sidewalk.",
        "No wine bottle or bottle label appears in the hydrant scene.",
    ),
    (
        "horse-bridle-not-barcode",
        "tgvf_v4_teacher_50k:visual_genome:2383628:0::focus1",
        "horse face, muzzle, and straps wrapping around the head",
        "rectangular printed barcode block near the top-right corner",
        "The horse's head and surrounding bridle straps are visible.",
        "No printed barcode appears in the horse-and-people scene.",
    ),
    (
        "bottle-body-not-scooter-box",
        "tgvf_v4_teacher_50k:textvqa:8263a74635bddc23:0::focus1",
        "main bottle body glass surface with highlights and edges away from the white label",
        "box-shaped storage compartment attached behind a scooter seat",
        "The green glass bottle body is clearly visible.",
        "No scooter or rear storage box appears in the bottle image.",
    ),
    (
        "kite-not-refrigerator",
        "tgvf_v4_teacher_50k:visual_genome:2320766:0::focus1",
        "broad fabric surface of the large foreground kite with its printed design visible",
        "large stainless-steel refrigerator doors and handles",
        "The large black kite is clearly visible against the sky.",
        "No refrigerator or indoor kitchen appliance appears in the beach image.",
    ),
    (
        "corner-flag-not-dragon",
        "tgvf_v4_teacher_50k:textocr:7887cb5d9895ddd6:0::focus1",
        "foreground corner flag fabric and pole area with visible colored cloth",
        "large membranous dragon wings spread across a book cover",
        "The red corner flag and pole are visible in the soccer scene.",
        "No dragon or illustrated dragon wings appear in the soccer image.",
    ),
    (
        "coin-not-teddy-bow",
        "tgvf_v4_teacher_50k:textocr:e08ccd92443c5924:2::focus1",
        "outer rim and full circular boundary of the coin against the rock",
        "red fabric bow tied around a teddy bear's neck",
        "The round coin is clearly visible on the rock.",
        "No teddy bear or red neck bow appears in the rock-and-coin image.",
    ),
    (
        "meal-plate-not-locomotive",
        "tgvf_v4_teacher_50k:visual_genome:2373606:0::focus1",
        "broad rim of the plate surrounding the food, with visible surface color",
        "front face and side panels of a railway locomotive",
        "The blue plate rim is visible around the meal.",
        "No railway locomotive appears in the close meal image.",
    ),
    (
        "runner-kite-not-refrigerator",
        "tgvf_v4_teacher_50k:visual_genome:2416706:0::focus1",
        "small flying kite canopy against the sky, with its visible colored panels",
        "large stainless-steel refrigerator doors and handles",
        "A blue kite canopy is visible above the runners.",
        "No refrigerator or kitchen appliance appears in the outdoor running scene.",
    ),
    (
        "kitchen-floor-not-polar-bear",
        "tgvf_v4_teacher_50k:visual_genome:2366990:0::focus1",
        "floor tile surfaces along the lower left walkway with grout lines preserved",
        "polar bear figure standing on top of a decorated cake",
        "The red-brown tiled floor is visible along the kitchen walkway.",
        "No cake or polar-bear figure appears in the commercial kitchen.",
    ),
    (
        "teddy-clothing-not-motorcycle",
        "tgvf_v4_teacher_50k:visual_genome:2325101:0::focus1",
        "central fabric covering the bear's torso and legs with surrounding fur edges",
        "painted motorcycle fuel tank with colored stripe graphics",
        "The teddy bear's red clothing is clearly visible.",
        "No motorcycle or fuel tank appears in the teddy-bear image.",
    ),
    (
        "barcode-not-horse",
        "tgvf_v4_teacher_50k:textocr:616dbff1d697c919:2::focus1",
        "top-right barcode block with its full outer border and nearby white margin",
        "horse face and muzzle with straps wrapping around its head",
        "A rectangular barcode block is visible at the cover's top right.",
        "No horse appears on or around the printed cover.",
    ),
    (
        "building-shutters-not-dragon",
        "tgvf_v4_teacher_50k:visual_genome:3000:0::focus1",
        "painted shutter slats and surrounding frame on the right-side building facade",
        "large membranous dragon wings spread across a book cover",
        "Green shutter slats are visible on the right-side building.",
        "No dragon or illustrated dragon wings appear in the street scene.",
    ),
    (
        "book-background-not-hydrant",
        "tgvf_v4_teacher_50k:textocr:4f0d3773f7edbe0e:0::focus1",
        "outer border around the book cover with the visible surrounding surface color",
        "painted metal body of a street fire hydrant",
        "The bright-blue surface surrounding the book is visible.",
        "No street fire hydrant appears in the book-cover image.",
    ),
    (
        "green-tie-not-cake",
        "tgvf_v4_teacher_50k:visual_genome:2323361:0::focus1",
        "central necktie fabric below the collar with its full visible color area",
        "oval patch on top of a cake surrounded by white icing",
        "The man's light-green necktie is visibly centered below his collar.",
        "No cake or icing appears in the portrait scene.",
    ),
    (
        "scholar-row-not-scooter",
        "tgvf_v4_teacher_50k:docvqa:lpyc0227_2:0::focus1",
        "left table row with the scholar name and adjacent rank text",
        "painted side and rear body panels of a parked scooter",
        "The scholar-name row is visible in the document table.",
        "No scooter appears on the document page.",
    ),
    (
        "helicopter-body-not-cake",
        "tgvf_v4_teacher_50k:textocr:a7334a81098d4ab8:0::focus1",
        "broad side fuselage panel around the large FIREHAWK lettering and adjacent paint bands",
        "polar bear figure standing on top of a decorated cake",
        "The blue FIREHAWK helicopter fuselage panel is clearly visible.",
        "No cake or polar-bear figure appears in the helicopter image.",
    ),
    (
        "chart-tallest-not-horse",
        "tgvf_v4_teacher_50k:chartqa:train_017485:0::focus1",
        "the tallest blue bar with its year label and nearby value label",
        "horse face and muzzle with straps wrapping around its head",
        "The chart contains a tallest blue bar labeled 2019.",
        "No horse appears in the bar-chart image.",
    ),
    (
        "tomato-not-locomotive",
        "tgvf_v4_teacher_50k:visual_genome:2372542:0::focus1",
        "close-up of the lower-left tomato wedge and its glossy skin surface",
        "front face and side panels of a railway locomotive",
        "The red tomato wedge is visible at the lower-left of the plate.",
        "No railway locomotive appears in the food image.",
    ),
    (
        "ruler-not-horse",
        "tgvf_v4_teacher_50k:textvqa:1fe2bcaa49b8d54e:0::focus1",
        "right-side ruler surface with wood grain and printed measurement marks",
        "horse face and muzzle with straps wrapping around its head",
        "The wooden ruler and its measurement marks are visible at right.",
        "No horse appears in the tabletop image.",
    ),
    (
        "envelope-return-address-not-motorcycle",
        "tgvf_v4_teacher_50k:docvqa:hqjm0081_2:0::focus1",
        "printed return address block at lower left with the bold top name line",
        "painted motorcycle fuel tank with colored stripe graphics",
        "The printed return-address block is visible at lower left.",
        "No motorcycle appears on the envelope.",
    ),
    (
        "committee-title-not-train",
        "tgvf_v4_teacher_50k:docvqa:grmp0227_13:0::focus1",
        "large centered committee heading lines near the top with surrounding blank margin",
        "front face and side panels of a railway locomotive",
        "The large centered committee title is visible near the page top.",
        "No railway locomotive appears on the committee page.",
    ),
    (
        "workbook-binding-not-dragon",
        "tgvf_v4_teacher_50k:textvqa:ae31a3dfe7ef32a5:0::focus1",
        "close view of the spiral binding loops between the two workbook pages",
        "large membranous dragon wings spread across a book cover",
        "The white spiral-binding loops are visible between the workbook pages.",
        "No dragon or dragon-wing illustration appears in the workbook scene.",
    ),
    (
        "chart-share-not-teddy",
        "tgvf_v4_teacher_50k:chartqa:train_002420:0::focus1",
        "Dem/Lean Dem row bars with segment numbers and nearby category labels",
        "red fabric bow tied around a teddy bear's neck",
        "The Dem/Lean Dem stacked row and its segment values are visible.",
        "No teddy bear or neck bow appears in the chart.",
    ),
    (
        "motorcycle-tank-not-cake",
        "tgvf_v4_teacher_50k:visual_genome:2393223:0::focus1",
        "painted motorcycle fuel tank surface with visible stripe graphics and highlights",
        "oval patch on top of a cake surrounded by white icing",
        "The red motorcycle fuel tank and colored stripes are clearly visible.",
        "No cake or icing appears in the motorcycle scene.",
    ),
    (
        "dragon-wings-not-locomotive",
        "tgvf_v4_teacher_50k:textvqa:5bf9e10aeff884e4:1::focus1",
        "dragon illustration with both extended membranous wings and surrounding body",
        "front face and side panels of a railway locomotive",
        "The illustrated dragon's large spread wings are visible on the cover.",
        "No railway locomotive appears on the fantasy book cover.",
    ),
    (
        "kidzone-flyer-not-horse",
        "tgvf_v4_teacher_50k:textvqa:9296cda2a81e5f4c:0::focus1",
        "broad paper surface of the central Kidzone flyer with surrounding pins and logo",
        "horse face and muzzle with straps wrapping around its head",
        "The large white Kidzone flyer is visible at the center.",
        "No horse appears on the notice-board image.",
    ),
    (
        "clock-face-not-hydrant",
        "tgvf_v4_teacher_50k:textvqa:511fbb06f065b6e1:0::focus1",
        "large rectangular clock face and surrounding bezel at middle-right",
        "painted metal body of a street fire hydrant",
        "The large blue rectangular clock face is visible at middle-right.",
        "No street fire hydrant appears on the printed page.",
    ),
    (
        "box-stack-not-dragon",
        "tgvf_v4_teacher_50k:textvqa:80b9d8b00093851a:0::focus1",
        "entire vertical stack of cardboard boxes beside the white door",
        "large membranous dragon wings spread across a book cover",
        "The vertical stack of cardboard boxes is visible beside the door.",
        "No dragon or dragon-wing illustration appears in the room scene.",
    ),
    (
        "black-belt-not-train",
        "tgvf_v4_teacher_50k:textocr:1bd5b29c45a41ef0:0::focus1",
        "belt knot and wrapped fabric around the central person's waist",
        "front face and side panels of a railway locomotive",
        "The black tied belt is visible around the central person's waist.",
        "No railway locomotive appears in the uniformed-person image.",
    ),
    (
        "routine-checkbox-not-motorcycle",
        "tgvf_v4_teacher_50k:docvqa:npbb0079_11:0::focus1",
        "right-side urgency checkbox block with option labels and adjacent mark lines",
        "painted motorcycle fuel tank with colored stripe graphics",
        "The form's urgency checkbox block and Routine mark are visible.",
        "No motorcycle appears on the form page.",
    ),
)


def main() -> None:
    dataset = load_retained_representation_jsonl(
        SOURCE_PATH,
        expected_source_sha256=SOURCE_SHA256,
        warn_on_leakage=False,
    )
    if dataset.manifest.manifest_sha256 != RETAINED_MANIFEST_SHA256:
        raise RuntimeError("retained v4 clean-imend test manifest changed")
    samples = {sample.sample_id: sample for sample in dataset.samples}

    def source_fields(sample_id: str) -> tuple[str, str]:
        sample = samples[sample_id]
        return sample.content_sha256, sha256(
            Path(sample.image).read_bytes()
        ).hexdigest()

    cross = []
    for (
        pair_id,
        sample_a_id,
        sample_b_id,
        question,
        target,
        value_a,
        value_b,
        rationale_a,
        rationale_b,
    ) in CROSS_IMAGE_AUDIT:
        content_a, image_a = source_fields(sample_a_id)
        content_b, image_b = source_fields(sample_b_id)
        cross.append(
            {
                "schema_version": "qwen3_cross_image_probe_v1",
                "pair_id": pair_id,
                "source_sample_a_id": sample_a_id,
                "source_sample_a_content_sha256": content_a,
                "source_image_a_sha256": image_a,
                "source_sample_b_id": sample_b_id,
                "source_sample_b_content_sha256": content_b,
                "source_image_b_sha256": image_b,
                "question": question,
                "target": target,
                "expected_value_a": value_a,
                "expected_value_b": value_b,
                "audit_rationale_a": rationale_a,
                "audit_rationale_b": rationale_b,
                "pair_audit_identity": "manual-visual-audit-20260721-v1",
            }
        )

    presence = []
    for (
        pair_id,
        sample_id,
        positive_target,
        negative_target,
        positive_rationale,
        negative_rationale,
    ) in TARGET_PRESENCE_AUDIT:
        content, image = source_fields(sample_id)
        presence.append(
            {
                "schema_version": "qwen3_target_presence_probe_v1",
                "pair_id": pair_id,
                "source_sample_id": sample_id,
                "source_sample_content_sha256": content,
                "source_image_sha256": image,
                "positive_target": positive_target,
                "negative_target": negative_target,
                "positive_audit_rationale": positive_rationale,
                "negative_audit_rationale": negative_rationale,
                "pair_audit_identity": "manual-visual-audit-20260721-v1",
                "question": TARGET_PRESENCE_QUESTION,
                "present_value": "PRESENT",
                "not_present_value": "NOT_PRESENT",
            }
        )

    payload = {
        "schema_version": "qwen3_grounding_manifest_v1",
        "identity": "qwen3-v4-clean-imend-audited-grounding-v1",
        "source_data_manifest_sha256": RETAINED_MANIFEST_SHA256,
        "cross_image_probes": cross,
        "target_presence_probes": presence,
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    if "--stdout" in sys.argv[1:]:
        print(encoded, end="")
        return
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists() and OUTPUT_PATH.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"refusing to overwrite changed manifest: {OUTPUT_PATH}")
    OUTPUT_PATH.write_text(encoded, encoding="utf-8")
    print(f"{sha256(encoded.encode()).hexdigest()}  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
