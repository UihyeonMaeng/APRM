"""Fixed anatomical prior used by the APRM lung CT release."""

ANATOMY_LABELS = {
    "background": 0,
    "RUL": 1,
    "RML": 2,
    "RLL": 3,
    "LUL": 4,
    "LLL": 5,
    "CENTRAL": 6,
}
NUM_ANATOMY_CLASSES = 7
LOBE_FILES = {
    "RUL": "lung_upper_lobe_right",
    "RML": "lung_middle_lobe_right",
    "RLL": "lung_lower_lobe_right",
    "LUL": "lung_upper_lobe_left",
    "LLL": "lung_lower_lobe_left",
}
CENTRAL_ORGANS = (
    "heart",
    "aorta",
    "trachea",
    "esophagus",
    "pulmonary_vein",
    "superior_vena_cava",
    "inferior_vena_cava",
)
REQUIRED_ROIS = (*LOBE_FILES.values(), *CENTRAL_ORGANS)
