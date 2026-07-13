# Binary correctness label used across the feature-extraction and Qwen scripts.
# "otherwise" (i.e. no "Action Correctness" attribute set) is intentionally
# absent here and gets filtered out wherever this map is used as a membership test.
LABEL_MAP = {
    "Correct Action": 0,
    "Wrong Action, corrected by instructor verbally": 1,
    "Wrong Action, corrected by student": 1,
    "Wrong Action, not corrected": 1,
}
