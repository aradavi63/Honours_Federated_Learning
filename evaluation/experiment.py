"""Save the settings used for a training or attack run."""

import json
from pathlib import Path

from evaluation.paths import portable_options


def write_configuration(output, experiment, scheme, config, options, test_count):
    payload = {
        "experiment": experiment,
        "scheme": scheme,
        "training": portable_options(config),
        "options": portable_options(vars(options)),
        "evaluation": {"split": "official_test_split", "records": test_count},
    }
    (Path(output) / "configuration.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
