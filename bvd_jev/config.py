"""Default model selection for the BVD adapter."""

import os


DEFAULT_MODEL = os.environ.get(
    "VIDEOJEV_MODEL", "ZhuYuhan/VideoJev-4B")
