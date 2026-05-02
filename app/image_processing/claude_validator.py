"""
Claude Haiku-based validator for spray coverage analysis.

Replaces Moondream with Anthropic's Claude Haiku 4.5 via API.
Better visual reasoning and consistency than the local Moondream model,
at a low cost (~$0.002 per image).

Acts as a backup/second-opinion to OpenCV detection:
- If OpenCV and Claude disagree by more than VALIDATION_THRESHOLD (20%),
  use Claude's estimate (assumes OpenCV failed).
- If they agree, average the two values for refined accuracy.
"""
import base64
import logging
import os
import re
from typing import Tuple

from anthropic import Anthropic

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = (
    os.getenv("ANTHROPIC_API_KEY")
    or os.getenv("CLAUDE_API_KEY")
    or os.getenv("CLAUDE")  # Railway tiene la variable nombrada así
)
MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
VALIDATION_THRESHOLD = float(os.getenv("CLAUDE_THRESHOLD", "20.0"))

PROMPT = """This image shows a plant leaf under UV light. Bright cyan, blue, or white spots are spray droplets fluorescing under UV; the dark purple/magenta area is the leaf surface itself.

Estimate what percentage of the leaf surface is covered by spray droplets.

IMPORTANT:
- Be precise. If you see almost no droplets, say so (e.g., 2%).
- If most of the leaf is covered, say a high number (e.g., 90%).
- Don't default to 50% when uncertain — give your best estimate.
- Distinguish spray droplets (cyan/blue/white) from purple leaf reflections.

Respond in EXACTLY this format:
COVERAGE: X%
REASONING: brief one-sentence explanation"""


class ClaudeValidator:
    """Wraps Claude Haiku API + result comparison logic."""

    _client = None

    @classmethod
    def _get_client(cls):
        if cls._client is None:
            cls._client = Anthropic(api_key=ANTHROPIC_API_KEY)
        return cls._client

    @staticmethod
    def is_available() -> bool:
        """Check if API key is configured."""
        return bool(ANTHROPIC_API_KEY)

    @staticmethod
    def estimate_coverage(image_bytes: bytes, media_type: str = "image/jpeg") -> Tuple[float, str]:
        """
        Ask Claude Haiku to estimate spray coverage percentage.

        Returns: (coverage_percentage, raw_response)
                 coverage_percentage = -1.0 if API failed/unavailable
        """
        if not ClaudeValidator.is_available():
            logger.warning("Claude not available: no ANTHROPIC_API_KEY set")
            return -1.0, "NO_API_KEY"

        try:
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            client = ClaudeValidator._get_client()

            response = client.messages.create(
                model=MODEL,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": PROMPT}
                    ]
                }]
            )

            raw = response.content[0].text if response.content else ""
            coverage = ClaudeValidator._parse_percentage(raw)

            logger.info(f"Claude raw: '{raw[:200]}' → parsed: {coverage}%")
            return coverage, raw

        except Exception as e:
            logger.warning(f"Claude API error: {e}")
            return -1.0, f"ERROR: {e}"

    @staticmethod
    def _parse_percentage(text: str) -> float:
        """
        Extract numeric percentage from Claude's response.

        Priority:
          1. "COVERAGE: X%" format (preferred, structured)
          2. Any "X%" in response
          3. Bare number 0-100
        """
        if not text:
            return -1.0

        # Priority 1: structured COVERAGE: X%
        match = re.search(
            r"COVERAGE:\s*(\d{1,3}(?:\.\d+)?)\s*%?",
            text,
            re.IGNORECASE,
        )
        if match:
            v = float(match.group(1))
            if 0 <= v <= 100:
                return round(v, 2)

        # Priority 2: any X% in text
        match = re.search(
            r"(\d{1,3}(?:\.\d+)?)\s*(?:-\s*(\d{1,3}(?:\.\d+)?))?\s*%",
            text,
        )
        if match:
            v1 = float(match.group(1))
            v2 = match.group(2)
            v = (v1 + float(v2)) / 2 if v2 else v1
            if 0 <= v <= 100:
                return round(v, 2)

        # Priority 3: any 0-100 number
        match = re.search(r"\b(\d{1,3}(?:\.\d+)?)\b", text)
        if match:
            v = float(match.group(1))
            if 0 <= v <= 100:
                return round(v, 2)

        return -1.0

    @staticmethod
    def compare_results(
        coverage_opencv: float,
        coverage_claude: float,
        threshold: float = VALIDATION_THRESHOLD,
    ) -> Tuple[float, str, bool]:
        """
        Compare OpenCV vs Claude coverage estimates.

        Logic:
          - If Claude failed → use OpenCV (no validation)
          - If |diff| > threshold → use Claude (OpenCV likely failed)
          - If |diff| <= threshold → average both (refined estimate)

        Returns: (final_coverage, validation_flag, used_backup)
                 validation_flag ∈ {"opencv_only", "validated", "backup_used"}
        """
        if coverage_claude < 0:
            return coverage_opencv, "opencv_only", False

        diff = abs(coverage_opencv - coverage_claude)

        if diff > threshold:
            logger.info(
                f"Coverage disagreement: OpenCV={coverage_opencv}%, "
                f"Claude={coverage_claude}%, diff={diff:.1f}% > {threshold}% → "
                f"using Claude (backup)"
            )
            return coverage_claude, "backup_used", True

        averaged = round((coverage_opencv + coverage_claude) / 2, 2)
        logger.info(
            f"Coverage agreement: OpenCV={coverage_opencv}%, "
            f"Claude={coverage_claude}%, diff={diff:.1f}% <= {threshold}% → "
            f"averaged={averaged}%"
        )
        return averaged, "validated", False
