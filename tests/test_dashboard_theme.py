"""Check the intended theme pairs; this is not a claim of whole-app WCAG conformance."""

from pathlib import Path
import tomllib
import unittest


def luminance(hex_color: str) -> float:
    channels = [int(hex_color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [channel / 12.92 if channel <= .04045 else ((channel + .055) / 1.055) ** 2.4
              for channel in channels]
    return sum(channel * weight for channel, weight in zip(linear, [.2126, .7152, .0722]))


class DashboardThemeTest(unittest.TestCase):
    def test_intended_text_and_button_color_pairs_have_aa_contrast(self):
        path = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
        theme = tomllib.loads(path.read_text(encoding="utf-8"))["theme"]
        pairs = [(theme["textColor"], theme["backgroundColor"]),
                 (theme["textColor"], theme["secondaryBackgroundColor"]),
                 (theme["grayColor"], theme["backgroundColor"]),
                 (theme["orangeColor"], theme["backgroundColor"]),
                 ("#FFFFFF", theme["primaryColor"]),
                 (theme["sidebar"]["textColor"], theme["sidebar"]["backgroundColor"])]
        for foreground, background in pairs:
            high, low = sorted((luminance(foreground), luminance(background)), reverse=True)
            self.assertGreaterEqual((high + .05) / (low + .05), 4.5, (foreground, background))
        self.assertNotIn("fonts.googleapis", theme["font"])


if __name__ == "__main__":
    unittest.main()
