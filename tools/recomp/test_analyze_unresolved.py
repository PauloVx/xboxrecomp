import json
import tempfile
import unittest
from pathlib import Path

from tools.recomp.analyze_unresolved import find_section, main


class FindSectionTests(unittest.TestCase):
    def test_uses_virtual_ranges_from_xbe_metadata(self):
        sections = [
            {"name": ".text", "virtual_addr": "0x2000", "virtual_size": 0x100},
            {"name": "WMADEC", "virtual_addr": "0x3000", "virtual_size": 0x80},
            {"name": ".data", "virtual_addr": "0x4000", "virtual_size": 0x100},
        ]

        self.assertEqual(find_section(0x2000, sections), ".text")
        self.assertEqual(find_section(0x20FF, sections), ".text")
        self.assertIsNone(find_section(0x2100, sections))
        self.assertEqual(find_section(0x307F, sections), "WMADEC")
        self.assertIsNone(find_section(0x11000, sections))

    def test_cli_classifies_using_the_supplied_xbe_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analysis = root / "xbe-analysis.json"
            functions = root / "functions.json"
            unresolved = root / "unresolved.txt"
            output = root / "out"
            analysis.write_text(json.dumps({"sections": [
                {"name": ".text", "virtual_addr": "0x2000", "virtual_size": 0x100},
                {"name": ".rdata", "virtual_addr": "0x3000", "virtual_size": 0x80},
            ]}), encoding="utf-8")
            functions.write_text(json.dumps([
                {"start": "0x00002000", "end": "0x00002020", "name": "known"},
            ]), encoding="utf-8")
            unresolved.write_text("sub_00002010\nsub_00003008\nsub_00005000\n", encoding="utf-8")

            main(["--xbe-analysis", str(analysis), "--functions", str(functions),
                  "--unresolved", str(unresolved), "--output-dir", str(output)])

            results = json.loads((output / "missing_functions.json").read_text(encoding="utf-8"))
            self.assertEqual([item["type"] for item in results],
                             ["mid_function", "data_section", "unknown"])


if __name__ == "__main__":
    unittest.main()
