"""Retired experiments must not remain available through the project CLI."""

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'python/main.py'


class CommandRetirementTests(unittest.TestCase):
    def test_help_exposes_current_workflow_only(self):
        result = subprocess.run([sys.executable, '-B', str(ENTRY), '--help'],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ('predict', 'cad', 'mesh', 'train'):
            self.assertIn(command, result.stdout)
        self.assertNotIn('qem', result.stdout.lower())

    def test_retired_command_is_rejected_before_loading_dependencies(self):
        result = subprocess.run([sys.executable, '-B', str(ENTRY), 'qem'],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)
        self.assertNotIn('ModuleNotFoundError', result.stderr)


if __name__ == '__main__':
    unittest.main()
