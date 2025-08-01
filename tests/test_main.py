import unittest
import subprocess

class TestMainEntry(unittest.TestCase):
    def test_main_runs(self):
        """Check if __main__.py executes without error."""
        result = subprocess.run(['python', '__main__.py'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg="Main script failed to run")

    def test_main_output(self):
        """Optional: Verify expected output string if any."""
        result = subprocess.run(['python', '__main__.py'], capture_output=True, text=True)
        self.assertIn("QuasiStarSim initialized", result.stdout)

if __name__ == "__main__":
    unittest.main()