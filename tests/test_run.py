from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import run


class RunEntrypointTest(TestCase):
    @patch("run.subprocess.call", return_value=0)
    def test_webui_launch_is_lazy_and_uses_repository_root(self, call):
        self.assertEqual(run.main(["--webui"]), 0)
        command = call.call_args.args[0]
        self.assertEqual(command[:3], [run.sys.executable, "-m", "streamlit"])
        self.assertEqual(call.call_args.kwargs["cwd"], Path(run.__file__).resolve().parent)

    @patch("scripts.view_nifti_3d.main")
    def test_gui_dispatches_remaining_arguments(self, gui_main):
        self.assertEqual(run.main(["--gui", "--image", "case.nii"]), 0)
        gui_main.assert_called_once_with(["--image", "case.nii"])


if __name__ == "__main__":
    import unittest

    unittest.main()
