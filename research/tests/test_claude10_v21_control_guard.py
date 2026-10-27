"""Recovery authorization must prevent the old collector overwriting root fees."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
control = importlib.import_module('review10_control')


class RecoveryGuardTest(unittest.TestCase):
    def test_recovery_authorization_blocks_old_collector_before_query_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            recovery = package / 'recovery_v21'
            recovery.mkdir()
            (recovery / 'AUTHORIZATION.json').write_text('{}\n')
            with patch.object(control, 'PACKAGE', package), \
                 patch.object(sys, 'argv', ['review10_control.py', '--collect']), \
                 patch.object(control.collect, 'load') as load, \
                 patch.object(control, 'recover_stage') as recover, \
                 patch.object(control, 'atomic_json') as write:
                load.side_effect = AssertionError('Legacy collector accessed old control before recovery guard')
                with self.assertRaisesRegex(RuntimeError, 'recovery_v21'):
                    control.main()
                load.assert_not_called()
                recover.assert_not_called()
                write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
