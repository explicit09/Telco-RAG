from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_teleqna import requested_release


class ReleaseProtocolTests(unittest.TestCase):
    def test_preserves_requested_release(self):
        for release in ('14','16','17','18','19'):
            self.assertEqual(requested_release(f'Example question? [3GPP Release {release}]'),release)

    def test_missing_or_ambiguous_release_fails_closed(self):
        for text in ('Question without a release', 'Compare 3GPP Release 17 and 3GPP Release 18'):
            with self.assertRaises(ValueError):
                requested_release(text)
