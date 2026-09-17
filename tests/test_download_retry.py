from email.message import Message
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from download_corpus import retry_delay


class DownloadRetryTests(unittest.TestCase):
    def test_throttling_honors_seconds_and_http_date(self):
        headers = Message()
        headers['Retry-After'] = '120'
        error = HTTPError('https://example.test', 429, 'limited', headers, None)
        self.assertEqual(retry_delay(error, 0), 120)
        headers.replace_header('Retry-After', 'Thu, 01 Jan 1970 00:03:00 GMT')
        with patch('download_corpus.time.time', return_value=60):
            self.assertEqual(retry_delay(error, 0), 120)
        headers.replace_header('Retry-After', 'invalid')
        self.assertEqual(retry_delay(error, 0), 60)

    def test_permanent_http_failure_is_not_retried(self):
        error = HTTPError('https://example.test', 404, 'missing', {}, None)
        with self.assertRaises(HTTPError):
            retry_delay(error, 0)


if __name__ == '__main__':
    unittest.main()
