"""Resolution choice is explicit, narrowly scoped, reversible, and never changes the HTTPS host."""
import socket
import unittest
from unittest.mock import patch,Mock

from scripts.campus_direct import campus_resolution,HOST,CAMPUS_IP


class CampusResolutionTests(unittest.TestCase):
    def test_only_campus_hostname_rewritten_and_original_restored(self):
        original=Mock(return_value=['addresses'])
        with patch.object(socket,'getaddrinfo',original):
            with self.assertRaisesRegex(ValueError,'controlled'):
                with campus_resolution(CAMPUS_IP):
                    socket.getaddrinfo(HOST,443,type=socket.SOCK_STREAM)
                    socket.getaddrinfo('example.org',443)
                    raise ValueError('controlled')
            self.assertIs(socket.getaddrinfo,original)
        self.assertEqual(original.call_args_list[0].args,(CAMPUS_IP,443))
        self.assertEqual(original.call_args_list[1].args,('example.org',443))

    def test_no_arbitrary_endpoint_redirection(self):
        with self.assertRaises(ValueError):
            with campus_resolution('127.0.0.1'):self.fail('Unexpected resolver activation')


if __name__=='__main__':unittest.main()
