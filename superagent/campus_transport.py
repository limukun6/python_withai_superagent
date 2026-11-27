"""Per-client explicit campus address, retaining the HTTPS hostname and verification."""
from urllib.parse import urlsplit

from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPSConnection
from urllib3.connectionpool import HTTPSConnectionPool

HOST = 'aigc-api.hkust-gz.edu.cn'
CAMPUS_IP = '10.121.10.250'
ENDPOINT = 'https://' + HOST + '/api/v2/chat/completions'


def validate_address(endpoint, address):
    if address is not None and (address != CAMPUS_IP or endpoint != ENDPOINT):
        raise ValueError('Explicit campus IP is valid only for the documented campus endpoint')


class CampusConnection(HTTPSConnection):
    def _new_conn(self):
        if self.host != HOST or self.port != 443 or self.proxy is not None:
            raise ValueError('Explicit campus connection requires the direct official HTTPS origin')
        # This connection instance owns the address. No global socket/DNS changes.
        # Restore it before HTTPSConnection.connect reads the hostname for TLS/SNI.
        original = self._dns_host
        try:
            self._dns_host = CAMPUS_IP
            return super()._new_conn()
        finally:
            self._dns_host = original


class CampusPool(HTTPSConnectionPool):
    ConnectionCls = CampusConnection


class CampusAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = dict(self.poolmanager.pool_classes_by_scheme, https=CampusPool)


def configure_session(session, endpoint, address):
    validate_address(endpoint, address)
    if address is not None:
        session.trust_env = False
        session.mount('https://' + HOST + '/', CampusAdapter(max_retries=0))
