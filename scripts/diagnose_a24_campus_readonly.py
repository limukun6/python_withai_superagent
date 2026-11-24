"""No credential or HTTP: diagnose DNS and the user-supplied campus address once."""
import argparse
import socket
import ssl
import time
from pathlib import Path
from scripts.a_batch_common import save


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);results=[]
    save(a.output/'protocol.json',dict(batch='A24',hostname='aigc-api.hkust-gz.edu.cn',
        known_user_provided_ip='10.121.10.250',chat_requests=0,http_requests=0,credentials_read=False,
        socket_timeout_seconds=5,network_configuration_modified=False))
    start=time.monotonic()
    try:
        addresses=socket.getaddrinfo('aigc-api.hkust-gz.edu.cn',443,type=socket.SOCK_STREAM)
        results.append(dict(check='DNS',status='resolved',addresses=sorted({x[4][0] for x in addresses})))
    except Exception as error:results.append(dict(check='DNS',status='failed',error_type=type(error).__name__,error=str(error)))
    try:
        with socket.create_connection(('10.121.10.250',443),timeout=5) as connection:
            with ssl.create_default_context().wrap_socket(connection,server_hostname='aigc-api.hkust-gz.edu.cn') as tls:
                results.append(dict(check='known_IP_TLS',status='verified',tls_version=tls.version()))
    except Exception as error:results.append(dict(check='known_IP_TLS',status='failed',error_type=type(error).__name__,error=str(error)))
    save(a.output/'results.json',results)
    save(a.output/'completion.json',dict(status='diagnosed',chat_requests=0,classifier_fits=0,
        elapsed_seconds=time.monotonic()-start,network_configuration_modified=False))
    print(__import__('json').dumps(results))
