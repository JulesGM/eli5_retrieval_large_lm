import time
from pyngrok import ngrok

ssh_tunnel = ngrok.connect(8888, "http")
print(ssh_tunnel.public_url, flush=True)

while True:
    time.sleep(10)
