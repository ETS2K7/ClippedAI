import modal
import socket

app = modal.App("udp-test")

@app.function()
def test_udp():
    try:
        # Try to send a UDP packet to 1.1.1.1:53 and get a response
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0)
        # Send a valid DNS query for google.com
        dns_query = b'\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x06google\x03com\x00\x00\x01\x00\x01'
        sock.sendto(dns_query, ('1.1.1.1', 53))
        data, addr = sock.recvfrom(1024)
        print("UDP port 53 works.")
        
        # Now try a non-DNS UDP port (like Cloudflare WARP port 2408)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0)
        sock.sendto(b'ping', ('162.159.192.1', 2408))
        data, addr = sock.recvfrom(1024)
        print("UDP port 2408 works.")
    except Exception as e:
        print(f"UDP failed: {e}")

if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            test_udp.remote()
