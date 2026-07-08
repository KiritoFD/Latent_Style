import socket, urllib.request
def test(host, port):
    try:
        s = socket.create_connection((host, port), timeout=10)
        print(host, port, "OK"); s.close()
    except Exception as e:
        print(host, port, "FAIL", repr(e))
test("github.com", 443)
test("huggingface.co", 443)
for url in ["https://huggingface.co", "https://www.google.com"]:
    try:
        r = urllib.request.urlopen(url, timeout=10)
        print("GET", url, r.status)
    except Exception as e:
        print("GET", url, "FAIL", repr(e))
