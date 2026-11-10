import hmac
import os
from urllib.parse import urlsplit


def origin_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment or any(c.isspace() for c in value)):
        raise ValueError("Origins must be complete HTTP or HTTPS origins without a path")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Remote origins require HTTPS")
    return parsed.scheme + "://" + parsed.netloc


class AccessPolicy:
    def __init__(self, host="127.0.0.1", token=None, origins=None, public_url=None):
        self.token = os.environ.get("UNIBRAIN_ACCESS_TOKEN", "") if token is None else token
        values = os.environ.get("UNIBRAIN_ALLOWED_ORIGINS", "").split(",") if origins is None else origins
        self.origins = {origin_url(value.strip()) for value in values if value.strip()}
        self.public_url = public_url or os.environ.get("UNIBRAIN_PUBLIC_URL", "")
        if self.public_url:
            self.public_url = origin_url(self.public_url)
        if self.token and (len(self.token) < 32 or not self.token.isascii() or any(c.isspace() for c in self.token)):
            raise ValueError("Use an access token with at least 32 ASCII characters and no spaces")
        if (self.origins or host not in {"127.0.0.1", "localhost", "::1"} or self.public_url) and not self.token:
            raise ValueError("Remote access requires UNIBRAIN_ACCESS_TOKEN")

    def hosts(self, port):
        values = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        if self.public_url:
            values.add(urlsplit(self.public_url).netloc)
        return values

    def accepts_origin(self, origin, port):
        local = {f"http://127.0.0.1:{port}", f"http://localhost:{port}", f"http://[::1]:{port}"}
        return not origin or origin in self.origins | local | ({self.public_url} if self.public_url else set())

    def authenticated(self, header):
        return not self.token or hmac.compare_digest((header or "").encode(), ("Bearer " + self.token).encode())
