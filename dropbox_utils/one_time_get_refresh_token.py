# one_time_get_refresh_token.py
import http.server, socketserver, webbrowser, urllib.parse, requests, os
from utils.config_loader import load_config

config = load_config()
dbx = config['dropbox']
APP_KEY = dbx['app_key']
APP_SECRET = dbx['app_secret']
REDIRECT_URI = dbx['redirect_uri']
SCOPES = " ".join(dbx['scopes'])
PORT = dbx['port']

auth_url = (
    "https://www.dropbox.com/oauth2/authorize?" +
    urllib.parse.urlencode({
        "response_type": "code",
        "client_id": APP_KEY,
        "redirect_uri": REDIRECT_URI,
        "token_access_type": "offline",
        "scope": SCOPES
    })
)

code_holder = {}
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        q = urllib.parse.urlparse(self.path)
        if q.path == "/callback":
            params = urllib.parse.parse_qs(q.query)
            code_holder["code"] = params.get("code", [None])[0]
            self.send_response(200); self.end_headers()
            self.wfile.write(b"Dropbox authorization complete. You can close this tab.")
        else:
            self.send_response(404); self.end_headers()

with socketserver.TCPServer(("localhost", PORT), Handler) as httpd:
    print("Opening browser for Dropbox consent…")
    webbrowser.open(auth_url)
    httpd.handle_request()

code = code_holder.get("code")
if not code: raise SystemExit("No auth code received.")

data = {
    "code": code,
    "grant_type": "authorization_code",
    "redirect_uri": REDIRECT_URI,
    "client_id": APP_KEY,
}
if APP_SECRET:  # confidential client flow
    data["client_secret"] = APP_SECRET

r = requests.post("https://api.dropboxapi.com/oauth2/token", data=data)
r.raise_for_status()
tokens = r.json()
print("\nREFRESH TOKEN (store securely):", tokens.get("refresh_token"))
print("ACCESS TOKEN (short-lived):", tokens.get("access_token"))
print("EXPIRES IN (sec):", tokens.get("expires_in"))

