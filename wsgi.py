"""
WSGI entry point for production deployments.
Compatible with Gunicorn, Waitress, and any WSGI server.
"""

import os
import threading
import webbrowser
from app import app, find_available_port

if __name__ == "__main__":
    from waitress import serve
    preferred_port = int(os.environ.get("PORT", 5001))
    port = find_available_port(preferred_port)
    host = os.environ.get("HOST", "0.0.0.0")
    url = f"http://localhost:{port}"

    print(f" Serving InsurePredict AI with Waitress (Production WSGI) on {url} (and http://127.0.0.1:{port})")
    
    # Auto-open browser
    if os.environ.get("NO_BROWSER", "").lower() not in ["1", "true"]:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    serve(app, host=host, port=port)
