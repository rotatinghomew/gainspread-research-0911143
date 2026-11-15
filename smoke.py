import json
import tempfile
import threading
from urllib.request import Request, urlopen

from gainspread.service import Application, make_server


def main():
    with tempfile.TemporaryDirectory() as state:
        server = make_server(Application("data", state), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            for path in ("/health", "/tables", "/view"):
                with urlopen(base + path, timeout=5) as response:
                    print(path, json.dumps(json.load(response), ensure_ascii=False))
            with urlopen(base + "/panel?base=2026-07&current=2026-10", timeout=5) as response:
                panel = json.load(response)
            print("/panel", json.dumps({
                "base_merchants": panel["base"]["aggregate"]["merchant_count"],
                "current_merchants": panel["current"]["aggregate"]["merchant_count"],
                "current_negative_savings": panel["distribution_indicators"]["current"][
                    "total_savings"]["negative_count"],
                "gini_applicable": panel["distribution_indicators"]["gini"][
                    "current_total_savings"]["applicable"],
            }, ensure_ascii=False))
            request = Request(base + "/cases", data=json.dumps({"label": "月度经营讨论", "options": {"period": "2026-09"}}, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=5) as response:
                created = json.load(response)
            with urlopen(base + "/cases/" + created["id"], timeout=5) as response:
                restored = json.load(response)
            assert created == restored
            print("研究记录已回读", restored["id"], restored["source_fingerprint"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    main()
