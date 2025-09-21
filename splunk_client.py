import os

import requests

from dotenv import load_dotenv

# Load environment variables from .env automatically
load_dotenv()


class SplunkClient:
    def __init__(self, mock: bool = True):
        self.mock = os.getenv("USE_MOCK", "true").lower() == "true"
        self.splunk_host = os.getenv("SPLUNK_HOST", "")
        self.splunk_token = os.getenv("SPLUNK_TOKEN", "")
        self.mock_logs = [{"event": "payment_failed", "message": "Payment service timeout on PCF node 3"},
                          {"event": "page_down", "message": "Checkout page returned 500 error"},
                          {"event": "slow_response", "message": "Search API latency > 2s"},
                          {"event": "auth_error", "message": "Okta authentication token expired"}, ]

    def query_logs(self, query: str):
        if self.mock:
            return [log for log in self.mock_logs if query in log["event"] or query in log["message"]]
        url = f"{self.splunk_host}/services/search/jobs"
        headers = {"Authorization": f"Bearer {self.splunk_token}"}
        data = {"search": f"search {query}", "output_mode": "json"}

        try:
            response = requests.post(url, headers=headers, data=data, verify=False)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    splunk = SplunkClient()
    result = splunk.query_logs("error")
    print("splunk_results", result)
