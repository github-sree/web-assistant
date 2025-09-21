import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

# Load .env
load_dotenv()

class PCFClient:
    def __init__(self):
        self.mock = os.getenv("USE_MOCK", "true").lower() == "true"
        self.api_url = os.getenv("PCF_API", "https://api.local.pcf")
        self.username = os.getenv("PCF_USERNAME", "user")
        self.password = os.getenv("PCF_PASSWORD", "pass")

    def get_app_status(self, app_name: str):
        if self.mock:
            return {"app": app_name, "status": "READY", "message": "Mock PCF status"}
        else:
            try:
                url = f"{self.api_url}/v3/apps?names={app_name}"
                response = requests.get(url, auth=HTTPBasicAuth(self.username, self.password), verify=False)
                response.raise_for_status()
                apps = response.json().get("resources", [])
                if apps:
                    return {
                        "app": app_name,
                        "guid": apps[0]["guid"],
                        "status": apps[0]["state"]
                    }
                else:
                    return {"app": app_name, "status": "NOT_FOUND"}
            except Exception as e:
                return {"error": str(e)}

    def restart_app(self, app_guid: str):
        if self.mock:
            return {"app_guid": app_guid, "action": "RESTARTED", "message": "Mock PCF restart"}
        else:
            try:
                url = f"{self.api_url}/v3/apps/{app_guid}/actions/restart"
                response = requests.post(url, auth=HTTPBasicAuth(self.username, self.password), verify=False)
                response.raise_for_status()
                return {"app_guid": app_guid, "action": "RESTARTED"}
            except Exception as e:
                return {"error": str(e)}

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    pcf = PCFClient()

    # Check status
    status = pcf.get_app_status("payment-service")
    print("PCF Status:", status)

    # Restart app (using mock GUID for demo)
    restart = pcf.restart_app("mock-guid-123")
    print("PCF Restart:", restart)