# jira_client.py
import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

# Load .env
load_dotenv()

class JiraClient:
    def __init__(self):
        self.mock = os.getenv("USE_MOCK", "true").lower() == "true"
        self.api_url = os.getenv("JIRA_API", "https://jira.local/rest/api/2")
        self.username = os.getenv("JIRA_USERNAME", "user")
        self.password = os.getenv("JIRA_PASSWORD", "pass")
        self.project_key = os.getenv("JIRA_PROJECT", "TEST")

    def create_issue(self, summary: str, description: str, issue_type: str = "Task"):
        if self.mock:
            return {
                "id": "TEST-1",
                "project": self.project_key,
                "summary": summary,
                "description": description,
                "issue_type": issue_type,
                "status": "Open",
                "message": "Mock Jira issue created"
            }
        else:
            try:
                url = f"{self.api_url}/issue"
                payload = {
                    "fields": {
                        "project": {"key": self.project_key},
                        "summary": summary,
                        "description": description,
                        "issuetype": {"name": issue_type}
                    }
                }
                response = requests.post(
                    url,
                    json=payload,
                    auth=HTTPBasicAuth(self.username, self.password),
                    verify=False
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e)}

    def get_issue(self, issue_id: str):
        if self.mock:
            return {
                "id": issue_id,
                "summary": "Mock issue summary",
                "status": "Open",
                "message": "Mock Jira issue fetch"
            }
        else:
            try:
                url = f"{self.api_url}/issue/{issue_id}"
                response = requests.get(
                    url,
                    auth=HTTPBasicAuth(self.username, self.password),
                    verify=False
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e)}

    def transition_issue(self, issue_id: str, new_status: str):
        if self.mock:
            return {
                "id": issue_id,
                "status": new_status,
                "message": "Mock Jira issue transitioned"
            }
        else:
            try:
                # ⚠️ Jira requires transition IDs, not names. For simplicity, assume mapping handled elsewhere
                url = f"{self.api_url}/issue/{issue_id}/transitions"
                payload = {"transition": {"id": new_status}}
                response = requests.post(
                    url,
                    json=payload,
                    auth=HTTPBasicAuth(self.username, self.password),
                    verify=False
                )
                response.raise_for_status()
                return {"id": issue_id, "status": new_status}
            except Exception as e:
                return {"error": str(e)}

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    jira = JiraClient()

    # Create issue
    issue = jira.create_issue("Payment failed", "Customer reports payment gateway down")
    print("Jira Create:", issue)

    # Get issue
    issue_id = issue.get("id", "TEST-1")
    fetched = jira.get_issue(issue_id)
    print("Jira Fetch:", fetched)

    # Transition issue
    transitioned = jira.transition_issue(issue_id, "2")  # Example transition ID
    print("Jira Transition:", transitioned)