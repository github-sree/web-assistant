import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

# Load .env
load_dotenv()

class ConcourseClient:
    def __init__(self):
        self.mock = os.getenv("USE_MOCK", "true").lower() == "true"
        self.api_url = os.getenv("CONCOURSE_API", "https://localhost:8080/api/v1")
        self.team = os.getenv("CONCOURSE_TEAM", "main")
        self.username = os.getenv("CONCOURSE_USERNAME", "user")
        self.password = os.getenv("CONCOURSE_PASSWORD", "pass")

    def trigger_pipeline(self, pipeline_name: str, job_name: str):
        if self.mock:
            return {
                "pipeline": pipeline_name,
                "job": job_name,
                "status": "SUCCESS",
                "message": "Mock pipeline triggered successfully"
            }
        else:
            try:
                url = f"{self.api_url}/teams/{self.team}/pipelines/{pipeline_name}/jobs/{job_name}/builds"
                response = requests.post(url, auth=HTTPBasicAuth(self.username, self.password), verify=False)
                response.raise_for_status()
                return {
                    "pipeline": pipeline_name,
                    "job": job_name,
                    "status": "TRIGGERED",
                    "response": response.json() if response.content else {}
                }
            except Exception as e:
                return {"error": str(e)}

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    concourse = ConcourseClient()

    # Trigger pipeline
    result = concourse.trigger_pipeline("sample-pipeline", "build")
    print("Concourse Result:", result)