import logging
import time
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Form
from langchain_community.llms.ollama import Ollama
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, FileResponse
from starlette.staticfiles import StaticFiles

from concourse_client import ConcourseClient
from intent_prediction import IntentPrediction
from jira_client import JiraClient
from pcf_client import PCFClient
from splunk_client import SplunkClient

# -------------------------
# Config & setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("self-healing-assistant")

app = FastAPI(title="Self-Healing AI Assistant", version="1.1")

# Initialize integrations (singletons at startup)
splunk = SplunkClient()
pcf = PCFClient()
concourse = ConcourseClient()
jira = JiraClient()
clf = IntentPrediction()
llm = Ollama(model="mistral")  # preload llama once

# -------------------------
# Optimized Prompt Template
# -------------------------
PROMPT_TEMPLATE = """You are an SRE assistant.
Inputs:
- Query: {query}
- Intent: {intent}
- Confidence: {confidence}
- Result: {result}

Rules:
- If Confidence < 0.5 or Result contains "No mapped action", reply exactly: "Intent unclear. Please provide more details."
- Otherwise, reply with 1–3 confident, customer-friendly sentences that:
  • Explain the issue briefly (based on inputs & result).  
  • Naturally mention Splunk, PCF, Concourse, or Jira if relevant.  
  • Indicate actions already performed (e.g., restarted app, triggered pipeline).  

Do not output steps, lists, or extra text.
"""

# -------------------------
# Utils
# -------------------------
loop = asyncio.get_event_loop()

async def handle_blocking(func, *args, **kwargs):
    """Run blocking code in threadpool executor."""
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

def sanitize_result(result):
    """Prevent passing massive logs to LLM (truncate if >1000 chars)."""
    text = str(result)
    return text[:1000] + "..." if len(text) > 1000 else text

# -------------------------
# Orchestration Logic
# -------------------------
async def handle_intent(intent: str, query: str):
    """Async orchestration of integrations based on predicted intent."""
    if intent == "check_logs":
        return await handle_blocking(splunk.query_logs, "payment-service")

    elif intent == "app_status":
        return await handle_blocking(pcf.get_app_status, "payment-service")

    elif intent == "restart_app":
        return await handle_blocking(pcf.restart_app, "mock-guid-123")

    elif intent in ["trigger_pipeline", "technical_issue"]:
        return await handle_blocking(concourse.trigger_pipeline, "ci-pipeline", "build-job")

    elif intent == "create_ticket":
        return await handle_blocking(
            jira.create_issue, "SELFHEAL", "Bug", f"Issue reported: {query}"
        )

    return {"intent": intent, "message": "No mapped action for this intent."}

# -------------------------
# API Endpoint
# -------------------------
@app.post("/assistant")
async def assistant(query: str = Form(...)):
    if not clf.classifier or not clf.embedder:
        raise HTTPException(status_code=400, detail="Classifier not trained. Train first using /train")

    # 1. Predict intent
    prediction = clf.predict(query)
    intent = prediction["intent"]
    confidence = prediction["confidence"]

    # 2. Apply confidence threshold
    if intent == "unknown" or confidence < 0.3:
        return {
            "intent": "unknown",
            "confidence": confidence,
            "message": "Not confident enough to take action."
        }

    # 3. Route to integration + stream response
    try:
        t0 = time.time()
        result = await handle_intent(intent, query)
        elapsed = time.time() - t0
        logger.info(f"[Timing] handle_intent took {elapsed:.2f}s")

        result_sanitized = sanitize_result(result)

        prompt = PROMPT_TEMPLATE.format(
            query=query,
            intent=intent,
            confidence=confidence,
            result=result_sanitized
        )

        logger.info(f"[Assistant] Intent={intent}, Confidence={confidence:.2f}, Query={query}")

        return StreamingResponse(
            stream_llm(prompt),
            media_type="text/plain; charset=utf-8;"
        )
    except Exception as e:
        logger.error(f"Error handling intent {intent}: {e}")
        raise HTTPException(status_code=500, detail="Internal error during intent handling")

# -------------------------
# LLM Streaming
# -------------------------
async def stream_llm(prompt: str):
    try:
        # Optimized generation options for speed + control
        options = {
            "num_ctx": 512,  # smaller context window for faster eval
            "max_tokens": 80,  # cap response length
            "temperature": 0.2,  # reduce randomness for faster convergence
            "top_p": 0.9,  # balanced sampling
            "stop": ["\n\n"]  # stop early to avoid long rambling
        }

        async for chunk in llm.astream(prompt, options=options):
            if chunk:
                yield chunk
    except Exception as e:
        yield f"[Error streaming response: {str(e)}]"

# -------------------------
# Lifespan Event
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if clf.load():
        logger.info("✅ Model loaded and ready at startup.")
    else:
        logger.warning("⚠️ No model found. Train with /train before usage.")
    yield
    logger.info("👋 Shutting down Self-Healing Assistant")

app.router.lifespan_context = lifespan

# ---------------------------------------------------------------------------
# Middleware & Frontend setup
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# ---------------------------------------
# Adaptive Learning Endpoints
# ---------------------------------------

@app.post("/feedback")
async def submit_feedback(
    query: str = Form(...),
    predicted_intent: str = Form(...),
    correct_intent: str = Form(...)
):
    """
    Collects feedback from user.
    - query: the original question asked
    - predicted_intent: what the classifier thought
    - correct_intent: what the user says is correct
    """
    try:
        clf.queue_feedback(query, correct_intent)
        logger.info(
            f"[Feedback] Query='{query}' Predicted='{predicted_intent}' → Correct='{correct_intent}'"
        )
        return {"status": "queued", "message": "Feedback stored for adaptive learning."}
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Could not store feedback")


@app.post("/apply-feedback")
async def apply_feedback():
    """
    Applies all queued feedback to update the model.
    Runs in batch (not immediate partial_fit) for stability.
    """
    try:
        updated_count = clf.apply_feedback()
        if updated_count > 0:
            logger.info(f"[AdaptiveLearning] Applied {updated_count} feedback samples.")
            return {"status": "success", "updated_samples": updated_count}
        else:
            return {"status": "no-op", "message": "No feedback to apply."}
    except Exception as e:
        logger.error(f"Error applying feedback: {e}")
        raise HTTPException(status_code=500, detail="Could not apply feedback")


@app.post("/train")
async def train_model():
    """
    (Re)train the classifier from labeled dataset.
    Useful for cold start or refreshing from scratch.
    """
    try:
        # Load your dataset here (MongoDB or CSV)
        dataset = clf.load_data_from_mongo()
        if not dataset:
            raise HTTPException(status_code=400, detail="No dataset found for training.")

        clf.train(dataset)
        logger.info("[Training] Classifier trained and saved successfully.")
        return {"status": "success", "message": "Model trained successfully"}
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail="Error during training")

# -------------------------
# Top-N Intent Predictions
# -------------------------
from fastapi import Form

@app.post("/top-intents")
async def top_intents(query: str = Form(...), n: int = 3):
    """
    Returns top-N predicted intents with their probabilities.
    Useful for UI when model is uncertain.
    """
    if not clf.classifier or not clf.embedder:
        raise HTTPException(status_code=400, detail="Classifier not trained. Train first using /train")

    emb = clf.embed(query).reshape(1, -1)
    probs = clf.classifier.predict_proba(emb)[0]

    classes = clf.classifier.classes_
    # Sort intents by probability descending
    top_indices = probs.argsort()[::-1][:n]
    top_intents = [classes[i] for i in top_indices]
    top_probs = [float(probs[i]) for i in top_indices]

    return {"top_intents": top_intents, "top_probs": top_probs}