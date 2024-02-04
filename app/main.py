"""Main entrypoint for the app."""

import json
import logging
import os
import sys
from importlib import import_module
from pathlib import Path

import joblib
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama
from sentence_transformers import CrossEncoder

sys.path.append(str(Path(__file__).parent.parent))
import resources as res
from schemas import WSMessage

from ragger_duck.prompt import BasicPromptingStrategy
from ragger_duck.retrieval import RetrieverReranker

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

logging.basicConfig(level=logging.INFO)

config_module = (
    os.getenv("CONFIGURATION") if os.getenv("CONFIGURATION") is not None else "default"
)
logging.info(f"Configuration: {config_module}")
conf = import_module(f"configuration.{config_module}")
DEVICE = os.getenv("DEVICE", "cpu")
logging.info(f"Device intended to be used: {DEVICE}")


async def send(ws, msg: str, type: str):
    message = WSMessage(sender="bot", message=msg, type=type)
    await ws.send_json(message.dict())


@app.on_event("startup")
async def startup_event():
    global agent

    api_semantic_retriever = joblib.load(conf.API_SEMANTIC_RETRIEVER_PATH)
    api_lexical_retriever = joblib.load(conf.API_LEXICAL_RETRIEVER_PATH)
    user_guide_semantic_retriever = joblib.load(conf.API_SEMANTIC_RETRIEVER_PATH)
    user_guide_lexical_retriever = joblib.load(conf.API_LEXICAL_RETRIEVER_PATH)
    cross_encoder = CrossEncoder(model_name=conf.CROSS_ENCODER_PATH, device=DEVICE)
    retriever = RetrieverReranker(
        retrievers=[
            api_semantic_retriever.set_params(top_k=conf.API_SEMANTIC_TOP_K),
            api_lexical_retriever.set_params(top_k=conf.API_LEXICAL_TOP_K),
            user_guide_semantic_retriever.set_params(
                top_k=conf.USER_GUIDE_SEMANTIC_TOP_K
            ),
            user_guide_lexical_retriever.set_params(
                top_k=conf.USER_GUIDE_LEXICAL_TOP_K
            ),
        ],
        cross_encoder=cross_encoder,
        threshold=conf.CROSS_ENCODER_THRESHOLD,
        min_top_k=conf.CROSS_ENCODER_MIN_TOP_K,
        max_top_k=conf.CROSS_ENCODER_MAX_TOP_K,
    )

    llm = Llama(
        model_path=conf.LLM_PATH,
        device=DEVICE,
        n_gpu_layers=conf.GPU_LAYERS,
        n_threads=conf.N_THREADS,
        n_ctx=conf.CONTEXT_TOKENS,
    )
    agent = BasicPromptingStrategy(llm=llm, retriever=retriever)
    logging.info("Server started")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/img/favicon.ico")


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "res": res, "conf": conf}
    )


@app.get("/inference.js")
async def get(request: Request):  # noqa: F811
    return templates.TemplateResponse(
        "inference.js",
        {"request": request, "wsurl": os.getenv("WSURL", ""), "res": res, "conf": conf},
    )


@app.websocket("/inference")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await send(
        websocket, "I'm ragger duck! Ask me question about scikit-learn!", "info"
    )

    while True:
        try:
            response_complete = ""
            start_type = ""

            received_text = await websocket.receive_text()
            payload = json.loads(received_text)

            prompt = payload["query"]
            start_type = "start"
            logging.info(f"Getting info from websocket: {payload}")

            await send(websocket, "Analyzing prompt...", "info")
            agent.set_params(
                retriever__threshold=payload["cutoff"],
                retriever__min_top_k=payload["min_top_k"],
                retriever__max_top_k=payload["max_top_k"],
            )
            stream, sources = agent(
                prompt,
                echo=False,
                stream=True,
                max_tokens=conf.MAX_RESPONSE_TOKENS,
                temperature=payload["temperature"],
            )
            for i in stream:
                response_text = i.get("choices", [])[0].get("text", "")
                answer_type = start_type if response_complete == "" else "stream"
                response_complete += response_text
                await send(websocket, response_text, answer_type)
            contextual_sources = "\n".join([f"<{src}>" for src in sources])
            response_complete += "\n\nContextual source(s):\n" + contextual_sources
            await send(websocket, response_complete, start_type)

            await send(websocket, "", "end")
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            await send(websocket, "Sorry, something went wrong. Try again.", "error")
