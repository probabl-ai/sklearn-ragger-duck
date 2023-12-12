"""Main entrypoint for the app."""

import json
import logging
import os
import sys
from importlib import import_module
from pathlib import Path

import joblib
import resources as res
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama
from schemas import WSMessage
from sentence_transformers import CrossEncoder

sys.path.append(str(Path(__file__).parent.parent))
from rag_based_llm.prompt import QueryAgent
from rag_based_llm.retrieval import RetrieverReranker

DEFAULT_PORT = 8123

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


config_module = (
    os.getenv("CONFIGURATION") if os.getenv("CONFIGURATION") is not None else "default"
)
logger.info(f"Configuration: {config_module} ")
conf = import_module(f"configuration.{config_module}")


async def send(ws, msg: str, type: str):
    message = WSMessage(sender="bot", message=msg, type=type)
    await ws.send_json(message.dict())


@app.on_event("startup")
async def startup_event():
    global agent

    api_semantic_retriever = joblib.load(conf.SEMANTIC_RETRIEVER_PATH)
    api_lexical_retriever = joblib.load(conf.LEXICAL_RETRIEVER_PATH)

    cross_encoder = CrossEncoder(model_name=conf.CROSS_ENCODER_PATH, device=conf.DEVICE)
    retriever_reranker = RetrieverReranker(
        cross_encoder=cross_encoder,
        semantic_retriever=api_semantic_retriever,
        lexical_retriever=api_lexical_retriever,
        threshold=conf.CROSS_ENCODER_THRESHOLD,
        min_top_k=conf.CROSS_ENCODER_MIN_TOP_K,
        max_top_k=conf.CROSS_ENCODER_MAX_TOP_K,
    )

    llm = Llama(
        model_path=conf.LLM_PATH,
        device=conf.DEVICE,
        n_gpu_layers=conf.GPU_LAYERS,
        n_threads=conf.N_THREADS,
        n_ctx=conf.CONTEXT_TOKENS,
    )
    agent = QueryAgent(llm=llm, retriever=retriever_reranker)
    logging.info("Server started")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")


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
    await send(websocket, "Question the scikit-learn bot!", "info")

    while True:
        try:
            response_complete = ""
            start_type = ""

            received_text = await websocket.receive_text()
            payload = json.loads(received_text)

            prompt = payload["query"]
            start_type = "start"
            logger.info(f"Temperature: {payload['temperature']}")
            logger.info(f"Prompt: {prompt}")

            await send(websocket, "Analyzing prompt...", "info")
            stream, sources = agent(
                prompt,
                echo=False,
                stream=True,
                max_tokens=conf.MAX_RESPONSE_TOKENS,
                temperature=conf.TEMPERATURE,
            )
            for i in stream:
                response_text = i.get("choices", [])[0].get("text", "")
                answer_type = start_type if response_complete == "" else "stream"
                response_complete += response_text
                await send(websocket, response_text, answer_type)
            response_complete += "\n\nSource(s):\n" + ";\n".join(sources)
            await send(websocket, response_complete, start_type)

            await send(websocket, "", "end")
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            await send(websocket, "Sorry, something went wrong. Try again.", "error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT)
