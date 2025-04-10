import uvicorn
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
import json
import traceback
import logging
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

from telegram_bot import create_telegram_app, run_telegram_bot, get_chat_id
from tools import get_tools, tool_mapping

# Load environment variables
load_dotenv()

app = FastAPI()
current_dir = Path(__file__).parent

# Configure logging
logging.basicConfig(
    filename="server.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Initialize LLM with tools
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
).bind_tools(get_tools())

async def ask(content):
    try:
        prompt = HumanMessage(content=content)
        messages = [prompt]

        ai_message = llm.invoke(messages)
        logger.debug(f"AI message received: {ai_message}")
        for tool_call in ai_message.tool_calls:
            selected_tool = tool_mapping[tool_call["name"].lower()]
            tool_output = selected_tool.invoke(tool_call["args"])

            tool_output_str = json.dumps(tool_output)
            messages.append(ToolMessage(tool_output_str, tool_call_id=tool_call["id"]))

            logger.debug(f"Tool output: {tool_output_str}")

        for m in messages:
            logger.debug(f"Message type: {type(m)} - Content: {m}")

        result = llm.invoke(messages)
        logger.debug(f"Final AI response: {result}")
        return result
    except Exception as e:
        error = {"content": {"status": "error", "message": str(e)}}
        logger.debug(f"Ask error: {error}. {traceback.format_exc()}")
        return error

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def chat(request: Request):
    logger.info("Serving chat page")
    with open(current_dir / "html" / "page.html") as f:
        return HTMLResponse(f.read())

@app.post("/ask", include_in_schema=False)
async def query(request: Request):
    try:
        body = await request.json()
        logger.info(f"Received query: {body}")
        response = await ask(body["message"])
        if hasattr(response, "content"):
            content = response.content
        elif "content" in response:
            content = response["content"]

        print("Got response.content", content)
        print("Got response.content type", type(content))
        result = {"response": str(content)}
        logger.debug(f"Query response: {result}")
        print(">>> type of result", type(result))
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error response: {e}")

@app.get("/")
async def root():
    return {"message": "FastAPI is working!"}


# Endpoint to send a message
@app.get("/send-message")
async def send_message(username: str, text: str):
    chat_id = get_chat_id(username)
    if chat_id is None:
        return {"status": "error", "message": "User not found"}
    
    try:
        await telegram_app.bot.send_message(chat_id=chat_id, text=text)
        return {"status": "message sent"}
    except Exception as e:
        print(f"Error sending message: {e}")
        return {"status": "error", "message": str(e)}

# Main coroutine to run both services
async def main():
    # Initialize telegram app
    global telegram_app
    telegram_app = create_telegram_app()
    
    # Start both tasks
    _ = asyncio.create_task(run_telegram_bot(telegram_app))
    
    config = uvicorn.Config(app=app, host="0.0.0.0", port=9081, reload=False)
    server = uvicorn.Server(config)
    await server.serve()

    # Clean shutdown
    await telegram_app.stop()
    await telegram_app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
