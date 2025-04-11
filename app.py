import uvicorn
from collections import deque
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
import uuid
from typing import Dict

from telegram_bot import create_telegram_app, run_telegram_bot, get_chat_id
from tools import get_tools, tool_mapping
from queue_storage import QueueStorage
from models import UserQuery, UserQueryQueueItem, UserQueryResultPending, CustomUUID

# Load environment variables
load_dotenv()

app = FastAPI()
current_dir = Path(__file__).parent

# Initialize queue storage
request_storage = QueueStorage("storage/requests")
response_storage = QueueStorage("storage/responses")

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

# Queue to store requests
request_queue: deque['UserQueryQueueItem'] = deque()
response_queue: deque['UserQueryQueueItem'] = deque()

@app.post("/ask", response_model=UserQueryResultPending, include_in_schema=False)
async def query(user_query: UserQuery):
    try:
        logger.info(f"Received query: {user_query}")
        
        # Create queue item with UUID
        queue_item = UserQueryQueueItem(
            id=str(uuid.uuid4()),
            query=user_query.query
        )
        
        # Add request to the queue and storage
        request_queue.append(queue_item)
        request_storage.save_item(queue_item)
        logger.info(f"Request added to queue: {queue_item}")
        
        result = UserQueryResultPending(
            status="queued", 
            message="Your request has been added to the queue",
            request_id=queue_item.id
        )
        logger.debug(f"Query response: {result}")
        return result
    except Exception as e:
        logger.error(f"Error response: {e}")

@app.get("/list_requests", include_in_schema=False)
async def list_requests():
    try:
        # Get items from both memory and storage
        memory_items = [queue_item.dict() for queue_item in request_queue]
        storage_items = [item.dict() for item in request_storage.list_items()]
        
        # Combine and deduplicate items
        all_items = {item['id']: item for item in memory_items + storage_items}
        queued_requests = list(all_items.values())
        
        logger.info(f"Listing queued requests: {queued_requests}")
        return JSONResponse({"queued_requests": queued_requests})
    except Exception as e:
        logger.error(f"Error listing requests: {e}")
        return JSONResponse({"status": "error", "message": str(e)})

@app.get("/get_request/{request_id}", include_in_schema=False)
async def get_request(request_id: CustomUUID):
    try:
        # Check in-memory queue first
        for queue_item in request_queue:
            if str(queue_item.id) == request_id:
                logger.info(f"Found request in memory: {queue_item}")
                return JSONResponse({
                    "status": "found", 
                    "request": queue_item.dict(),
                    "request_id": str(queue_item.id)
                })
        
        # Check storage
        stored_item = request_storage.load_item(request_id)
        if stored_item:
            logger.info(f"Found request in storage: {stored_item}")
            return JSONResponse({
                "status": "found",
                "request": stored_item.dict(),
                "request_id": str(stored_item.id)
            })
            
        return JSONResponse({"status": "not found", "message": "Request not found in the queue"})
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return JSONResponse({"status": "error", "message": str(e)})

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
