import uvicorn
from collections import deque
from models import RequestState
import asyncio
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_ollama import ChatOllama
import json
import traceback
import logging
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid
from typing import Dict
from datetime import datetime, timedelta
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

from telegram_bot import create_telegram_app, run_telegram_bot, get_chat_id, send_notification, get_first_registered_chat_id
from tools import get_tools, tool_mapping
from queue_storage import QueueStorage
from models import UserQuery, UserQueryQueueItem, UserQueryResult, CustomUUID
from shared_state import pending_approvals  # Import shared state

# Load environment variables
load_dotenv()

app = FastAPI()
current_dir = Path(__file__).parent

# Initialize queue storage
request_storage = QueueStorage("storage/requests")

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

# Bypass flag for approval system
BYPASS_APPROVAL = False  # Set to True to skip approval process

async def ask(query: str) -> AIMessage:
    """Process a user query using the LLM."""
    try:
        print(f"\n[ASK] Starting query processing for: {query}")
        
        # Create a new chat model instance for each query
        chat_model = ChatOllama(
            model="llama3.1:8b",
            temperature=0.1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        ).bind_tools(get_tools())  # Bind tools to the chat model
        
        # Create initial message
        prompt = HumanMessage(content=query)
        messages = [prompt]
        
        # Get first response from LLM
        print("[ASK] Invoking LLM with initial message")
        ai_message = await chat_model.ainvoke(messages)
        print(f"[ASK] AI message received: {ai_message}")
        
        # Process tool calls if any
        if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
            print("[ASK] Processing tool calls")
            for tool_call in ai_message.tool_calls:
                print(f"[ASK] Tool call: {tool_call}")
                selected_tool = tool_mapping[tool_call["name"].lower()]
                print(f"[ASK] Selected tool: {selected_tool}")
                
                tool_output = selected_tool.invoke(tool_call["args"])
                print(f"[ASK] Tool output: {tool_output}")
                
                tool_output_str = json.dumps(tool_output)
                messages.append(ToolMessage(tool_output_str, tool_call_id=tool_call["id"]))
                print(f"[ASK] Added tool message to conversation")
            
            # Get final response with tool outputs
            print("[ASK] Getting final response with tool outputs")
            result = await chat_model.ainvoke(messages)
            print(f"[ASK] Final result: {result}")
            return result
        
        return ai_message
    except Exception as e:
        print(f"[ASK] Error details: {type(e)} - {str(e)}")
        print(f"[ASK] Error traceback: {traceback.format_exc()}")
        logger.error(f"Error processing query: {e}")
        return AIMessage(content=f"Error processing query: {str(e)}")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def chat(request: Request):
    logger.info("Serving chat page")
    with open(current_dir / "html" / "page.html") as f:
        return HTMLResponse(f.read())

# Queue to store requests
request_queue: deque['UserQueryQueueItem'] = deque()
response_queue: deque['UserQueryQueueItem'] = deque()

@app.post("/ask", response_model=UserQueryResult, include_in_schema=False)
async def query(
    user_query: UserQuery,
    wait_time_secs: int = Query(default=60, description="Time to wait for approval in seconds")
):
    try:
        print(f"Received query: {user_query}")
        print(f"Wait time received: {wait_time_secs} seconds")
        
        if BYPASS_APPROVAL:
            print("[APP.PY] Bypassing approval system")
            # Directly process the query
            response = await ask(user_query.query)
            print(f"[APP.PY] Response from ask: {response}")
            print(f"[APP.PY] Response content: {response.content}")
            
            return UserQueryResult(
                status=RequestState.APPROVED,
                message="Request processed (approval bypassed)",
                request_id=str(uuid.uuid4()),
                query=user_query.query,
                result=response.content
            )
        
        # Create queue item with UUID
        queue_item = UserQueryQueueItem(
            id=str(uuid.uuid4()),
            query=user_query.query
        )
        
        # Add request to the queue and storage
        request_queue.append(queue_item)
        request_storage.save_item(queue_item)
        logger.info(f"Request added to queue: {queue_item}")
        
        # Send Telegram notification
        notification_message = f"New request queued:\nID: {queue_item.id}\nQuery: {queue_item.query}"
        await send_notification(telegram_app, notification_message)
        
        # Wait for approval/rejection if wait_time_secs > 0
        if wait_time_secs > 0:
            start_time = datetime.now()
            while (datetime.now() - start_time).total_seconds() < wait_time_secs:
                # Check request status
                stored_item = request_storage.load_item(queue_item.id)
                if stored_item:
                    if stored_item.can_run:
                        print(f"\n[APP.PY] Processing request {queue_item.id} with can_run=True")
                        # Run the ask function on the request
                        response = await ask(user_query.query)
                        print(f"[APP.PY] Response from ask: {response}")
                        print(f"[APP.PY] Response content: {response.content}")
                        
                        # Update the stored item with the result
                        stored_item.result = response.content
                        print(f"[APP.PY] Updated stored item result: {stored_item.result}")
                        request_storage.save_item(stored_item)
                        print(f"[APP.PY] Saved item to storage with result: {stored_item.result}")
                        
                        # If approved, return the result immediately
                        if stored_item.state == RequestState.APPROVED:
                            print(f"[APP.PY] Request {queue_item.id} is approved, returning result")
                            return UserQueryResult(
                                status=RequestState.APPROVED,
                                message="Request approved and processed",
                                request_id=queue_item.id,
                                query=user_query.query,
                                result=response.content
                            )
                        # If just can_run but not approved, send follow-up message
                        else:
                            print(f"\n[APP.PY] Processing request {queue_item.id} with can_run=True but not approved")
                            # Add to pending approvals with the result first
                            chat_id = get_first_registered_chat_id()
                            if chat_id:
                                print(f"[APP.PY] Adding to pending_approvals: chat_id={chat_id}, request_id={queue_item.id}, action=approve_with_result")
                                # Ensure chat_id is an integer
                                chat_id = int(chat_id)
                                pending_approvals[chat_id] = (queue_item.id, queue_item.query, "approve_with_result")
                                print(f"[APP.PY] Current pending_approvals: {pending_approvals}")
                            else:
                                print("[APP.PY] No chat_id found for first registered user!")
                            
                            # Send follow-up message with the result
                            follow_up_message = (
                                f"Request processed but not approved:\n"
                                f"ID: {queue_item.id}\n"
                                f"Query: {queue_item.query}\n"
                                f"Result: {response.content}\n\n"
                                "Would you like to approve this result? (y/n)"
                            )
                            print(f"[APP.PY] Sending follow-up message: {follow_up_message}")
                            await send_notification(telegram_app, follow_up_message)
                            
                            # Continue waiting for final approval/rejection
                            print(f"[APP.PY] Starting wait loop for final approval/rejection")
                            while (datetime.now() - start_time).total_seconds() < wait_time_secs:
                                stored_item = request_storage.load_item(queue_item.id)
                                if stored_item:
                                    print(f"[APP.PY] Checking stored item state: {stored_item.state}")
                                    if stored_item.state == RequestState.APPROVED:
                                        print(f"[APP.PY] Request {queue_item.id} was approved, returning result")
                                        return UserQueryResult(
                                            status=RequestState.APPROVED,
                                            message="Request approved and processed",
                                            request_id=queue_item.id,
                                            query=user_query.query,
                                            result=response.content
                                        )
                                    elif stored_item.state == RequestState.REJECTED:
                                        print(f"[APP.PY] Request {queue_item.id} was rejected")
                                        return UserQueryResult(
                                            status=RequestState.REJECTED,
                                            message="Request was rejected",
                                            request_id=queue_item.id,
                                            query=user_query.query
                                        )
                                await asyncio.sleep(1)
                            
                            print(f"[APP.PY] Timed out waiting for approval/rejection of request {queue_item.id}")
                            return UserQueryResult(
                                status=RequestState.PENDING,
                                message="Request processed, waiting for approval",
                                request_id=queue_item.id,
                                query=user_query.query
                            )
                    elif stored_item.state == RequestState.REJECTED:
                        return UserQueryResult(
                            status=RequestState.REJECTED,
                            message="Request was rejected",
                            request_id=queue_item.id,
                            query=user_query.query
                        )
                await asyncio.sleep(1)  # Check every second
        
        # If we get here, either wait_time_secs was 0 or we timed out
        result = UserQueryResult(
            status=RequestState.PENDING, 
            message="Your request has been added to the queue",
            request_id=queue_item.id,
            query=user_query.query
        )
        logger.debug(f"Query response: {result}")
        return result
    except Exception as e:
        logger.error(f"Error response: {e}")
        raise

@app.get("/list_requests", include_in_schema=False)
async def list_requests():
    try:
        # Get all items from storage
        print("Attempting to retrieve all items from storage.")
        storage_items = request_storage.list_items()
        print(f"Retrieved {len(storage_items)} items from storage.")
        
        # Group requests by state
        grouped_requests = {
            RequestState.PENDING.value: [],
            RequestState.APPROVED.value: [],
            RequestState.REJECTED.value: []
        }
        
        for request in storage_items:
            state = request.state
            print(f"Processing request with ID: {request.id}, State: {state}")
            # Convert the request to a dictionary and handle datetime serialization
            request_dict = request.model_dump()
            request_dict['created_at'] = request_dict['created_at'].isoformat()
            grouped_requests[state].append(request_dict)
        
        logger.info(f"Listing grouped requests: {grouped_requests}")
        print(f"Grouped requests: {grouped_requests}")
        return JSONResponse({"grouped_requests": grouped_requests})
    except Exception as e:
        logger.error(f"Error listing requests: {e}")
        print(f"Error occurred: {e}")
        return JSONResponse({"status": "error", "message": str(e)})

@app.get("/get_request/{request_id}", include_in_schema=False)
async def get_request(request_id: CustomUUID):
    try:
        # Check storage first since it has the final state
        stored_item = request_storage.load_item(request_id)
        if stored_item:
            logger.info(f"Found request in storage: {stored_item}")
            request_dict = stored_item.model_dump()
            request_dict['created_at'] = request_dict['created_at'].isoformat()
            
            # Only include result if the request is approved
            result = stored_item.result if (stored_item.state == RequestState.APPROVED and hasattr(stored_item, 'result')) else None
            
            return JSONResponse({
                "status": stored_item.state,
                "message": "Request found",
                "request_id": str(stored_item.id),
                "query": stored_item.query,
                "result": result
            })
        
        # If not in storage, check in-memory queue
        for queue_item in request_queue:
            if str(queue_item.id) == request_id:
                logger.info(f"Found request in memory: {queue_item}")
                request_dict = queue_item.model_dump()
                request_dict['created_at'] = request_dict['created_at'].isoformat()
                
                # Only include result if the request is approved
                result = queue_item.result if (queue_item.state == RequestState.APPROVED and hasattr(queue_item, 'result')) else None
                
                return JSONResponse({
                    "status": queue_item.state,
                    "message": "Request found",
                    "request_id": str(queue_item.id),
                    "query": queue_item.query,
                    "result": result
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

async def process_requests():
    while True:
        try:
            # Get all pending requests
            requests = request_storage.get_all_items()
            
            for request in requests:
                # Only process requests that are marked to run
                if request.can_run:
                    try:
                        # Process the request
                        result = await process_request(request)
                        
                        # Update the request with the result
                        request.result = result
                        request_storage.update_item(request.id, request)
                        
                        logger.info(f"Processed request {request.id}")
                    except Exception as e:
                        logger.error(f"Error processing request {request.id}: {str(e)}")
                        request.result = f"Error processing request: {str(e)}"
                        request_storage.update_item(request.id, request)
            
            await asyncio.sleep(1)  # Check every second
        except Exception as e:
            logger.error(f"Error in process_requests: {str(e)}")
            await asyncio.sleep(1)  # Wait before retrying

if __name__ == "__main__":
    asyncio.run(main())
