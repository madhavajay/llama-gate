import json
import os
from pathlib import Path
from typing import List, Optional
from models import UserQueryQueueItem, CustomUUID, RequestState
from datetime import datetime

class QueueStorage:
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_item(self, item: UserQueryQueueItem):
        """Save a queue item to a file named by its UUID."""
        file_path = self.storage_dir / f"{item.id}.json"
        # Convert the item to a dictionary and handle datetime serialization
        item_dict = item.model_dump()
        item_dict['created_at'] = item_dict['created_at'].isoformat()
        # Ensure boolean values are Python-style
        item_dict['can_run'] = bool(item_dict['can_run'])
        with open(file_path, "w") as f:
            json.dump(item_dict, f)

    def load_item(self, item_id: CustomUUID) -> Optional[UserQueryQueueItem]:
        """Load a queue item from its UUID file."""
        file_path = self.storage_dir / f"{item_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r") as f:
            data = json.load(f)
            # Convert the ISO format string back to datetime
            if 'created_at' in data:
                data['created_at'] = datetime.fromisoformat(data['created_at'])
            # Ensure boolean values are Python-style
            if 'can_run' in data:
                data['can_run'] = bool(data['can_run'])
            return UserQueryQueueItem(**data)

    def list_items(self, state: Optional[RequestState] = None) -> List[UserQueryQueueItem]:
        """List all queue items in the directory, optionally filtered by state."""
        items = []
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    # Convert the ISO format string back to datetime
                    if 'created_at' in data:
                        data['created_at'] = datetime.fromisoformat(data['created_at'])
                    item = UserQueryQueueItem(**data)
                    if state is None or item.state == state:
                        items.append(item)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error loading file {file_path}: {e}")
                continue
        return items

    def update_item_state(self, item_id: CustomUUID, new_state: RequestState) -> bool:
        """Update the state of a queue item."""
        file_path = self.storage_dir / f"{item_id}.json"
        if not file_path.exists():
            return False
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        data["state"] = new_state
        
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        return True

    def remove_item(self, item_id: CustomUUID) -> bool:
        """Remove a queue item file by its UUID."""
        file_path = self.storage_dir / f"{item_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False 