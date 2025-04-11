import json
from pathlib import Path
from typing import List, Optional
from models import UserQueryQueueItem, CustomUUID

class QueueStorage:
    def __init__(self, queue_dir: str):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def save_item(self, item: UserQueryQueueItem) -> None:
        """Save a queue item to a file named by its UUID."""
        file_path = self.queue_dir / f"{item.id}.json"
        with open(file_path, 'w') as f:
            json.dump(item.dict(), f)

    def load_item(self, item_id: CustomUUID) -> Optional[UserQueryQueueItem]:
        """Load a queue item from its UUID file."""
        file_path = self.queue_dir / f"{item_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, 'r') as f:
            data = json.load(f)
            return UserQueryQueueItem(**data)

    def list_items(self) -> List[UserQueryQueueItem]:
        """List all queue items in the directory."""
        items = []
        for file_path in self.queue_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
                items.append(UserQueryQueueItem(**data))
        return items

    def remove_item(self, item_id: CustomUUID) -> bool:
        """Remove a queue item file by its UUID."""
        file_path = self.queue_dir / f"{item_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False 