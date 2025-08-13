import os
import json
import time
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from agency_swarm.tools import BaseTool
from pydantic import Field
from typing import Dict
from dotenv import load_dotenv


load_dotenv()

service_account_key = json.loads(os.getenv("SERVICE_ACCOUNT_KEY_FIREBASE"))

if not firebase_admin._apps:
    cred = credentials.Certificate(service_account_key) 
    default_app = initialize_app(cred)
db = firestore.client()

# Max waiting time for function completion
timeout = 50

def load_task_data(conversation_id, task_id):
    key = f"task_{task_id}"
    doc = db.collection("agencii-chats").document(conversation_id).get()

    if doc.exists:
        if key in doc.to_dict():
            return doc.to_dict()[key]
        
    return None

def delete_task(conversation_id, task_id):
    key = f"task_{task_id}"
    db.collection("agencii-chats").document(conversation_id).update({key: firestore.DELETE_FIELD})

class CheckProgress(BaseTool):
    """
    Tool that checks the progress of an asynchronous task and returns its status and result if completed.
    """

    task_id: str = Field(
        description="The ID of the task to check"
    )

    class ToolConfig:
        one_call_at_a_time = True

    def run(self) -> Dict:
        """
        Check the status of a task and return its current state
        
        Returns:
            Dict containing:
            - status: The task status ("processing", "completed", "error")
            - result: The task result (if completed)
            - error: Error message (if failed)
            - elapsed_time: Time elapsed since task start
        """
        # try:
        start_time = time.time()
        conversation_id = self._shared_state.get("CONVERSATION_ID", None)
        if conversation_id is None:
            raise ValueError("Conversation ID not found")
        time.sleep(3) # Wait for the db to refresh

        status = ""
        # Make request to check task status
        while status != "completed" and time.time() < start_time + timeout:
            task_data = load_task_data(conversation_id, self.task_id)

            if task_data is None:
                return {
                    "status": "not_found",
                    "error": "Task not found"
                }

            status = task_data["status"]

            # Return error information if task failed
            if status == "error":
                # delete_task(conversation_id, self.task_id)
                return {
                    "status": "error",
                    "error": task_data["error"],
                }
            time.sleep(3)
        
        # Return a simplified version if the task is still processing
        if status == "processing":
            return {
                "status": "processing",
                "message": ("Task is still being processed. "
                            "Use the CheckProgress tool again and wait for the task to complete.")
            }
        
        # Return the complete result if task is completed
        if status == "completed":
            # delete_task(conversation_id, self.task_id)
            return {
                "status": "completed",
                "result": task_data["result"],
            }

if __name__ == "__main__":
    tool = CheckProgress(task_id="ab60e06c_aa17_4228_a591_17af6ae3fafb")
    tool._shared_state.set("CONVERSATION_ID", "91G8t6AHbRV7sI1RoqYr")
    print(tool.run())
