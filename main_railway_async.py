import json
import os
import uuid
from threading import Thread

from agency_swarm.tools import ToolFactory
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from google.api_core import exceptions

import firebase_admin
from firebase_admin import credentials, firestore, initialize_app

load_dotenv()

# Authenticate on firebase
service_account_key = json.loads(os.getenv("SERVICE_ACCOUNT_KEY_FIREBASE"))

if not firebase_admin._apps:
    cred = credentials.Certificate(service_account_key)
    default_app = initialize_app(cred)

app = Flask(__name__)

db_token = os.getenv("DB_TOKEN")

db = firestore.client()

THREAD_TIMEOUT = int(os.getenv("THREAD_TIMEOUT"))


def save_shared_state(conversation_id, key, value):
    # Update value if it exists, otherwise create it
    try:
        db.collection("agencii-chats").document(conversation_id).update({key: value})
    except exceptions.NotFound:
        db.collection("agencii-chats").document(conversation_id).set(
            {key: value}, merge=True
        )


def load_shared_state(conversation_id, key):
    doc = db.collection("agencii-chats").document(conversation_id).get()

    if doc.exists:
        if key in doc.to_dict():
            return doc.to_dict()[key]

    return None


# Modify the run_tool_async function
def run_tool_async(tool, conversation_id, task_id):
    try:
        # Create a local copy of shared state to avoid race conditions
        shared_state = {}
        for key in ["KEYWORDS", "GENERATED_AD_COPY", "URL", "SEARCH_DATA"]:
            shared_state[key] = load_shared_state(conversation_id, key)
            tool._shared_state.set(key, shared_state[key])

        initial_keys = tool._shared_state.data.copy()

        response = tool.run()

        out_keys = tool._shared_state.data.copy()

        # Save changed values in shared state
        if initial_keys != out_keys:
            for key in ["KEYWORDS", "GENERATED_AD_COPY", "URL", "SEARCH_DATA"]:
                initial_value = initial_keys[key]
                out_value = out_keys[key]
                if out_value is not None and initial_value != out_value:
                    save_shared_state(conversation_id, key, out_value)

        # Save task status in shared state
        save_shared_state(
            conversation_id,
            f"task_{task_id}",
            {
                "status": "completed",
                "result": response,
            },
        )
    except Exception as e:
        save_shared_state(
            conversation_id,
            f"task_{task_id}",
            {
                "status": "error",
                "error": str(e),
            },
        )


# Modify the create_async_endpoint function
def create_async_endpoint(route, tool_class):
    @app.route(f"{route}", methods=["POST"], endpoint=f"{tool_class.__name__}")
    def async_endpoint():
        print(f"Async endpoint {route} called")
        token = request.headers.get("Authorization").split("Bearer ")[1]
        if token != db_token:
            return jsonify({"message": "Unauthorized"}), 401

        try:
            tool = tool_class(**request.get_json())
            conversation_id = request.headers.get("X-Chat-Id")
            tool._shared_state.set("CONVERSATION_ID", conversation_id)

            task_id = str(uuid.uuid4()).replace("-", "_")

            # Save initial task status
            save_shared_state(
                conversation_id,
                f"task_{task_id}",
                {"status": "processing",},
            )

            thread = Thread(
                target=run_tool_async, args=(tool, conversation_id, task_id)
            )
            thread.start()

            return jsonify(
                {
                    "status": "processing",
                    "task_id": task_id,
                    "message": (
                        "Request is being processed asynchronously. "
                        "Use CheckProgress tool to poll for updates."
                    ),
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)})


def create_endpoint(route, tool_class):
    @app.route(route, methods=["POST"], endpoint=tool_class.__name__)
    def endpoint():
        print(f"Endpoint {route} called")  # Debug print
        token = request.headers.get("Authorization").split("Bearer ")[1]
        if token != db_token:
            return jsonify({"message": "Unauthorized"}), 401

        try:
            tool = tool_class(**request.get_json())
            conversation_id = request.headers.get("X-Chat-Id")
            tool._shared_state.set("CONVERSATION_ID", conversation_id)
            for key in ["KEYWORDS", "GENERATED_AD_COPY", "URL", "SEARCH_DATA"]:
                tool._shared_state.set(key, load_shared_state(conversation_id, key))

            initial_keys = tool._shared_state.data.copy()
            response = tool.run()
            out_keys = tool._shared_state.data.copy()

            # Save changed values in shared state
            if initial_keys != out_keys:
                for key in ["KEYWORDS", "GENERATED_AD_COPY", "URL", "SEARCH_DATA"]:
                    initial_value = initial_keys[key]
                    out_value = out_keys[key]
                    if out_value is not None and initial_value != out_value:
                        save_shared_state(conversation_id, key, out_value)

            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"Error": str(e)})


def parse_all_tools():
    tools_folder = "./tools"
    tools_dict = {}
    for root, dirs, files in os.walk(tools_folder):
        # Skip util folders
        if "util" in root:
            continue

        relative_path = os.path.relpath(root, tools_folder)
        folder = relative_path if relative_path != "." else "root"
        for filename in files:
            if filename.endswith(".py"):
                tool_path = os.path.join(root, filename)
                tool_class = ToolFactory.from_file(tool_path)
                tools_dict.setdefault(folder, []).append(tool_class)
    return tools_dict


# create endpoints for each file in ./tools
tools = parse_all_tools()
tools = [tool for tool_list in tools.values() for tool in tool_list]
print(f"Tools found: {tools}")  # Debug print

for tool in tools:
    route = f"/{tool.__name__}"
    print(f"Creating endpoint for {route}")  # Debug print
    if tool.__name__ in ["ExpandKeywords", "KeywordSearch", "GenerateAdCopy"]:
        create_async_endpoint(route, tool)
    else:
        create_endpoint(route, tool)


@app.route("/", methods=["POST"])
def tools_handler():
    print("tools_handler called")  # Debug print
    print(request.headers)  # Debug print
    try:
        token = request.headers.get("Authorization").split("Bearer ")[1]
    except Exception:
        return jsonify({"message": "Unauthorized"}), 401

    if token != db_token:
        return jsonify({"message": "Unauthorized"}), 401

    with app.request_context(request.environ):
        return app.full_dispatch_request()


if __name__ == "__main__":
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(
            {
                "endpoint": rule.endpoint,
                "methods": list(rule.methods),
                "route": str(rule),
            }
        )
    print(routes)
    app.run(port=os.getenv("PORT", default=5000))
