import os
import json
from flask import Flask, request, jsonify
from agency_swarm.tools import ToolFactory
from dotenv import load_dotenv
from firebase_admin import initialize_app, credentials, firestore
import firebase_admin

load_dotenv()

# Authenticate on firebase
service_account_key = json.loads(os.getenv("SERVICE_ACCOUNT_KEY_FIREBASE"))

if not firebase_admin._apps:
    cred = credentials.Certificate(service_account_key) 
    default_app = initialize_app(cred)
db = firestore.client()

app = Flask(__name__)

db_token = os.getenv("DB_TOKEN")

db = firestore.client()

def save_shared_state(conversation_id, key, value):
    db.collection("agencii-chats").document(conversation_id).set({key: value}, merge=True)


def load_shared_state(conversation_id, key):
    doc = db.collection("agencii-chats").document(conversation_id).get()

    if doc.exists:
        if key in doc.to_dict():
            return doc.to_dict()[key]
        
    return None

def create_endpoint(route, tool_class):
    @app.route(route, methods=['POST'], endpoint=tool_class.__name__)
    def endpoint():
        print(f"Endpoint {route} called")  # Debug print
        token = request.headers.get("Authorization").split("Bearer ")[1]
        if token != db_token:
            return jsonify({"message": "Unauthorized"}), 401

        try:
            tool = tool_class(**request.get_json())
            conversation_id = request.headers.get("X-Chat-Id")
            for key in ["KEYWORDS", "GENERATED_AD_COPY", "URL", "SEARCH_DATA"]:
                tool._shared_state.set(key, load_shared_state(conversation_id, key))
            response = tool.run()
            for key in ["KEYWORDS", "GENERATED_AD_COPY", "URL", "SEARCH_DATA"]:
                save_shared_state(conversation_id, key, tool._shared_state.get(key, None))
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"Error": str(e)})
        
def parse_all_tools():
    tools_folder = './tools'
    tools_dict = {}
    for root, dirs, files in os.walk(tools_folder):
        # Skip util folders
        if 'util' in root:
            continue
            
        relative_path = os.path.relpath(root, tools_folder)
        folder = relative_path if relative_path != '.' else 'root'
        for filename in files:
            if filename.endswith('.py'):
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
    create_endpoint(route, tool)

@app.route("/", methods=['POST'])
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

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))