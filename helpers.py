import os
import logging
from agency_swarm.tools import ToolFactory
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_all_tools():
    tools_folder = "./tools"
    tools = []
    logger.info(f"Parsing tools from folder: {tools_folder}")
    try:
        for filename in os.listdir(tools_folder):
            if filename.endswith(".py") and filename != "constants.py":
                tool_path = os.path.join(tools_folder, filename)
                logger.debug(f"Loading tool from: {tool_path}")
                tool_class = ToolFactory.from_file(tool_path)
                tools.append(tool_class)
        logger.info(f"Successfully loaded {len(tools)} tools")
        return tools
    except Exception as e:
        logger.error(f"Error parsing tools: {str(e)}")
        raise
