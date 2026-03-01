"""
ORACLE — Base Agent Framework
Abstract base class for all ORACLE agents with shared infrastructure.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from src.core.models import AgentMessage, AgentStatus

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base agent providing:
    - State management
    - Message passing
    - Logging and metrics
    - MCP client connectivity
    - Error handling and retries
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self.messages_in: list[AgentMessage] = []
        self.messages_out: list[AgentMessage] = []
        self.execution_log: list[dict] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.result: Optional[dict] = None
        self._error: Optional[str] = None
        logger.info(f"Agent [{self.name}] initialized: {self.description}")

    @abstractmethod
    def execute(self, input_data: dict) -> dict:
        """Execute the agent's primary task. Must be implemented by subclasses."""
        pass

    def run(self, input_data: dict, max_retries: int = 3) -> dict:
        """Run the agent with error handling and retries."""
        self.status = AgentStatus.RUNNING
        self.start_time = time.time()
        self._log("STARTED", f"Agent beginning execution with input keys: {list(input_data.keys())}")

        for attempt in range(max_retries):
            try:
                self.result = self.execute(input_data)
                self.status = AgentStatus.COMPLETED
                self.end_time = time.time()
                duration = self.end_time - self.start_time
                self._log("COMPLETED", f"Execution completed in {duration:.2f}s")

                return {
                    "agent": self.name,
                    "status": "success",
                    "duration_seconds": round(duration, 2),
                    "result": self.result,
                    "execution_log": self.execution_log,
                }

            except Exception as e:
                self._error = str(e)
                self._log("ERROR", f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    self.status = AgentStatus.FAILED
                    self.end_time = time.time()
                    return {
                        "agent": self.name,
                        "status": "failed",
                        "error": str(e),
                        "duration_seconds": round(self.end_time - self.start_time, 2),
                        "execution_log": self.execution_log,
                    }

    def send_message(self, receiver: str, content: dict, msg_type: str = "data"):
        """Send a message to another agent."""
        msg = AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=msg_type,
        )
        self.messages_out.append(msg)
        self._log("MSG_SENT", f"Message sent to {receiver}: {msg_type}")
        return msg

    def receive_message(self, message: AgentMessage):
        """Receive a message from another agent."""
        self.messages_in.append(message)
        self._log("MSG_RECEIVED", f"Message from {message.sender}: {message.message_type}")

    def _log(self, event: str, message: str):
        """Log an agent event."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "event": event,
            "message": message,
        }
        self.execution_log.append(entry)
        logger.info(f"[{self.name}] {event}: {message}")

    def get_status(self) -> dict:
        """Get current agent status."""
        duration = None
        if self.start_time:
            end = self.end_time or time.time()
            duration = round(end - self.start_time, 2)

        return {
            "agent": self.name,
            "status": self.status.value,
            "duration_seconds": duration,
            "messages_sent": len(self.messages_out),
            "messages_received": len(self.messages_in),
            "error": self._error,
        }

    def reset(self):
        """Reset agent state for a new run."""
        self.status = AgentStatus.IDLE
        self.messages_in.clear()
        self.messages_out.clear()
        self.execution_log.clear()
        self.start_time = None
        self.end_time = None
        self.result = None
        self._error = None
