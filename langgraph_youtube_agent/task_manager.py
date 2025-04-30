import json
import logging
import asyncio
from typing import AsyncIterable, List, Tuple

# Assuming 'common' library is available via install or copied path
try:
    from common.server.task_manager import InMemoryTaskManager
    from common.server import utils
    from common.types import (
        Artifact, DataPart, TextPart, JSONRPCResponse, SendTaskRequest,
        SendTaskResponse, SendTaskStreamingRequest, SendTaskStreamingResponse,
        TaskSendParams, TaskState, TaskStatus, Message, TaskStatusUpdateEvent,
        TaskArtifactUpdateEvent, InternalError
    )
except ImportError:
     raise ImportError("Could not import A2A common library. Ensure 'a2a-samples' is installed or 'common' directory is in PYTHONPATH.")

from .agent import YouTubeVideoAgent

logger = logging.getLogger(__name__)

class AgentTaskManager(InMemoryTaskManager):
    """Task manager for the LangGraph YouTube Agent."""

    def __init__(self, agent: YouTubeVideoAgent):
        super().__init__()
        self.agent = agent

    def _parse_agent_result(self, result_content: str) -> tuple[TaskState, List[Artifact]]:
        """Parses the agent's JSON string result into A2A TaskStatus and Artifacts."""
        try:
            # Result should be a JSON string list of video IDs or an error message list
            data = json.loads(result_content)
            if isinstance(data, list):
                # Check if it's an error message list
                if data and isinstance(data[0], str) and data[0].startswith("Error:"):
                     # Return the error message as text part in a FAILED task status
                     parts = [TextPart(text=data[0])]
                     task_state = TaskState.FAILED
                     # Artifacts list is usually None for failure, but we include the error text here for clarity in the message
                     artifacts = [Artifact(parts=parts)]
                else:
                    # Success - list of video IDs
                    # Create DataPart artifact for structured data exchange
                    parts = [DataPart(data=data)]
                    task_state = TaskState.COMPLETED
                    artifacts = [Artifact(parts=parts)]
            else:
                # Parsed as JSON but not a list, treat as error/unexpected text
                parts = [TextPart(text=f"Error: Agent returned unexpected JSON type: {type(data).__name__}")]
                task_state = TaskState.FAILED
                logger.warning(f"Agent returned non-list JSON: {result_content}")
                artifacts = [Artifact(parts=parts)]
        except json.JSONDecodeError:
            # Not JSON, likely a direct error message or unexpected output
            parts = [TextPart(text=result_content)]
            # Assume FAILED if not JSON, as the agent is instructed to return JSON list or JSON error list
            task_state = TaskState.FAILED
            logger.warning(f"Agent returned non-JSON string: {result_content}")
            artifacts = [Artifact(parts=parts)]
        except Exception as e:
            logger.error(f"Unexpected error parsing agent result: {e}", exc_info=True)
            parts = [TextPart(text=f"Internal error processing agent response: {type(e).__name__}")]
            return TaskState.FAILED, [Artifact(parts=parts)]

        return task_state, artifacts

    async def _stream_generator(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse]:
        """Handles streaming responses from the LangGraph agent."""
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)
        final_result_content = None
        final_task_state = TaskState.FAILED # Default to failed unless explicitly completed
        final_artifacts = None

        try:
            async for item in self.agent.stream(query, task_send_params.sessionId):
                is_complete = item.get("is_task_complete", False)
                content = item.get("content", "Processing...")
                final_result_content = content # Store latest content

                # Yield 'working' status updates
                if not is_complete:
                    task_status = TaskStatus(
                        state=TaskState.WORKING,
                        message=Message(role="agent", parts=[TextPart(text=str(content))])
                    )
                    yield SendTaskStreamingResponse(
                        id=request.id,
                        result=TaskStatusUpdateEvent(id=task_send_params.id, status=task_status, final=False)
                    )
                else:
                    # Final content received from agent stream
                    break

            # Process the final result
            if final_result_content is not None:
                 final_task_state, final_artifacts = self._parse_agent_result(str(final_result_content))
            else:
                 # If stream finished without final content
                 final_task_state = TaskState.FAILED
                 final_artifacts = [Artifact(parts=[TextPart(text="Error: Agent stream ended without final output.")])]

        except Exception as e:
            logger.error(f"An error occurred while streaming the response: {e}", exc_info=True)
            final_task_state = TaskState.FAILED
            fail_message_str = f"Stream failed: {type(e).__name__}"
            final_artifacts = [Artifact(parts=[TextPart(text=fail_message_str)])]
            # Yield internal error response for JSON-RPC level
            yield JSONRPCResponse(
                 id=request.id,
                 error=InternalError(message=f"An error occurred while streaming the response: {e}")
            )
        finally:
            # Update store and send final events
            final_status = TaskStatus(state=final_task_state)
            # Add error message to status ONLY if FAILED state
            if final_task_state == TaskState.FAILED and final_artifacts and final_artifacts[0].parts[0].type == "text":
                 final_status.message = Message(role="agent", parts=final_artifacts[0].parts)

            # Update store with final status. Send artifacts only if COMPLETED.
            await self.update_store(task_send_params.id, final_status, final_artifacts if final_task_state == TaskState.COMPLETED else None)

            # Yield final artifacts ONLY if task completed successfully
            if final_task_state == TaskState.COMPLETED and final_artifacts:
                for artifact in final_artifacts:
                    yield SendTaskStreamingResponse(
                        id=request.id,
                        result=TaskArtifactUpdateEvent(id=task_send_params.id, artifact=artifact)
                    )
            # Always yield final status update (last event)
            yield SendTaskStreamingResponse(
                id=request.id,
                result=TaskStatusUpdateEvent(id=task_send_params.id, status=final_status, final=True)
            )


    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handles non-streaming requests."""
        if not utils.are_modalities_compatible(
            request.params.acceptedOutputModes,
            self.agent.SUPPORTED_OUTPUT_TYPES,
        ):
            logger.warning("Incompatible content types requested.")
            return utils.new_incompatible_types_error(request.id)

        task_send_params: TaskSendParams = request.params
        await self.upsert_task(task_send_params)

        query = self._get_user_query(task_send_params)
        try:
            result_content = self.agent.invoke(query, task_send_params.sessionId)
            task_state, artifacts = self._parse_agent_result(str(result_content))

            task_status = TaskStatus(state=task_state)
            if task_state != TaskState.COMPLETED and artifacts and artifacts[0].parts[0].type == "text":
                task_status.message = Message(role="agent", parts=artifacts[0].parts)

            # Send artifacts only if completed successfully
            task = await self.update_store(task_send_params.id, task_status, artifacts if task_state == TaskState.COMPLETED else None)
            task_result = self.append_task_history(task, task_send_params.historyLength)
            return SendTaskResponse(id=request.id, result=task_result)
        except Exception as e:
            logger.error(f"Error invoking agent: {e}", exc_info=True)
            task_status = TaskStatus(state=TaskState.FAILED, message=Message(role="agent", parts=[TextPart(text=f"Failed: {type(e).__name__}")]))
            task = await self.update_store(task_send_params.id, task_status, None)
            return SendTaskResponse(id=request.id, result=task)


    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        """Handles streaming requests."""
        if not utils.are_modalities_compatible(
            request.params.acceptedOutputModes,
            self.agent.SUPPORTED_OUTPUT_TYPES,
        ):
            logger.warning("Incompatible content types requested.")
            return utils.new_incompatible_types_error(request.id)

        await self.upsert_task(request.params)
        sse_event_queue = await self.setup_sse_consumer(request.params.id, False)
        asyncio.create_task(self._run_streaming_agent(request))
        return self.dequeue_events_for_sse(request.id, request.params.id, sse_event_queue) # type: ignore

    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        """Extracts the user query from the request parameters."""
        if not task_send_params.message.parts:
            raise ValueError("Received task with no message parts.")
        part = task_send_params.message.parts[0]
        if isinstance(part, TextPart):
            return part.text
        else:
            logger.warning(f"Received non-TextPart as primary query: {part.type}. Using empty query.")
            return ""

    async def _run_streaming_agent(self, request: SendTaskStreamingRequest):
         """Runs the agent's stream method and puts events onto the SSE queue."""
         async for item in self._stream_generator(request):
             await self.enqueue_events_for_sse(request.params.id, item.result if hasattr(item, 'result') else item.error) # type: ignore
