import json
import logging
import asyncio
from typing import AsyncIterable, Dict, Any, Union, List

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

from .agent import SummaryAgent # Import your ADK agent

logger = logging.getLogger(__name__)

class AgentTaskManager(InMemoryTaskManager):
    """Task manager for the ADK Summary Agent."""

    def __init__(self, agent: SummaryAgent):
        super().__init__()
        self.agent = agent

    async def _stream_generator(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse]:
        """Handles streaming responses from the ADK agent."""
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)
        final_task_state = TaskState.FAILED # Default to failed unless explicitly completed
        final_artifacts: Optional[List[Artifact]] = None
        final_content: str = ""

        try:
            async for item in self.agent.stream(query, task_send_params.sessionId):
                is_complete = item.get("is_task_complete", False)
                content = item.get("content", "Processing step...")
                artifacts_payload = item.get("artifacts", None) # Extract potential artifacts
                final_content = str(content) # Keep track of the latest content as string

                current_task_state = TaskState.COMPLETED if is_complete else TaskState.WORKING
                # Check for errors signaled in content string
                if isinstance(content, str) and (content.lower().startswith("error:") or "failed" in content.lower()):
                    current_task_state = TaskState.FAILED
                    is_complete = True # Treat errors as completion

                final_status = TaskStatus(
                    state=current_task_state,
                    # Include intermediate message only if content exists
                    message=Message(role="agent", parts=[TextPart(text=str(content))]) if content else None
                )

                yield SendTaskStreamingResponse(
                    id=request.id,
                    result=TaskStatusUpdateEvent(id=task_send_params.id, status=final_status, final=is_complete)
                )

                # Handle potential artifacts if yielded by the agent stream (currently placeholder)
                if artifacts_payload:
                     try:
                         # Assuming artifacts_payload is a list of dicts or Artifact objects
                         current_artifacts = [Artifact(**a) if isinstance(a, dict) else a for a in artifacts_payload]
                         for artifact in current_artifacts:
                              if isinstance(artifact, Artifact):
                                   yield SendTaskStreamingResponse(
                                       id=request.id,
                                       result=TaskArtifactUpdateEvent(id=task_send_params.id, artifact=artifact)
                                   )
                     except Exception as artifact_err:
                          logger.error(f"Error processing artifacts in stream: {artifact_err}")


                if is_complete:
                    final_task_state = current_task_state
                    # Create final artifact only if COMPLETED successfully and content exists
                    if final_task_state == TaskState.COMPLETED and final_content:
                         final_artifacts = [Artifact(parts=[TextPart(text=final_content)])]
                    elif final_task_state == TaskState.FAILED and final_content:
                         # If failed, the error message is in the final status message, not artifact
                         final_artifacts = None
                         final_status.message = Message(role="agent", parts=[TextPart(text=final_content)])
                    else: # Handle case where task completes without content
                         final_artifacts = None
                         if final_task_state == TaskState.COMPLETED:
                             logger.warning("Task completed but no final content yielded.")
                             final_status.message = Message(role="agent", parts=[TextPart(text="Task finished.")])

                    # Update store with final status and artifacts (if successful)
                    await self.update_store(task_send_params.id, final_status, final_artifacts)
                    break # Exit loop once task is complete or failed

        except Exception as e:
            logger.error(f"An error occurred while streaming the response: {e}", exc_info=True)
            final_task_state = TaskState.FAILED
            fail_message_str = f"Stream failed: {type(e).__name__}"
            final_artifacts = None # No artifacts on internal failure
            # Yield internal error response for JSON-RPC level
            yield JSONRPCResponse(
                 id=request.id,
                 error=InternalError(message=f"An error occurred while streaming the response: {e}")
            )
            # Update store and send final failed status via SSE
            final_status = TaskStatus(state=final_task_state, message=Message(role="agent", parts=[TextPart(text=fail_message_str)]))
            await self.update_store(task_send_params.id, final_status, None)
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
            parts = [TextPart(text=str(result_content))]
            task_state = TaskState.COMPLETED
            if isinstance(result_content, str) and (result_content.lower().startswith("error:") or "failed" in result_content.lower()):
                task_state = TaskState.FAILED

            task_status = TaskStatus(state=task_state)
            artifacts = [Artifact(parts=parts)] if task_state == TaskState.COMPLETED else None
            if task_state == TaskState.FAILED:
                task_status.message = Message(role="agent", parts=parts)

            task = await self.update_store(task_send_params.id, task_status, artifacts)
            task_result = self.append_task_history(task, task_send_params.historyLength)
            return SendTaskResponse(id=request.id, result=task_result)
        except Exception as e:
            logger.error(f"Error invoking ADK agent: {e}", exc_info=True)
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

