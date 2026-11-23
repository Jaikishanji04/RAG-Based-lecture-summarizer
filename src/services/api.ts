/**
 * API service client for StudyMatrix RAG and Autonomous Agent endpoints.
 */

import { AgentResponse, AgentStatus, AgentStep, ToolDefinition } from '../types';

const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8000';

export async function getAgentStatus(): Promise<AgentStatus> {
  const response = await fetch(`${API_BASE_URL}/agent/status`);
  if (!response.ok) {
    throw new Error(`Failed to fetch agent status: ${response.statusText}`);
  }
  return response.json();
}

export async function getAgentTools(): Promise<ToolDefinition[]> {
  const response = await fetch(`${API_BASE_URL}/agent/tools`);
  if (!response.ok) {
    throw new Error(`Failed to fetch agent tools: ${response.statusText}`);
  }
  return response.json();
}

export async function runAgent(goal: string): Promise<AgentResponse> {
  const response = await fetch(`${API_BASE_URL}/agent/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ goal }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Agent run failed: ${errorText || response.statusText}`);
  }
  return response.json();
}

export async function connectGmail(): Promise<{ success: boolean; message: string }> {
  const response = await fetch(`${API_BASE_URL}/agent/auth/gmail`, {
    method: 'POST',
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gmail authorization failed: ${errorText || response.statusText}`);
  }
  return response.json();
}

/**
 * Stream agent steps in real time via Server-Sent Events.
 */
export function streamAgent(
  goal: string,
  onStep: (step: AgentStep) => void,
  onError: (error: string) => void,
  onComplete: () => void
): () => void {
  const controller = new AbortController();

  fetch(`${API_BASE_URL}/agent/run/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ goal }),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`Streaming failed: ${response.statusText}`);
      }
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) return;

      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              onStep(data);
            } catch (err) {
              console.error('Failed to parse SSE step:', err);
            }
          }
        }
      }
      onComplete();
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onError(err.message || 'Stream connection error');
      }
    });

  return () => controller.abort();
}
