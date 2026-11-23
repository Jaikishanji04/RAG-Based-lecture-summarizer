/**
 * TypeScript type definitions for StudyMatrix RAG Agent & Lecture features.
 */

export interface AgentStep {
  step_number: number;
  step_type: 'thought' | 'tool_call' | 'tool_result' | 'final_answer' | 'error';
  tool_name?: string;
  tool_args?: Record<string, any>;
  tool_output?: string;
  content?: string;
}

export interface AgentResponse {
  goal: string;
  steps: AgentStep[];
  final_answer: string;
}

export interface AgentStatus {
  huggingface: {
    token_configured: boolean;
    chat_model: string;
    embed_model: string;
  };
  gmail: {
    credentials_present: boolean;
    credentials_file: string;
    authenticated: boolean;
    scopes: string[];
    status: 'connected' | 'credentials_found' | 'missing_credentials';
  };
}

export interface ToolDefinition {
  type: string;
  function: {
    name: string;
    description: string;
    parameters: Record<string, any>;
  };
}

export interface Flashcard {
  front: string;
  back: string;
}

export interface FlashcardDeck {
  title: string;
  cards: Flashcard[];
}
