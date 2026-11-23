import React, { useEffect, useState } from 'react';
import { connectGmail, getAgentStatus, streamAgent } from '../services/api';
import { AgentStatus, AgentStep } from '../types';
import './AgentPanel.css';

export const AgentPanel: React.FC = () => {
  const [goal, setGoal] = useState<string>('');
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [finalAnswer, setFinalAnswer] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const data = await getAgentStatus();
      setStatus(data);
    } catch (err: any) {
      console.warn('Backend status not available yet:', err.message);
    }
  };

  const handleConnectGmail = async () => {
    try {
      setLoading(true);
      await connectGmail();
      await fetchStatus();
    } catch (err: any) {
      setError(`Gmail connection error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRunAgent = () => {
    if (!goal.trim()) return;

    setLoading(true);
    setError(null);
    setSteps([]);
    setFinalAnswer('');

    streamAgent(
      goal,
      (step: AgentStep) => {
        setSteps((prev) => [...prev, step]);
        if (step.step_type === 'final_answer' && step.content) {
          setFinalAnswer(step.content);
        }
      },
      (err: string) => {
        setError(err);
        setLoading(false);
      },
      () => {
        setLoading(false);
      }
    );
  };

  const handleQuickPrompt = (promptText: string) => {
    setGoal(promptText);
  };

  return (
    <div className="agent-panel">
      <div className="agent-header">
        <h2>Autonomous Academic & Inbox Agent</h2>
        <p>
          Ask questions across your Gmail inbox and lecture notes. The agent decides
          which tools to invoke, sanitizes external email, and synthesizes study outputs.
        </p>

        {status && (
          <div className="status-bar">
            <span
              className={`status-badge ${
                status.huggingface.token_configured ? 'connected' : 'error'
              }`}
            >
              ● HF: {status.huggingface.token_configured ? status.huggingface.chat_model : 'Missing Token'}
            </span>

            <span
              className={`status-badge ${
                status.gmail.status === 'connected'
                  ? 'connected'
                  : status.gmail.status === 'credentials_found'
                  ? 'warning'
                  : 'neutral'
              }`}
            >
              ● Gmail MCP:{' '}
              {status.gmail.status === 'connected'
                ? 'Connected (Read-Only)'
                : status.gmail.status === 'credentials_found'
                ? 'Login Required'
                : 'No credentials.json'}
            </span>

            <span className="status-badge neutral">
              ● Embeddings: {status.huggingface.embed_model}
            </span>
          </div>
        )}
      </div>

      {status?.gmail.status === 'credentials_found' && (
        <div className="gmail-auth-banner">
          <div>
            <strong>Google OAuth Credentials Detected</strong>
            <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: '#475569' }}>
              Authorize StudyMatrix RAG with read-only Gmail access to enable inbox features.
            </p>
          </div>
          <button className="btn-auth" onClick={handleConnectGmail} disabled={loading}>
            Authorize Gmail (Read-Only)
          </button>
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      <div className="quick-prompts">
        <label>Quick Academic Actions:</label>
        <div className="prompt-chips">
          <button
            className="chip-btn"
            onClick={() =>
              handleQuickPrompt('Summarize my unread emails from today, highlighting any urgent notices.')
            }
          >
            📬 Unread Emails Today
          </button>
          <button
            className="chip-btn"
            onClick={() =>
              handleQuickPrompt('Find emails from my teacher about assignments and list all upcoming deadlines.')
            }
          >
            📝 Teacher Assignments & Deadlines
          </button>
          <button
            className="chip-btn"
            onClick={() =>
              handleQuickPrompt('Turn the key points from this email thread into study flashcards.')
            }
          >
            🃏 Thread to Flashcards
          </button>
          <button
            className="chip-btn"
            onClick={() =>
              handleQuickPrompt('Cross-reference email announcement about the midterm exam with lecture notes.')
            }
          >
            🔍 Email vs Lecture Notes
          </button>
        </div>
      </div>

      <div className="input-section">
        <textarea
          className="goal-textarea"
          rows={3}
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder="What would you like the agent to do? (e.g. Find emails from prof about assignment 2 and compare with Lecture 3 notes)"
          disabled={loading}
        />
        <button
          className="btn-run"
          onClick={handleRunAgent}
          disabled={loading || !goal.trim()}
        >
          {loading ? 'Agent Running...' : 'Execute Agent Plan'}
        </button>
      </div>

      {steps.length > 0 && (
        <div className="trace-section">
          <h3>Execution Plan & Trace</h3>
          {steps.map((step, idx) => (
            <div key={idx} className={`step-card ${step.step_type}`}>
              <div className="step-header">
                <span>
                  STEP {step.step_number}:{' '}
                  {step.step_type === 'tool_call'
                    ? `TOOL CALL (${step.tool_name})`
                    : step.step_type === 'tool_result'
                    ? `TOOL RESULT (${step.tool_name})`
                    : step.step_type === 'final_answer'
                    ? 'FINAL ANSWER'
                    : 'THOUGHT'}
                </span>
              </div>
              <div className="step-body">
                {step.content && <p style={{ margin: 0 }}>{step.content}</p>}
                {step.tool_args && (
                  <pre>{JSON.stringify(step.tool_args, null, 2)}</pre>
                )}
                {step.tool_output && (
                  <pre>{step.tool_output}</pre>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {finalAnswer && (
        <div className="final-answer-box">
          <h3>Final Output</h3>
          <div className="markdown-content" style={{ whiteSpace: 'pre-wrap' }}>
            {finalAnswer}
          </div>
        </div>
      )}
    </div>
  );
};
export default AgentPanel;
