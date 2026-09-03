import { useState, useRef, useEffect } from "react";
import "./App.css";

// In dev, Vite (port 5173) and FastAPI (port 8000) run separately.
// In production, FastAPI serves this frontend directly, so relative paths work.
const API_URL = import.meta.env.DEV ? "http://127.0.0.1:8000" : "";

function detectIsUrdu(text) {
  // Urdu/Arabic unicode block check - used to apply RTL styling per message
  return /[\u0600-\u06FF]/.test(text);
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.answer,
          sources: data.sources,
          latency: data.latency_ms,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Error: ${err.message}`, isError: true },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>Citizen Services Assistant</h1>
        <p>Ask in English or Urdu · اردو یا انگریزی میں پوچھیں</p>
      </header>

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-state">
            Ask a question about digital ID, birth/death registration, or online safety.
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div
              className={`bubble ${msg.isError ? "error" : ""}`}
              dir={detectIsUrdu(msg.text) ? "rtl" : "ltr"}
            >
              {msg.text}
            </div>
            {msg.sources && msg.sources.length > 0 && (
              <div className="sources">
                Sources: {msg.sources.join(", ")}
                {msg.latency != null && (
                  <span className="latency"> · {msg.latency}ms</span>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <div className="bubble typing">Thinking...</div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="input-row">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your question... / اپنا سوال لکھیں..."
          rows={1}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
