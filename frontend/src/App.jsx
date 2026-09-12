import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_URL = import.meta.env.DEV ? "http://127.0.0.1:8000" : "";

function detectIsUrdu(text) {
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

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

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
        { role: "assistant", text: `Something went wrong: ${err.message}`, isError: true },
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

  const suggestions = [
    "How do I register for a digital ID?",
    "پیدائش کے اندراج کے لیے کون سے دستاویزات درکار ہیں؟",
    "How can children stay safe online?",
  ];

  return (
    <div className="app-shell">
      <nav className="topnav">
        <div className="topnav-brand">
          <svg viewBox="0 0 48 48" width="22" height="22" aria-hidden="true">
            <circle cx="24" cy="24" r="21" fill="none" stroke="currentColor" strokeWidth="2.5" />
            <path d="M24 12 L27 21 L36 21 L29 27 L32 36 L24 30 L16 36 L19 27 L12 21 L21 21 Z"
              fill="currentColor" />
          </svg>
          <span>Citizen Services Assistant</span>
        </div>
        <a
          className="topnav-link"
          href="https://github.com/sidrayounas123/citizen-services-multilingual-rag-assistant"
          target="_blank" rel="noreferrer"
        >
          View source ↗
        </a>
      </nav>

      <section className="hero">
        <div className="hero-inner">
          <p className="hero-eyebrow">Digital ID · Registration · Online safety</p>
          <h1>Ask your citizen-services question — in English or اردو.</h1>
          <p className="hero-sub">
            A bilingual assistant built on NADRA's public guidance documents, retrieving and
            citing the exact source page behind every answer.
          </p>
        </div>
      </section>

      <main className="portal">
        <div className="thread">
          {messages.length === 0 && (
            <div className="thread-empty">
              <p className="intro-lead">Try one of these, or type your own question below.</p>
              <div className="suggestions">
                {suggestions.map((s, i) => (
                  <button
                    key={i}
                    className="suggestion-chip"
                    dir={detectIsUrdu(s) ? "rtl" : "ltr"}
                    onClick={() => setInput(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`row ${msg.role}`}>
              {msg.role === "user" ? (
                <div className="user-bubble" dir={detectIsUrdu(msg.text) ? "rtl" : "ltr"}>
                  {msg.text}
                </div>
              ) : (
                <div className={`answer-card ${msg.isError ? "is-error" : ""}`}>
                  <div className="answer-text" dir={detectIsUrdu(msg.text) ? "rtl" : "ltr"}>
                    {msg.text}
                  </div>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="reference-row">
                      {msg.sources.map((s, j) => (
                        <span className="reference-tag" key={j}>{s}</span>
                      ))}
                      {msg.latency != null && (
                        <span className="latency-tag">{msg.latency}ms</span>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="row assistant">
              <div className="answer-card is-loading">
                <span className="dot" /><span className="dot" /><span className="dot" />
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        <div className="composer">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your question · اپنا سوال لکھیں"
            rows={1}
          />
          <button className="send-btn" onClick={sendMessage} disabled={loading || !input.trim()}>
            Ask
          </button>
        </div>
      </main>

      <footer className="site-footer">
        <span>FastAPI · React · ChromaDB · Groq</span>
        <span>Built by Sidra Younas</span>
      </footer>
    </div>
  );
}
