"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { sendChatMessage, type ChatMessage } from "@/services/api";
import { FiSend, FiMessageCircle, FiUser, FiCpu, FiTrash2 } from "react-icons/fi";
import { motion } from "framer-motion";
import toast from "react-hot-toast";

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>(() =>
    typeof window !== "undefined"
      ? `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
      : "default"
  );
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;

    const userMsg: ChatMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await sendChatMessage(text, sessionId);
      if (response.session_id) setSessionId(response.session_id);
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: response.reponse,
        sources: response.sources,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      console.error(err);
      toast.error("Erreur de communication avec le chatbot");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Désolé, une erreur est survenue. Veuillez réessayer.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setSessionId(
      `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
    );
    toast.success("Conversation réinitialisée");
  };

  const suggestions = [
    "Quels sont vos délais de livraison ?",
    "Comment retourner un article ?",
    "Quels moyens de paiement acceptez-vous ?",
    "Recommandez-moi des chaussures de sport",
  ];

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-3 mb-3">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center text-white">
            <FiMessageCircle className="w-6 h-6" />
          </div>
          <div className="text-left">
            <h1 className="text-2xl font-bold text-slate-900">
              Assistant IA
            </h1>
            <p className="text-sm text-slate-500">
              Chatbot RAG propulsé par Mistral-7B
            </p>
          </div>
        </div>
      </div>

      {/* Chat container */}
      <div className="card p-0 overflow-hidden flex flex-col" style={{ height: "65vh" }}>
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center gap-6">
              <FiCpu className="w-16 h-16 text-slate-200" />
              <div>
                <p className="text-lg font-semibold text-slate-600 mb-2">
                  Comment puis-je vous aider ?
                </p>
                <p className="text-sm text-slate-400 mb-6">
                  Posez une question sur nos produits, services ou politique.
                </p>
                <div className="flex flex-wrap justify-center gap-2">
                  {suggestions.map((s) => (
                    <button
                      key={s}
                      onClick={() => setInput(s)}
                      className="text-xs px-3 py-1.5 rounded-full border border-slate-200 text-slate-500 hover:bg-primary-50 hover:text-primary-600 hover:border-primary-200 transition-all"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex gap-3 ${
                msg.role === "user" ? "flex-row-reverse" : ""
              }`}
            >
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  msg.role === "user"
                    ? "bg-primary-100 text-primary-600"
                    : "bg-slate-100 text-slate-500"
                }`}
              >
                {msg.role === "user" ? (
                  <FiUser className="w-4 h-4" />
                ) : (
                  <FiCpu className="w-4 h-4" />
                )}
              </div>
              <div
                className={`max-w-[75%] rounded-2xl px-4 py-3 ${
                  msg.role === "user"
                    ? "bg-primary-600 text-white rounded-tr-md"
                    : "bg-slate-100 text-slate-800 rounded-tl-md"
                }`}
              >
                <p className="text-sm whitespace-pre-wrap leading-relaxed">
                  {msg.content}
                </p>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-slate-200/50">
                    <p className="text-[10px] text-slate-400 uppercase tracking-wider mb-1">
                      Sources
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {msg.sources.map((src, j) => (
                        <span
                          key={j}
                          className="text-[10px] px-2 py-0.5 rounded-full bg-white/20 text-slate-500"
                        >
                          {src}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          ))}

          {isLoading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center">
                <FiCpu className="w-4 h-4 text-slate-500" />
              </div>
              <div className="bg-slate-100 rounded-2xl rounded-tl-md px-4 py-3">
                <div className="flex gap-1.5">
                  <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" />
                  <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce [animation-delay:150ms]" />
                  <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce [animation-delay:300ms]" />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-slate-100 p-4 bg-white">
          <form onSubmit={handleSend} className="flex gap-3">
            {messages.length > 0 && (
              <button
                type="button"
                onClick={handleClearChat}
                className="p-3 rounded-xl text-slate-400 hover:text-red-500 hover:bg-red-50 transition-colors"
                title="Nouvelle conversation"
              >
                <FiTrash2 className="w-5 h-5" />
              </button>
            )}
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Tapez votre message..."
              className="input-field flex-1"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="btn-primary px-4 disabled:opacity-40"
            >
              <FiSend className="w-5 h-5" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
