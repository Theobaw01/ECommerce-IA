"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { sendChatMessage, type ChatMessage } from "@/services/api";
import { FiSend, FiMessageCircle, FiUser, FiCpu, FiTrash2, FiTarget, FiSmile, FiHash, FiAlertTriangle } from "react-icons/fi";
import { motion } from "framer-motion";
import toast from "react-hot-toast";

interface NLPInfo {
  intent: string;
  intent_confidence: number;
  sentiment: string;
  sentiment_score: number;
  entities: { type: string; value: string }[];
  keywords: string[];
}

interface ChatMessageWithNLP extends ChatMessage {
  nlp?: NLPInfo;
  confiance?: number;
  escalade_humain?: boolean;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessageWithNLP[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showNLP, setShowNLP] = useState(true);
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

    const userMsg: ChatMessageWithNLP = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await sendChatMessage(text, sessionId);
      if (response.session_id) setSessionId(response.session_id);
      const assistantMsg: ChatMessageWithNLP = {
        role: "assistant",
        content: response.reponse,
        sources: response.sources,
        nlp: response.nlp,
        confiance: response.confiance,
        escalade_humain: response.escalade_humain,
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

  const sentimentColor = (label: string) => {
    if (label === "positif") return "text-green-600 bg-green-50";
    if (label === "négatif") return "text-red-600 bg-red-50";
    return "text-slate-500 bg-slate-50";
  };

  const sentimentEmoji = (label: string) => {
    if (label === "positif") return "😊";
    if (label === "négatif") return "😟";
    return "😐";
  };

  const suggestions = [
    "Quels sont vos délais de livraison ?",
    "Comment retourner un article ?",
    "Quels moyens de paiement acceptez-vous ?",
    "Recommandez-moi des chaussures de sport",
    "Je cherche un produit à moins de 50€",
    "Où en est ma commande CMD-2024-001 ?",
  ];

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
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
              Chatbot RAG + NLP — LangChain + ChromaDB + Moteur NLP
            </p>
          </div>
        </div>
        {/* Toggle NLP */}
        <button
          onClick={() => setShowNLP(!showNLP)}
          className={`mt-2 text-xs px-3 py-1 rounded-full transition-all ${
            showNLP 
              ? "bg-orange-100 text-orange-700 border border-orange-200" 
              : "bg-slate-100 text-slate-500 border border-slate-200"
          }`}
        >
          {showNLP ? "🧠 NLP Activé" : "NLP Désactivé"}
        </button>
      </div>

      {/* Chat container */}
      <div className="card p-0 overflow-hidden flex flex-col" style={{ height: "68vh" }}>
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center gap-6">
              <FiCpu className="w-16 h-16 text-slate-200" />
              <div>
                <p className="text-lg font-semibold text-slate-600 mb-2">
                  Comment puis-je vous aider ?
                </p>
                <p className="text-sm text-slate-400 mb-2">
                  Pipeline : NLP (intent + NER + sentiment) → RAG (ChromaDB) → Génération
                </p>
                <p className="text-xs text-slate-300 mb-6">
                  Chaque message est analysé par le moteur NLP avant le retrieval
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
              <div className={`max-w-[80%] ${msg.role === "user" ? "text-right" : ""}`}>
                <div
                  className={`rounded-2xl px-4 py-3 ${
                    msg.role === "user"
                      ? "bg-primary-600 text-white rounded-tr-md"
                      : "bg-slate-100 text-slate-800 rounded-tl-md"
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap leading-relaxed">
                    {msg.content}
                  </p>

                  {/* Sources RAG */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-slate-200/50">
                      <p className="text-[10px] text-slate-400 uppercase tracking-wider mb-1">
                        Sources RAG
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {msg.sources.map((src, j) => (
                          <span
                            key={j}
                            className="text-[10px] px-2 py-0.5 rounded-full bg-white/20 text-slate-500"
                          >
                            {src.slice(0, 60)}...
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* NLP Info Panel */}
                {showNLP && msg.role === "assistant" && msg.nlp && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    className="mt-2 rounded-xl bg-white border border-slate-200 px-3 py-2 text-left"
                  >
                    <p className="text-[10px] text-slate-400 uppercase tracking-wider mb-2 font-medium">
                      🧠 Analyse NLP
                    </p>
                    <div className="flex flex-wrap gap-2 text-xs">
                      {/* Intent */}
                      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-blue-50 text-blue-700">
                        <FiTarget className="w-3 h-3" />
                        {msg.nlp.intent}
                        <span className="text-blue-400 text-[10px]">
                          {(msg.nlp.intent_confidence * 100).toFixed(0)}%
                        </span>
                      </span>

                      {/* Sentiment */}
                      <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-md ${sentimentColor(msg.nlp.sentiment)}`}>
                        <FiSmile className="w-3 h-3" />
                        {sentimentEmoji(msg.nlp.sentiment)} {msg.nlp.sentiment}
                        <span className="text-[10px] opacity-60">
                          ({msg.nlp.sentiment_score > 0 ? "+" : ""}{msg.nlp.sentiment_score.toFixed(2)})
                        </span>
                      </span>

                      {/* Confiance */}
                      {msg.confiance !== undefined && (
                        <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-md ${
                          msg.confiance > 0.7 ? "bg-green-50 text-green-700" : 
                          msg.confiance > 0.4 ? "bg-yellow-50 text-yellow-700" : 
                          "bg-red-50 text-red-700"
                        }`}>
                          Confiance: {(msg.confiance * 100).toFixed(0)}%
                        </span>
                      )}

                      {/* Escalade */}
                      {msg.escalade_humain && (
                        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-red-50 text-red-600">
                          <FiAlertTriangle className="w-3 h-3" />
                          Escalade
                        </span>
                      )}
                    </div>

                    {/* Entités */}
                    {msg.nlp.entities && msg.nlp.entities.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {msg.nlp.entities.map((ent, j) => (
                          <span
                            key={j}
                            className="text-[10px] px-2 py-0.5 rounded-full bg-purple-50 text-purple-600 border border-purple-100"
                          >
                            <FiHash className="w-2.5 h-2.5 inline mr-0.5" />
                            {ent.type}: {ent.value}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Keywords */}
                    {msg.nlp.keywords && msg.nlp.keywords.length > 0 && (
                      <div className="mt-1.5 text-[10px] text-slate-400">
                        Mots-clés : {msg.nlp.keywords.join(", ")}
                      </div>
                    )}
                  </motion.div>
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
          <p className="text-[10px] text-slate-300 mt-2 text-center">
            Pipeline : Prétraitement → Détection d&apos;Intent (15 classes) → NER → Sentiment → RAG (ChromaDB) → Génération
          </p>
        </div>
      </div>
    </div>
  );
}
