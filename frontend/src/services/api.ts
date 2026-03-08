/**
 * ECommerce-IA — Service API
 *
 * Client Axios pour communiquer avec le backend FastAPI.
 * Gère l'authentification JWT, la classification, les recommandations.
 */

import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_URL,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

// ---- Intercepteur JWT ----
api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("auth_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401 && typeof window !== "undefined") {
      localStorage.removeItem("auth_token");
    }
    return Promise.reject(err);
  }
);

// ============================================
// Types
// ============================================
export interface ClassificationResult {
  categorie: string;
  confiance: number;
  top5: { categorie: string; confiance: number }[];
  temps_inference: number;
}

export interface Product {
  id: number;
  nom: string;
  description: string;
  prix: number;
  categorie: string;
  image_url: string;
  note_moyenne: number;
}

export interface Recommendation {
  produit: Product;
  score: number;
  raison: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  sources?: string[];
  nlp?: {
    intent: string;
    intent_confidence: number;
    sentiment: string;
    sentiment_score: number;
    entities: { type: string; value: string }[];
    keywords: string[];
  };
  confiance?: number;
  escalade_humain?: boolean;
}

export interface NLPAnalysis {
  intent: string;
  intent_confidence: number;
  sentiment: string;
  sentiment_score: number;
  entities: { type: string; value: string }[];
  keywords: string[];
  routing: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user_id: number;
  username: string;
}

// ============================================
// Auth
// ============================================
export async function login(username: string, password: string): Promise<AuthResponse> {
  const formData = new URLSearchParams();
  formData.append("username", username);
  formData.append("password", password);
  const { data } = await api.post<AuthResponse>("/auth/token", formData, {
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  });
  localStorage.setItem("auth_token", data.access_token);
  return data;
}

export async function register(
  username: string,
  email: string,
  password: string
): Promise<AuthResponse> {
  const { data } = await api.post<AuthResponse>("/auth/register", {
    username,
    email,
    password,
  });
  localStorage.setItem("auth_token", data.access_token);
  return data;
}

export function logout() {
  localStorage.removeItem("auth_token");
}

// ============================================
// Classification
// ============================================
export async function classifyImage(file: File): Promise<ClassificationResult> {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await api.post<ClassificationResult>("/classify", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 60000,
  });
  return data;
}

export async function classifyBatch(files: File[]): Promise<ClassificationResult[]> {
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));
  const { data } = await api.post<ClassificationResult[]>("/classify/batch", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 120000,
  });
  return data;
}

// ============================================
// Recommandations
// ============================================
export async function getRecommendations(
  userId: number,
  topN: number = 10
): Promise<Recommendation[]> {
  const { data } = await api.get<Recommendation[]>(`/recommend/${userId}`, {
    params: { top_n: topN },
  });
  return data;
}

export async function getSimilarProducts(
  productId: number,
  topN: number = 8
): Promise<Product[]> {
  const { data } = await api.get<Product[]>(`/recommend/similar/${productId}`, {
    params: { top_n: topN },
  });
  return data;
}

export async function submitFeedback(
  userId: number,
  productId: number,
  rating: number
): Promise<void> {
  await api.post("/feedback", { user_id: userId, product_id: productId, rating });
}

// ============================================
// Produits
// ============================================
export async function getProducts(params?: {
  categorie?: string;
  prix_min?: number;
  prix_max?: number;
  page?: number;
  limit?: number;
}): Promise<Product[]> {
  const { data } = await api.get<Product[]>("/products", { params });
  return data;
}

export async function getProduct(id: number): Promise<Product> {
  const { data } = await api.get<Product>(`/products/${id}`);
  return data;
}

export async function searchProducts(query: string): Promise<Product[]> {
  const { data } = await api.get<Product[]>("/products/search", {
    params: { q: query },
  });
  return data;
}

// ============================================
// Chat
// ============================================
export async function sendChatMessage(
  message: string,
  sessionId?: string
): Promise<{
  reponse: string;
  sources: string[];
  session_id: string;
  nlp?: ChatMessage["nlp"];
  confiance?: number;
  escalade_humain?: boolean;
}> {
  const { data } = await api.post("/chat", {
    message,
    session_id: sessionId,
  });
  return data;
}

export async function getChatHistory(
  sessionId: string
): Promise<ChatMessage[]> {
  const { data } = await api.get<ChatMessage[]>(`/chat/history/${sessionId}`);
  return data;
}

// ============================================
// WebSocket Chat
// ============================================
export function createChatWebSocket(
  sessionId: string,
  onMessage: (msg: ChatMessage) => void,
  onError?: (err: Event) => void
): WebSocket {
  const wsUrl =
    process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
  const ws = new WebSocket(`${wsUrl}/ws/chat/${sessionId}`);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch {
      onMessage({ role: "assistant", content: event.data });
    }
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    onError?.(err);
  };

  return ws;
}

// ============================================
// NLP
// ============================================
export async function analyzeText(text: string): Promise<NLPAnalysis> {
  const { data } = await api.post<NLPAnalysis>("/nlp/analyze", { text });
  return data;
}

export async function detectIntent(
  text: string
): Promise<{ intent: string; confidence: number }> {
  const { data } = await api.post("/nlp/intent", { text });
  return data;
}

export async function extractEntities(
  text: string
): Promise<{ type: string; value: string }[]> {
  const { data } = await api.post("/nlp/entities", { text });
  return data;
}

export async function analyzeSentiment(
  text: string
): Promise<{ label: string; score: number }> {
  const { data } = await api.post("/nlp/sentiment", { text });
  return data;
}

export default api;
