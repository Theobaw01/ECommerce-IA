"use client";

import { useState } from "react";
import { login, register, getRecommendations, type Product, type Recommendation } from "@/services/api";
import { useAuthStore } from "@/stores/authStore";
import ProductCard from "@/components/ui/ProductCard";
import Spinner from "@/components/ui/Spinner";
import { FiUser, FiLogIn, FiLogOut, FiStar, FiMail, FiLock } from "react-icons/fi";
import { motion } from "framer-motion";
import toast from "react-hot-toast";

export default function ProfilePage() {
  const { isAuthenticated, username, userId, setAuth, clearAuth } =
    useAuthStore();
  const [isLogin, setIsLogin] = useState(true);
  const [form, setForm] = useState({ username: "", email: "", password: "" });
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [recsLoaded, setRecsLoaded] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      if (isLogin) {
        const res = await login(form.username, form.password);
        setAuth(res.access_token, res.user_id, res.username);
        toast.success(`Bienvenue, ${res.username} !`);
      } else {
        const res = await register(form.username, form.email, form.password);
        setAuth(res.access_token, res.user_id, res.username);
        toast.success("Compte créé avec succès !");
      }
    } catch (err: unknown) {
      const message = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || "Erreur d'authentification";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  const loadRecommendations = async () => {
    if (!userId) return;
    setLoading(true);
    try {
      const recs = await getRecommendations(userId, 12);
      setRecommendations(recs);
      setRecsLoaded(true);
    } catch {
      toast.error("Erreur lors du chargement des recommandations");
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    clearAuth();
    setRecommendations([]);
    setRecsLoaded(false);
    toast.success("Déconnecté");
  };

  // ---- Authenticated view ----
  if (isAuthenticated) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="card max-w-md mx-auto mb-10">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center text-white text-xl font-bold">
              {username?.[0]?.toUpperCase() || "U"}
            </div>
            <div className="flex-1">
              <h2 className="text-xl font-bold text-slate-900">{username}</h2>
              <p className="text-sm text-slate-500">ID: {userId}</p>
            </div>
            <button
              onClick={handleLogout}
              className="btn-secondary flex items-center gap-2 text-sm"
            >
              <FiLogOut className="w-4 h-4" />
              Déconnexion
            </button>
          </div>
        </div>

        {/* Recommendations */}
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-slate-900 mb-3">
            <FiStar className="w-6 h-6 inline-block text-amber-400 mr-2" />
            Vos Recommandations IA
          </h2>
          {!recsLoaded && (
            <button
              onClick={loadRecommendations}
              disabled={loading}
              className="btn-primary mt-2"
            >
              {loading ? "Chargement..." : "Charger mes recommandations"}
            </button>
          )}
        </div>

        {loading && <Spinner text="Calcul des recommandations..." />}

        {recsLoaded && recommendations.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
          >
            {recommendations.map((rec) => (
              <ProductCard
                key={rec.produit.id}
                product={rec.produit}
                score={rec.score}
                reason={rec.raison}
              />
            ))}
          </motion.div>
        )}
        {recsLoaded && recommendations.length === 0 && (
          <p className="text-center text-slate-500 mt-6">
            Pas encore de recommandations. Explorez des produits pour améliorer
            vos suggestions !
          </p>
        )}
      </div>
    );
  }

  // ---- Login / Register form ----
  return (
    <div className="max-w-md mx-auto px-4 py-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <div className="text-center mb-6">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center text-white mx-auto mb-4">
            <FiUser className="w-7 h-7" />
          </div>
          <h1 className="text-2xl font-bold text-slate-900">
            {isLogin ? "Connexion" : "Créer un compte"}
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            {isLogin
              ? "Accédez à vos recommandations personnalisées"
              : "Rejoignez ECommerce-IA"}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              <FiUser className="inline w-4 h-4 mr-1" />
              Nom d&apos;utilisateur
            </label>
            <input
              type="text"
              value={form.username}
              onChange={(e) =>
                setForm({ ...form, username: e.target.value })
              }
              className="input-field"
              required
            />
          </div>
          {!isLogin && (
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                <FiMail className="inline w-4 h-4 mr-1" />
                Email
              </label>
              <input
                type="email"
                value={form.email}
                onChange={(e) =>
                  setForm({ ...form, email: e.target.value })
                }
                className="input-field"
                required
              />
            </div>
          )}
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              <FiLock className="inline w-4 h-4 mr-1" />
              Mot de passe
            </label>
            <input
              type="password"
              value={form.password}
              onChange={(e) =>
                setForm({ ...form, password: e.target.value })
              }
              className="input-field"
              required
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            <FiLogIn className="w-4 h-4" />
            {loading
              ? "Chargement..."
              : isLogin
              ? "Se connecter"
              : "Créer le compte"}
          </button>
        </form>

        <div className="text-center mt-5">
          <button
            onClick={() => setIsLogin(!isLogin)}
            className="text-sm text-primary-600 hover:underline"
          >
            {isLogin
              ? "Pas de compte ? Inscrivez-vous"
              : "Déjà un compte ? Connectez-vous"}
          </button>
        </div>
      </motion.div>
    </div>
  );
}
