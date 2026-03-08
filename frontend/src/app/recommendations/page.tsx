"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import toast from "react-hot-toast";
import {
  FiStar,
  FiMapPin,
  FiDollarSign,
  FiUsers,
  FiGrid,
  FiTrendingUp,
  FiRefreshCw,
  FiShoppingCart,
  FiHeart,
  FiBarChart2,
} from "react-icons/fi";
import api from "@/services/api";
import { useCartStore } from "@/stores/cartStore";

interface Recommendation {
  product_id: string;
  nom: string;
  categorie: string;
  prix: number;
  score_final: number;
  facteurs: {
    historique: number;
    similarite: number;
    geographique: number;
    prix: number;
  };
}

interface Product {
  id: string;
  nom: string;
  categorie: string;
  prix: number;
  marque: string;
  stock: number;
  image_url?: string;
}

export default function RecommendationsPage() {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(false);
  const [userId, setUserId] = useState("U0001");
  const [numRecs, setNumRecs] = useState(10);
  const [budgetMin, setBudgetMin] = useState(10);
  const [budgetMax, setBudgetMax] = useState(300);
  const [view, setView] = useState<"recs" | "catalog">("recs");
  const addToCart = useCartStore((s) => s.addItem);

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const { data } = await api.get(`/recommend/${userId}`, {
        params: { n: numRecs },
      });
      setRecommendations(data.recommendations || []);
      toast.success(
        `${data.recommendations?.length || 0} recommandations générées`
      );
    } catch {
      toast.error("Erreur lors du chargement des recommandations");
    } finally {
      setLoading(false);
    }
  };

  const fetchProducts = async () => {
    setLoading(true);
    try {
      const { data } = await api.get("/products", {
        params: {
          page: 1,
          limit: 20,
          prix_min: budgetMin,
          prix_max: budgetMax,
        },
      });
      setProducts(data.products || []);
    } catch {
      // Silently handle error for catalog
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecommendations();
    fetchProducts();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const ScoreBar = ({
    label,
    value,
    color,
    icon,
  }: {
    label: string;
    value: number;
    color: string;
    icon: React.ReactNode;
  }) => (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-slate-400 w-4">{icon}</span>
      <span className="text-slate-500 w-16 truncate">{label}</span>
      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.8 }}
          className={`h-full rounded-full ${color}`}
        />
      </div>
      <span className="text-slate-600 font-mono w-10 text-right">
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* Header */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-3 mb-3">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white">
            <FiStar className="w-6 h-6" />
          </div>
          <div className="text-left">
            <h1 className="text-2xl font-bold text-slate-900">
              Recommandations IA
            </h1>
            <p className="text-sm text-slate-500">
              Algorithme hybride : SVD + Content-Based + Géo + Prix
            </p>
          </div>
        </div>
      </div>

      {/* Contrôles */}
      <div className="card mb-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1 block">
              Utilisateur
            </label>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              placeholder="U0001"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1 block">
              Nombre
            </label>
            <input
              type="number"
              value={numRecs}
              onChange={(e) => setNumRecs(Number(e.target.value))}
              min={1}
              max={50}
              className="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1 block">
              Budget min (€)
            </label>
            <input
              type="number"
              value={budgetMin}
              onChange={(e) => setBudgetMin(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1 block">
              Budget max (€)
            </label>
            <input
              type="number"
              value={budgetMax}
              onChange={(e) => setBudgetMax(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-lg border border-slate-200 text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={fetchRecommendations}
              disabled={loading}
              className="w-full px-4 py-2 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 text-white font-medium text-sm hover:shadow-lg transition-all disabled:opacity-50 flex items-center justify-center gap-2"
            >
              <FiRefreshCw
                className={`w-4 h-4 ${loading ? "animate-spin" : ""}`}
              />
              Générer
            </button>
          </div>
        </div>

        {/* Légende des 4 facteurs */}
        <div className="mt-4 pt-4 border-t border-slate-100">
          <p className="text-xs text-slate-400 mb-2 font-medium">
            Score final = 40% Historique + 30% Similarité + 15% Géographique +
            15% Prix
          </p>
          <div className="flex flex-wrap gap-4 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <FiUsers className="w-3 h-3 text-blue-500" />
              Historique (SVD)
            </span>
            <span className="flex items-center gap-1">
              <FiGrid className="w-3 h-3 text-green-500" />
              Similarité (Content-Based)
            </span>
            <span className="flex items-center gap-1">
              <FiMapPin className="w-3 h-3 text-orange-500" />
              Géographique (Haversine)
            </span>
            <span className="flex items-center gap-1">
              <FiDollarSign className="w-3 h-3 text-purple-500" />
              Prix (Budget)
            </span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setView("recs")}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            view === "recs"
              ? "bg-purple-100 text-purple-700"
              : "text-slate-500 hover:bg-slate-100"
          }`}
        >
          <FiTrendingUp className="inline w-4 h-4 mr-1" />
          Recommandations ({recommendations.length})
        </button>
        <button
          onClick={() => {
            setView("catalog");
            fetchProducts();
          }}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            view === "catalog"
              ? "bg-purple-100 text-purple-700"
              : "text-slate-500 hover:bg-slate-100"
          }`}
        >
          <FiGrid className="inline w-4 h-4 mr-1" />
          Catalogue ({products.length})
        </button>
      </div>

      {/* Recommendations */}
      {view === "recs" && (
        <div className="space-y-4">
          {recommendations.length === 0 && !loading && (
            <div className="text-center py-16 text-slate-400">
              <FiStar className="w-12 h-12 mx-auto mb-4 opacity-30" />
              <p>
                Aucune recommandation.
                <br />
                Cliquez sur &quot;Générer&quot; pour obtenir des suggestions
                personnalisées.
              </p>
            </div>
          )}

          {loading && (
            <div className="text-center py-16">
              <div className="w-10 h-10 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin mx-auto mb-4" />
              <p className="text-slate-400 text-sm">
                Calcul des recommandations...
              </p>
            </div>
          )}

          {recommendations.map((rec, i) => (
            <motion.div
              key={rec.product_id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              className="card hover:shadow-lg transition-shadow"
            >
              <div className="flex flex-col lg:flex-row gap-4">
                {/* Info produit */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h3 className="font-bold text-slate-900">{rec.nom}</h3>
                      <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full">
                        {rec.categorie}
                      </span>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-purple-600">
                        {rec.prix.toFixed(2)}€
                      </div>
                      <div className="text-xs text-slate-400">
                        Score: {(rec.score_final * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {/* Score global */}
                  <div className="mb-3">
                    <div className="flex items-center gap-2 mb-1">
                      <FiBarChart2 className="w-4 h-4 text-purple-500" />
                      <span className="text-sm font-medium text-slate-700">
                        Score final
                      </span>
                    </div>
                    <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${rec.score_final * 100}%` }}
                        transition={{ duration: 1 }}
                        className="h-full rounded-full bg-gradient-to-r from-purple-500 to-pink-500"
                      />
                    </div>
                  </div>

                  {/* 4 facteurs */}
                  <div className="space-y-1.5">
                    <ScoreBar
                      label="Historique"
                      value={rec.facteurs.historique}
                      color="bg-blue-500"
                      icon={<FiUsers />}
                    />
                    <ScoreBar
                      label="Similarité"
                      value={rec.facteurs.similarite}
                      color="bg-green-500"
                      icon={<FiGrid />}
                    />
                    <ScoreBar
                      label="Géo"
                      value={rec.facteurs.geographique}
                      color="bg-orange-500"
                      icon={<FiMapPin />}
                    />
                    <ScoreBar
                      label="Prix"
                      value={rec.facteurs.prix}
                      color="bg-purple-500"
                      icon={<FiDollarSign />}
                    />
                  </div>
                </div>

                {/* Actions */}
                <div className="flex lg:flex-col gap-2 lg:justify-center">
                  <button
                    onClick={() => {
                      addToCart({
                        id: rec.product_id,
                        nom: rec.nom,
                        prix: rec.prix,
                        image_url: "",
                        quantite: 1,
                      });
                      toast.success(`${rec.nom} ajouté au panier`);
                    }}
                    className="flex-1 lg:flex-none px-4 py-2 rounded-lg bg-purple-600 text-white text-sm font-medium hover:bg-purple-700 transition-colors flex items-center justify-center gap-2"
                  >
                    <FiShoppingCart className="w-4 h-4" />
                    Ajouter
                  </button>
                  <button
                    className="flex-1 lg:flex-none px-4 py-2 rounded-lg border border-slate-200 text-slate-600 text-sm hover:bg-slate-50 transition-colors flex items-center justify-center gap-2"
                  >
                    <FiHeart className="w-4 h-4" />
                    Favoris
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* Catalogue */}
      {view === "catalog" && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {products.map((product, i) => (
            <motion.div
              key={product.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.03 }}
              className="card hover:shadow-lg transition-shadow"
            >
              <div className="w-full h-32 bg-gradient-to-br from-slate-100 to-slate-200 rounded-lg mb-3 flex items-center justify-center">
                <FiGrid className="w-8 h-8 text-slate-300" />
              </div>
              <h3 className="font-semibold text-slate-900 text-sm mb-1 truncate">
                {product.nom}
              </h3>
              <p className="text-xs text-slate-400 mb-2">{product.categorie}</p>
              <div className="flex items-center justify-between">
                <span className="font-bold text-purple-600">
                  {product.prix.toFixed(2)}€
                </span>
                <span
                  className={`text-xs px-2 py-0.5 rounded-full ${
                    product.stock > 0
                      ? "bg-green-100 text-green-700"
                      : "bg-red-100 text-red-700"
                  }`}
                >
                  {product.stock > 0 ? `${product.stock} en stock` : "Rupture"}
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* Architecture */}
      <div className="mt-12 card bg-slate-50 border-slate-200">
        <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
          <FiBarChart2 className="text-purple-500" />
          Architecture du Système de Recommandation
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h4 className="font-semibold text-blue-600 mb-1">
              Facteur 1 — Collaborative Filtering (SVD)
            </h4>
            <p className="text-slate-500">
              Décomposition en Valeurs Singulières (scikit-surprise).
              50 facteurs latents, 20 epochs. Prédit les préférences
              basées sur les utilisateurs similaires.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-green-600 mb-1">
              Facteur 2 — Content-Based Filtering
            </h4>
            <p className="text-slate-500">
              Similarité cosine sur les features produits
              (catégorie one-hot, prix normalisé, marque).
              Matrice de similarité pré-calculée.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-orange-600 mb-1">
              Facteur 3 — Distance Géographique
            </h4>
            <p className="text-slate-500">
              Formule de Haversine pour la distance sphérique.
              Score décroissant pour les vendeurs à plus de 50 km.
              Pénalité exponentielle au-delà.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-purple-600 mb-1">
              Facteur 4 — Budget Utilisateur
            </h4>
            <p className="text-slate-500">
              Score basé sur la fourchette de prix de l&apos;utilisateur.
              Bonus pour les promotions actives. Estimation du budget
              par percentiles de l&apos;historique.
            </p>
          </div>
        </div>
        <div className="mt-4 pt-4 border-t border-slate-200">
          <p className="text-xs text-slate-400">
            Métriques : Precision@K, Recall@K, NDCG@K, Coverage | Bibliothèque :
            scikit-surprise (SVD), scikit-learn (cosine_similarity)
          </p>
        </div>
      </div>
    </div>
  );
}
