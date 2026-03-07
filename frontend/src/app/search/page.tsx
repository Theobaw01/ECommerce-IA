"use client";

import { useState } from "react";
import { classifyImage, searchProducts, type ClassificationResult, type Product } from "@/services/api";
import ImageDropzone from "@/components/ui/ImageDropzone";
import ConfidenceBar from "@/components/ui/ConfidenceBar";
import ProductCard from "@/components/ui/ProductCard";
import Spinner from "@/components/ui/Spinner";
import { FiSearch, FiCamera, FiZap } from "react-icons/fi";
import { motion, AnimatePresence } from "framer-motion";
import toast from "react-hot-toast";

export default function SearchPage() {
  const [mode, setMode] = useState<"visual" | "text">("visual");
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [classification, setClassification] = useState<ClassificationResult | null>(null);
  const [products, setProducts] = useState<Product[]>([]);

  // ---- Classification visuelle ----
  const handleImageSelected = async (file: File) => {
    setIsLoading(true);
    setClassification(null);
    setProducts([]);
    try {
      const result = await classifyImage(file);
      setClassification(result);
      // Rechercher des produits similaires
      const prods = await searchProducts(result.categorie);
      setProducts(prods);
      toast.success(`Catégorie détectée : ${result.categorie}`);
    } catch (err) {
      console.error(err);
      toast.error("Erreur lors de la classification");
    } finally {
      setIsLoading(false);
    }
  };

  // ---- Recherche textuelle ----
  const handleTextSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setIsLoading(true);
    setClassification(null);
    setProducts([]);
    try {
      const prods = await searchProducts(query);
      setProducts(prods);
      if (prods.length === 0) toast("Aucun produit trouvé", { icon: "🔍" });
    } catch (err) {
      console.error(err);
      toast.error("Erreur de recherche");
    } finally {
      setIsLoading(false);
    }
  };

  const barColors = [
    "bg-primary-500",
    "bg-cyan-500",
    "bg-purple-500",
    "bg-pink-500",
    "bg-amber-500",
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* Header */}
      <div className="text-center mb-10">
        <h1 className="text-3xl sm:text-4xl font-bold text-slate-900 mb-3">
          Recherche <span className="text-primary-600">Intelligente</span>
        </h1>
        <p className="text-slate-500 max-w-xl mx-auto">
          Uploadez une photo pour identifier un produit, ou recherchez par texte.
        </p>
      </div>

      {/* Mode toggle */}
      <div className="flex justify-center gap-2 mb-8">
        <button
          onClick={() => setMode("visual")}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all ${
            mode === "visual"
              ? "bg-primary-600 text-white shadow-md"
              : "bg-white text-slate-600 border hover:bg-slate-50"
          }`}
        >
          <FiCamera className="w-4 h-4" />
          Recherche Visuelle
        </button>
        <button
          onClick={() => setMode("text")}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all ${
            mode === "text"
              ? "bg-primary-600 text-white shadow-md"
              : "bg-white text-slate-600 border hover:bg-slate-50"
          }`}
        >
          <FiSearch className="w-4 h-4" />
          Recherche Texte
        </button>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Left: input */}
        <div>
          <AnimatePresence mode="wait">
            {mode === "visual" ? (
              <motion.div
                key="visual"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
              >
                <ImageDropzone
                  onFileSelected={handleImageSelected}
                  isLoading={isLoading}
                />
              </motion.div>
            ) : (
              <motion.div
                key="text"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
              >
                <form onSubmit={handleTextSearch} className="relative">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ex: robe bleue, chaussures sport, sac à main..."
                    className="input-field pr-14 text-lg"
                  />
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-2.5 rounded-xl bg-primary-600 text-white hover:bg-primary-700 transition-colors disabled:opacity-50"
                  >
                    <FiSearch className="w-5 h-5" />
                  </button>
                </form>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Classification results */}
          {classification && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="card mt-6"
            >
              <div className="flex items-center gap-2 mb-4">
                <FiZap className="w-5 h-5 text-primary-600" />
                <h3 className="font-bold text-lg">Résultats IA</h3>
                <span className="badge bg-green-100 text-green-700 ml-auto">
                  {classification.temps_inference.toFixed(0)} ms
                </span>
              </div>

              <div className="mb-4 p-4 rounded-xl bg-primary-50 border border-primary-100">
                <p className="text-sm text-primary-600 font-medium">
                  Catégorie principale
                </p>
                <p className="text-2xl font-bold text-primary-800 capitalize">
                  {classification.categorie}
                </p>
                <p className="text-sm text-primary-500 mt-1">
                  Confiance : {(classification.confiance * 100).toFixed(1)}%
                </p>
              </div>

              <div className="space-y-3">
                <p className="text-sm font-medium text-slate-500">Top 5 prédictions</p>
                {classification.top5.map((pred, i) => (
                  <ConfidenceBar
                    key={pred.categorie}
                    label={pred.categorie}
                    value={pred.confiance}
                    rank={i + 1}
                    colorClass={barColors[i]}
                  />
                ))}
              </div>
            </motion.div>
          )}
        </div>

        {/* Right: products */}
        <div>
          {isLoading && <Spinner text="Recherche en cours..." />}
          {!isLoading && products.length > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <h3 className="font-bold text-lg text-slate-800 mb-4">
                {products.length} produit{products.length > 1 ? "s" : ""} trouvé
                {products.length > 1 ? "s" : ""}
              </h3>
              <div className="grid sm:grid-cols-2 gap-4">
                {products.map((prod) => (
                  <ProductCard key={prod.id} product={prod} />
                ))}
              </div>
            </motion.div>
          )}
          {!isLoading && products.length === 0 && !classification && (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <FiSearch className="w-16 h-16 text-slate-200 mb-4" />
              <p className="text-slate-400">
                Uploadez une image ou tapez une recherche pour découvrir des
                produits.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
