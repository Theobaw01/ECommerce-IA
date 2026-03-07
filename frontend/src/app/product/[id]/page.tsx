"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { getProduct, getSimilarProducts, type Product } from "@/services/api";
import ProductCard from "@/components/ui/ProductCard";
import Spinner from "@/components/ui/Spinner";
import { FiShoppingCart, FiStar, FiArrowLeft, FiHeart } from "react-icons/fi";
import { useCartStore } from "@/stores/cartStore";
import Link from "next/link";
import toast from "react-hot-toast";
import { motion } from "framer-motion";

export default function ProductPage() {
  const params = useParams();
  const id = Number(params.id);
  const [product, setProduct] = useState<Product | null>(null);
  const [similar, setSimilar] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const addItem = useCartStore((s) => s.addItem);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    Promise.all([getProduct(id), getSimilarProducts(id)])
      .then(([prod, sims]) => {
        setProduct(prod);
        setSimilar(sims);
      })
      .catch((err) => {
        console.error(err);
        toast.error("Produit introuvable");
      })
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <Spinner text="Chargement du produit..." />;
  if (!product) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-20 text-center">
        <p className="text-slate-500">Produit introuvable.</p>
        <Link href="/search" className="btn-primary mt-4 inline-block">
          Retour à la recherche
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <Link
        href="/search"
        className="inline-flex items-center gap-2 text-sm text-slate-500 hover:text-primary-600 transition-colors mb-8"
      >
        <FiArrowLeft className="w-4 h-4" />
        Retour à la recherche
      </Link>

      <div className="grid lg:grid-cols-2 gap-10">
        {/* Image */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="relative rounded-2xl overflow-hidden bg-slate-50 border border-slate-100"
        >
          {product.image_url ? (
            <img
              src={product.image_url}
              alt={product.nom}
              className="w-full h-96 lg:h-[500px] object-contain"
            />
          ) : (
            <div className="w-full h-96 flex items-center justify-center text-slate-300">
              <FiShoppingCart className="w-20 h-20" />
            </div>
          )}
        </motion.div>

        {/* Details */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex flex-col"
        >
          <span className="badge bg-primary-100 text-primary-700 w-fit mb-3">
            {product.categorie}
          </span>
          <h1 className="text-3xl font-bold text-slate-900 mb-3">
            {product.nom}
          </h1>

          {product.note_moyenne > 0 && (
            <div className="flex items-center gap-2 mb-4">
              <div className="flex items-center gap-1">
                {Array.from({ length: 5 }, (_, i) => (
                  <FiStar
                    key={i}
                    className={`w-4 h-4 ${
                      i < Math.round(product.note_moyenne)
                        ? "text-amber-400 fill-amber-400"
                        : "text-slate-200"
                    }`}
                  />
                ))}
              </div>
              <span className="text-sm text-slate-500">
                {product.note_moyenne.toFixed(1)} / 5
              </span>
            </div>
          )}

          <p className="text-slate-600 leading-relaxed mb-6">
            {product.description || "Aucune description disponible."}
          </p>

          <div className="text-4xl font-bold text-slate-900 mb-6">
            {product.prix.toFixed(2)} €
          </div>

          <div className="flex gap-3 mt-auto">
            <button
              onClick={() => {
                addItem(product);
                toast.success("Ajouté au panier !");
              }}
              className="btn-primary flex-1 flex items-center justify-center gap-2 text-lg py-3"
            >
              <FiShoppingCart className="w-5 h-5" />
              Ajouter au panier
            </button>
            <button className="btn-secondary p-3">
              <FiHeart className="w-5 h-5" />
            </button>
          </div>
        </motion.div>
      </div>

      {/* Similar products */}
      {similar.length > 0 && (
        <section className="mt-16">
          <h2 className="text-2xl font-bold text-slate-900 mb-6">
            Produits similaires
          </h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {similar.map((prod) => (
              <ProductCard key={prod.id} product={prod} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
