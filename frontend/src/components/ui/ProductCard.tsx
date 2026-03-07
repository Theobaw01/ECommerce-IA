"use client";

import { Product } from "@/services/api";
import { FiShoppingCart, FiStar } from "react-icons/fi";
import { motion } from "framer-motion";
import Link from "next/link";
import { useCartStore } from "@/stores/cartStore";
import toast from "react-hot-toast";

interface ProductCardProps {
  product: Product;
  score?: number;
  reason?: string;
}

export default function ProductCard({ product, score, reason }: ProductCardProps) {
  const addItem = useCartStore((s) => s.addItem);

  const handleAddToCart = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    addItem(product);
    toast.success(`${product.nom} ajouté au panier`);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="card group overflow-hidden p-0"
    >
      <Link href={`/product/${product.id}`}>
        {/* Image */}
        <div className="relative h-52 bg-slate-50 overflow-hidden">
          {product.image_url ? (
            <img
              src={product.image_url}
              alt={product.nom}
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-slate-300">
              <FiShoppingCart className="w-12 h-12" />
            </div>
          )}
          {score !== undefined && (
            <span className="absolute top-3 right-3 badge bg-primary-100 text-primary-700">
              {Math.round(score * 100)}% match
            </span>
          )}
        </div>

        {/* Info */}
        <div className="p-4">
          <p className="text-xs font-medium text-primary-600 uppercase tracking-wide mb-1">
            {product.categorie}
          </p>
          <h3 className="font-semibold text-slate-800 truncate mb-1">
            {product.nom}
          </h3>
          {reason && (
            <p className="text-xs text-slate-400 truncate mb-2">{reason}</p>
          )}
          <div className="flex items-center justify-between mt-2">
            <span className="text-lg font-bold text-slate-900">
              {product.prix.toFixed(2)} €
            </span>
            <div className="flex items-center gap-1">
              {product.note_moyenne > 0 && (
                <span className="flex items-center gap-1 text-sm text-amber-500">
                  <FiStar className="w-3.5 h-3.5 fill-amber-400" />
                  {product.note_moyenne.toFixed(1)}
                </span>
              )}
            </div>
          </div>
          <button
            onClick={handleAddToCart}
            className="mt-3 w-full btn-primary text-sm py-2 flex items-center justify-center gap-2"
          >
            <FiShoppingCart className="w-4 h-4" />
            Ajouter au panier
          </button>
        </div>
      </Link>
    </motion.div>
  );
}
