"use client";

import { useCartStore } from "@/stores/cartStore";
import { FiTrash2, FiMinus, FiPlus, FiShoppingBag, FiArrowLeft } from "react-icons/fi";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import toast from "react-hot-toast";

export default function CartPage() {
  const { items, removeItem, updateQuantity, clearCart, total } = useCartStore();

  if (items.length === 0) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-20 text-center">
        <FiShoppingBag className="w-20 h-20 text-slate-200 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-slate-800 mb-2">
          Votre panier est vide
        </h2>
        <p className="text-slate-500 mb-6">
          Découvrez nos produits grâce à la recherche visuelle IA.
        </p>
        <Link href="/search" className="btn-primary inline-flex items-center gap-2">
          <FiArrowLeft className="w-4 h-4" />
          Explorer les produits
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold text-slate-900">
          Panier ({items.length})
        </h1>
        <button
          onClick={() => {
            clearCart();
            toast.success("Panier vidé");
          }}
          className="text-sm text-red-500 hover:text-red-700 flex items-center gap-1"
        >
          <FiTrash2 className="w-4 h-4" />
          Vider
        </button>
      </div>

      <div className="space-y-4">
        <AnimatePresence>
          {items.map((item) => (
            <motion.div
              key={item.product.id}
              layout
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: -100 }}
              className="card flex items-center gap-5 p-4"
            >
              {/* Image */}
              <div className="w-20 h-20 rounded-xl bg-slate-50 overflow-hidden flex-shrink-0">
                {item.product.image_url ? (
                  <img
                    src={item.product.image_url}
                    alt={item.product.nom}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-slate-300">
                    <FiShoppingBag className="w-8 h-8" />
                  </div>
                )}
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-slate-800 truncate">
                  {item.product.nom}
                </h3>
                <p className="text-xs text-primary-600">{item.product.categorie}</p>
                <p className="text-sm font-bold text-slate-900 mt-1">
                  {item.product.prix.toFixed(2)} €
                </p>
              </div>

              {/* Quantity */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() =>
                    updateQuantity(item.product.id, item.quantity - 1)
                  }
                  className="w-8 h-8 rounded-lg border border-slate-200 flex items-center justify-center hover:bg-slate-50"
                >
                  <FiMinus className="w-3.5 h-3.5" />
                </button>
                <span className="w-8 text-center font-medium">
                  {item.quantity}
                </span>
                <button
                  onClick={() =>
                    updateQuantity(item.product.id, item.quantity + 1)
                  }
                  className="w-8 h-8 rounded-lg border border-slate-200 flex items-center justify-center hover:bg-slate-50"
                >
                  <FiPlus className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Subtotal */}
              <div className="text-right min-w-[80px]">
                <p className="font-bold text-slate-900">
                  {(item.product.prix * item.quantity).toFixed(2)} €
                </p>
              </div>

              {/* Remove */}
              <button
                onClick={() => {
                  removeItem(item.product.id);
                  toast.success("Produit retiré");
                }}
                className="p-2 rounded-xl text-slate-400 hover:text-red-500 hover:bg-red-50 transition-colors"
              >
                <FiTrash2 className="w-4 h-4" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Total */}
      <div className="card mt-8 flex items-center justify-between">
        <div>
          <p className="text-sm text-slate-500">Total</p>
          <p className="text-3xl font-bold text-slate-900">
            {total().toFixed(2)} €
          </p>
        </div>
        <button className="btn-primary text-lg px-8 py-3">
          Commander
        </button>
      </div>
    </div>
  );
}
