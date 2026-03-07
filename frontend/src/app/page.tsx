"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { FiSearch, FiStar, FiMessageCircle, FiArrowRight, FiZap, FiShield, FiTrendingUp } from "react-icons/fi";

const features = [
  {
    icon: <FiSearch className="w-7 h-7" />,
    title: "Recherche Visuelle",
    description:
      "Uploadez une photo d'un produit et notre IA identifie instantanément sa catégorie parmi 500 classes avec 94% de précision.",
    href: "/search",
    color: "from-blue-500 to-cyan-500",
  },
  {
    icon: <FiStar className="w-7 h-7" />,
    title: "Recommandations IA",
    description:
      "Algorithme hybride combinant historique, similarité produits, géolocalisation et budget pour des suggestions ultra-pertinentes.",
    href: "/search",
    color: "from-purple-500 to-pink-500",
  },
  {
    icon: <FiMessageCircle className="w-7 h-7" />,
    title: "Chatbot Intelligent",
    description:
      "Assistant virtuel RAG propulsé par Mistral-7B, capable de répondre à toutes vos questions sur les produits et services.",
    href: "/chat",
    color: "from-orange-500 to-red-500",
  },
];

const stats = [
  { value: "500+", label: "Catégories", icon: <FiZap /> },
  { value: "94%", label: "Précision IA", icon: <FiTrendingUp /> },
  { value: "3000+", label: "Produits", icon: <FiShield /> },
];

export default function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden py-20 lg:py-32">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-600 via-primary-700 to-accent-700" />
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="text-center"
          >
            <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/10 text-white/90 text-sm font-medium mb-6 backdrop-blur-sm border border-white/20">
              <FiZap className="w-4 h-4" />
              Propulsé par EfficientNet-B4 & Mistral-7B
            </span>
            <h1 className="text-4xl sm:text-5xl lg:text-7xl font-bold text-white mb-6 leading-tight">
              Shopping{" "}
              <span className="bg-gradient-to-r from-cyan-300 to-pink-300 bg-clip-text text-transparent">
                Intelligent
              </span>
            </h1>
            <p className="text-lg sm:text-xl text-white/80 max-w-2xl mx-auto mb-10">
              Découvrez une expérience d&apos;achat révolutionnée par l&apos;intelligence
              artificielle. Classification visuelle, recommandations personnalisées
              et assistant IA à votre service.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/search"
                className="btn-primary text-lg px-8 py-3 bg-white text-primary-700 hover:bg-slate-100 inline-flex items-center gap-2"
              >
                Explorer
                <FiArrowRight />
              </Link>
              <Link
                href="/chat"
                className="btn-secondary text-lg px-8 py-3 border-white/30 text-white hover:bg-white/10 inline-flex items-center gap-2"
              >
                <FiMessageCircle />
                Chatbot IA
              </Link>
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.6 }}
            className="mt-16 grid grid-cols-3 gap-8 max-w-xl mx-auto"
          >
            {stats.map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-3xl sm:text-4xl font-bold text-white">
                  {stat.value}
                </div>
                <div className="text-sm text-white/60 mt-1">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-slate-900 mb-4">
            3 Modules IA Intégrés
          </h2>
          <p className="text-lg text-slate-500 max-w-2xl mx-auto">
            Chaque module est entraîné sur le dataset Products-10K et optimisé
            pour une expérience utilisateur fluide.
          </p>
        </div>
        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feat, i) => (
            <motion.div
              key={feat.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.15, duration: 0.5 }}
              viewport={{ once: true }}
            >
              <Link href={feat.href} className="card group block h-full">
                <div
                  className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${feat.color} flex items-center justify-center text-white mb-5 group-hover:scale-110 transition-transform`}
                >
                  {feat.icon}
                </div>
                <h3 className="text-xl font-bold mb-3 group-hover:text-primary-600 transition-colors">
                  {feat.title}
                </h3>
                <p className="text-slate-500 leading-relaxed">
                  {feat.description}
                </p>
                <span className="inline-flex items-center gap-1 text-primary-600 font-medium mt-4 text-sm">
                  Découvrir <FiArrowRight className="w-4 h-4" />
                </span>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Architecture */}
      <section className="py-20 bg-slate-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-slate-900 mb-4">
            Architecture Technique
          </h2>
          <p className="text-slate-500 mb-10 max-w-2xl mx-auto">
            Stack moderne et performant, conçu pour la scalabilité.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 max-w-3xl mx-auto">
            {[
              { name: "Next.js 14", desc: "Frontend" },
              { name: "FastAPI", desc: "Backend" },
              { name: "PyTorch", desc: "Deep Learning" },
              { name: "LangChain", desc: "RAG Chatbot" },
              { name: "PostgreSQL", desc: "Base de données" },
              { name: "ChromaDB", desc: "Vecteurs" },
              { name: "Docker", desc: "Déploiement" },
              { name: "TailwindCSS", desc: "UI / UX" },
            ].map((tech) => (
              <div key={tech.name} className="card text-center py-4 px-3">
                <div className="font-semibold text-slate-800">{tech.name}</div>
                <div className="text-xs text-slate-400 mt-1">{tech.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
