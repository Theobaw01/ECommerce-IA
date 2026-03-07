import Link from "next/link";
import { FiGithub, FiMail } from "react-icons/fi";

export default function Footer() {
  return (
    <footer className="bg-slate-900 text-slate-400 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-3 gap-8">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center text-white font-bold text-xs">
                IA
              </div>
              <span className="font-bold text-white">
                ECommerce<span className="text-primary-400">-IA</span>
              </span>
            </div>
            <p className="text-sm leading-relaxed">
              Plateforme e-commerce propulsée par l&apos;intelligence artificielle.
              Classification visuelle, recommandations et chatbot intégrés.
            </p>
          </div>
          <div>
            <h4 className="text-white font-semibold mb-4">Modules IA</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/search" className="hover:text-white transition-colors">
                  Classification EfficientNet-B4
                </Link>
              </li>
              <li>
                <Link href="/search" className="hover:text-white transition-colors">
                  Recommandations Hybrides
                </Link>
              </li>
              <li>
                <Link href="/chat" className="hover:text-white transition-colors">
                  Chatbot RAG Mistral-7B
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-semibold mb-4">Projet</h4>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <FiMail className="w-4 h-4" />
                Réalisé chez SAHELYS
              </li>
              <li className="flex items-center gap-2">
                <FiGithub className="w-4 h-4" />
                <span>BAWANA Théodore</span>
              </li>
              <li className="text-xs mt-4">
                Stack : Next.js · FastAPI · PyTorch · LangChain · Docker
              </li>
            </ul>
          </div>
        </div>
        <div className="border-t border-slate-800 mt-10 pt-6 text-center text-xs">
          © {new Date().getFullYear()} ECommerce-IA — Tous droits réservés
        </div>
      </div>
    </footer>
  );
}
