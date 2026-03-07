"use client";

import { FiLoader } from "react-icons/fi";

export default function Spinner({ text = "Chargement..." }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4">
      <FiLoader className="w-8 h-8 text-primary-500 animate-spin" />
      <p className="text-sm text-slate-500">{text}</p>
    </div>
  );
}
