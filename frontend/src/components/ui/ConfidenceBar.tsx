"use client";

import { motion } from "framer-motion";
import clsx from "clsx";

interface ConfidenceBarProps {
  label: string;
  value: number; // 0 à 1
  rank?: number;
  colorClass?: string;
}

export default function ConfidenceBar({
  label,
  value,
  rank,
  colorClass = "bg-primary-500",
}: ConfidenceBarProps) {
  const percentage = Math.round(value * 100);

  return (
    <div className="flex items-center gap-3">
      {rank !== undefined && (
        <span className="text-xs font-bold text-slate-400 w-5 text-right">
          #{rank}
        </span>
      )}
      <div className="flex-1">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm font-medium text-slate-700 truncate max-w-[200px]">
            {label}
          </span>
          <span className="text-sm font-bold text-slate-800">{percentage}%</span>
        </div>
        <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className={clsx("h-full rounded-full", colorClass)}
          />
        </div>
      </div>
    </div>
  );
}
