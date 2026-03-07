"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { FiUploadCloud, FiImage, FiX } from "react-icons/fi";
import { motion } from "framer-motion";
import Image from "next/image";

interface ImageDropzoneProps {
  onFileSelected: (file: File) => void;
  isLoading?: boolean;
}

export default function ImageDropzone({
  onFileSelected,
  isLoading = false,
}: ImageDropzoneProps) {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;
      setPreview(URL.createObjectURL(file));
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10 MB
    disabled: isLoading,
  });

  const clearPreview = () => {
    setPreview(null);
  };

  return (
    <div className="w-full">
      {preview ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative rounded-2xl overflow-hidden border-2 border-primary-200 bg-primary-50"
        >
          <Image
            src={preview}
            alt="Image uploadée"
            width={400}
            height={400}
            className="w-full h-64 object-contain bg-white"
            unoptimized
          />
          <button
            onClick={clearPreview}
            className="absolute top-3 right-3 p-2 rounded-full bg-white/90 hover:bg-white shadow-md transition-colors"
          >
            <FiX className="w-4 h-4 text-slate-600" />
          </button>
          {isLoading && (
            <div className="absolute inset-0 bg-white/60 backdrop-blur-sm flex items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <div className="w-10 h-10 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin" />
                <span className="text-sm font-medium text-primary-700">
                  Analyse en cours...
                </span>
              </div>
            </div>
          )}
        </motion.div>
      ) : (
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all ${
            isDragActive
              ? "border-primary-500 bg-primary-50"
              : "border-slate-200 hover:border-primary-300 hover:bg-slate-50"
          }`}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-3">
            {isDragActive ? (
              <FiImage className="w-12 h-12 text-primary-500" />
            ) : (
              <FiUploadCloud className="w-12 h-12 text-slate-300" />
            )}
            <div>
              <p className="text-sm font-medium text-slate-600">
                {isDragActive
                  ? "Déposez l'image ici..."
                  : "Glissez une image ou cliquez pour parcourir"}
              </p>
              <p className="text-xs text-slate-400 mt-1">
                JPG, PNG, WebP — 10 Mo max
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
