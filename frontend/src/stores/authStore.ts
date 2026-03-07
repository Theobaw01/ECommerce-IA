/**
 * ECommerce-IA — Auth Store (Zustand)
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AuthState {
  token: string | null;
  userId: number | null;
  username: string | null;
  isAuthenticated: boolean;
  setAuth: (token: string, userId: number, username: string) => void;
  clearAuth: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      userId: null,
      username: null,
      isAuthenticated: false,

      setAuth: (token, userId, username) =>
        set({ token, userId, username, isAuthenticated: true }),

      clearAuth: () =>
        set({
          token: null,
          userId: null,
          username: null,
          isAuthenticated: false,
        }),
    }),
    { name: "ecommerce-ia-auth" }
  )
);
