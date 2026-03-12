import { useState } from "react";

const KEY = "solvro_user_id";

export function useUserId(): string {
    const [userId] = useState<string>(() => {
        const stored = localStorage.getItem(KEY);
        if (stored) return stored;
        const id = crypto.randomUUID();
        localStorage.setItem(KEY, id);
        return id;
    });

    return userId;
}
